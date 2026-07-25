"""Phase L — Feature Engineering Pipeline (Day-3 Stage-2, capstone).

Assembles every Day-3 phase into one fit/transform object. The FROZEN execution
order (Phase L of the roadmap):

    X (from Phase 0)
      -> FeatureBuilder            (Phase F: temporal / title / goal_per_day)
      -> TextPreprocessor          (Phase G: clean titles)
      |- numeric branch:  Transform(log) -> OutlierCap -> Scale   (Phases C/D/E)
      |- categorical branch: Encode (OHE + frequency)             (Phase B)
      |- text branch:     TF-IDF                                  (Phase H)
      -> Fusion (sparse hstack)    -> X_fused                     (Phase J)
      -> FeatureSelection                                         (Phase K)
    Final: X_fused_selected

**Why one object matters — structural leakage safety.** Throughout Day 3 we
enforced "fit on train, transform test" by hand. Here it is guaranteed *by
construction*: ``fit`` is the only method that calls each sub-handler's ``fit``;
``transform`` calls only their ``transform``. Once fitted on train, applying the
pipeline to the test set (or a single future campaign) cannot leak, because no
statistic is ever re-estimated. It is also **row-preserving** end to end.

Deliverables: a fitted :class:`FeaturePipeline` (``feature_pipeline.joblib``) and
``feature_metadata.json`` (the output contract + provenance).

Example
-------
>>> pipe = FeaturePipeline().fit(X_train, y_train)
>>> X_train_sel = pipe.transform(X_train)     # scipy.sparse.csr_matrix
>>> X_test_sel  = pipe.transform(X_test)      # same columns, fit-on-train only
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
import scipy.sparse as sp

from src.features.feature_builder import FeatureBuilder
from src.features.feature_selection import FeatureSelector
from src.features.fusion import FeatureFusion
from src.features.text_preprocessing import TextPreprocessor
from src.features.tfidf_vectorizer import TfidfVectorizerHandler
from src.preprocessing.encoding import EncodingHandler
from src.preprocessing.outliers import OutlierHandler
from src.preprocessing.scaling import ScalingHandler
from src.preprocessing.transformations import TransformationHandler
from src.utils.config import load_config
from src.utils.logger import get_logger


class FeaturePipeline:
    """End-to-end Day-3 feature pipeline (fit-on-train, row-preserving, sparse out).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.features.pipeline")

        d3 = self.config.get("day3", {})
        self.numeric_features: list[str] = list(d3.get("numeric_features", []))

        # Sub-handlers (share the same config).
        self.builder = FeatureBuilder(self.config, self.logger)
        self.transformer = TransformationHandler(self.config, self.logger)
        self.capper = OutlierHandler(self.config, self.logger)
        self.scaler = ScalingHandler(self.config, self.logger)
        self.encoder = EncodingHandler(self.config, self.logger)
        self.text = TextPreprocessor(self.config, self.logger)
        self.tfidf = TfidfVectorizerHandler(self.config, self.logger)
        self.fusion = FeatureFusion(self.config, self.logger)
        self.selector = FeatureSelector(self.config, self.logger)

        self.feature_names_: list[str] = []
        self._fitted = False

    # -------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y) -> "FeaturePipeline":
        """Fit every stage on **train**, in the frozen order."""
        self.logger.info("Fitting FeaturePipeline on %d rows...", len(X))

        # Phase F: build engineered columns first.
        Xf = self.builder.fit(X, y).transform(X)

        # Numeric branch (Phases C -> D -> E), fit in sequence.
        Xf = self.transformer.fit(Xf, y).transform(Xf)
        Xf = self.capper.fit(Xf, y).transform(Xf)
        Xf = self.scaler.fit(Xf, y).transform(Xf)
        numeric = Xf[self.numeric_features].astype(float)

        # Categorical branch (Phase B).
        categorical = self.encoder.fit(X, y).transform(X)

        # Text branch (Phase G -> H).
        Xg = self.text.fit(X, y).transform(X)
        text = self.tfidf.fit(Xg, y).transform(Xg)

        # Phase J: fuse the three blocks.
        fused = self.fusion.fuse(
            {
                "text": (text, self.tfidf.get_feature_names_out()),
                "categorical": (categorical, self.encoder.get_feature_names_out()),
                "numeric": (numeric, self.numeric_features),
            }
        )

        # Phase K: select on the fused matrix.
        self.selector.fit(fused.X, y, fused.feature_names)
        self.feature_names_ = self.selector.selected_names_
        self._fitted = True
        self.logger.info(
            "FeaturePipeline fitted — final feature count: %d.", len(self.feature_names_)
        )
        return self

    # -------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> sp.csr_matrix:
        """Apply the fitted pipeline to any partition (transform-only, no leakage)."""
        self._check_fitted()

        Xf = self.builder.transform(X)
        Xf = self.transformer.transform(Xf)
        Xf = self.capper.transform(Xf)
        Xf = self.scaler.transform(Xf)
        numeric = Xf[self.numeric_features].astype(float)

        categorical = self.encoder.transform(X)
        text = self.tfidf.transform(self.text.transform(X))

        fused = self.fusion.fuse(
            {
                "text": (text, self.tfidf.get_feature_names_out()),
                "categorical": (categorical, self.encoder.get_feature_names_out()),
                "numeric": (numeric, self.numeric_features),
            }
        )
        return self.selector.transform(fused.X)

    def fit_transform(self, X: pd.DataFrame, y) -> sp.csr_matrix:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> list[str]:
        self._check_fitted()
        return list(self.feature_names_)

    # -------------------------------------------------------------- persistence
    def save(
        self,
        pipeline_path: str | Path = "artifacts/feature_pipeline/feature_pipeline.joblib",
        metadata_path: str | Path = "artifacts/feature_pipeline/feature_metadata.json",
    ) -> tuple[Path, Path]:
        self._check_fitted()
        pipeline_path, metadata_path = Path(pipeline_path), Path(metadata_path)
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, pipeline_path)
        metadata_path.write_text(json.dumps(self.build_metadata(), indent=2), encoding="utf-8")
        self.logger.info("Saved pipeline -> %s, metadata -> %s", pipeline_path, metadata_path)
        return pipeline_path, metadata_path

    @staticmethod
    def load(
        path: str | Path = "artifacts/feature_pipeline/feature_pipeline.joblib",
    ) -> "FeaturePipeline":
        return joblib.load(Path(path))

    def build_metadata(self) -> dict[str, Any]:
        self._check_fitted()
        sr = self.selector.result_
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_version": self.config.get("version", "unknown"),
            "n_features_out": len(self.feature_names_),
            "numeric_features": self.numeric_features,
            "selection": {
                "n_input": sr.n_input,
                "n_selected": sr.n_selected,
                "dropped_correlated": sr.dropped_correlated,
                "dropped_variance": len(sr.dropped_variance),
                "top_features": sr.top_features,
            },
            "tfidf": {
                "max_features": self.tfidf.max_features,
                "ngram_range": list(self.tfidf.ngram_range),
            },
            "feature_names": self.feature_names_,
        }

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("FeaturePipeline must be fit() before transform().")
