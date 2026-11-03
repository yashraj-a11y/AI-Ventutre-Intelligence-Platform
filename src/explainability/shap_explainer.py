"""Phase D — SHAP Explainability Engine (Day-5, Layer 5).

Phase B computed *mean* |SHAP| for a global ranking. Phase D promotes the SHAP
machinery into a reusable, first-class engine that returns the **full signed
per-feature, per-row** attribution matrix — the substrate every downstream phase
needs:

- **Phase E (local)** calls :meth:`SHAPExplainer.explain` on one transformed campaign
  → its exact positive/negative real-feature contributions.
- **Phase F (viz)** consumes the saved subsample matrix → beeswarm / waterfall plots.

Design decisions (carried over from the Phase-B investigation):

- **Interventional TreeSHAP, not the default fast path.** ``shap.TreeExplainer``'s
  default ``tree_path_dependent`` mode is numerically broken for this LightGBM
  (contributions in the thousands, additivity violated by ~1e6). We build the
  explainer with an explicit **background sample** and
  ``feature_perturbation="interventional"``, ``model_output="raw"`` (log-odds margin).
- **Additivity is asserted, never assumed.** Every ``explain`` verifies
  ``base_value + Σ shap ≈ model_margin`` within ``additivity_tol`` and raises if not.
  This is the unit test that caught the fast-path bug.
- **SHAP explains the UNCALIBRATED margin.** The shipped calibrated model is a
  *monotonic* squash on top of this margin, so the ranking / sign of every
  contribution is preserved; only the probability scale changes. Documented, not a bug.
- **Never densify 66k × 865.** We explain a ~2,000-row subsample for global work; a
  single row (local) is instant. The same fixed background powers both.

Example
-------
>>> engine = SHAPExplainer().load().build_explainer()
>>> res = engine.explain_subsample()        # global: ~2000 rows, saved to shap_values.pkl
>>> res.additivity_error < 1e-3
True
>>> one = engine.explain(X_row_dense)        # local: one campaign, instant
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import json
import numpy as np
import scipy.sparse as sp

from src.explainability.global_importance import _positive_class_shap
from src.utils.config import load_config
from src.utils.logger import get_logger


@dataclass
class ShapResult:
    """Signed SHAP attributions for a set of rows, in log-odds margin space.

    ``base_value + values.sum(axis=1) ≈ model.predict(X, raw_score=True)`` (asserted).
    """

    values: np.ndarray            # (n_rows, 865) signed SHAP for the "success" class
    base_value: float             # expected margin over the background (φ₀)
    feature_names: list[str]
    row_index: Optional[np.ndarray]   # indices into X_train_selected (None for ad-hoc rows)
    additivity_error: float
    meta: dict[str, Any] = field(default_factory=dict)

    def mean_abs(self) -> np.ndarray:
        """Mean |SHAP| per feature (global importance signal)."""
        return np.abs(self.values).mean(axis=0)


class SHAPExplainer:
    """Reusable interventional TreeSHAP engine for the base LightGBM.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.explain.shap")

        c = self.config.get("day5", {}).get("shap", {})
        self.model_path = Path(c.get("model_path", "artifacts/models/best_model.joblib"))
        self.fused_dir = Path(c.get("fused_dir", "artifacts/fused_features"))
        self.feature_meta_path = Path(
            c.get("feature_meta_path", "artifacts/feature_pipeline/feature_metadata.json")
        )
        self.subsample = int(c.get("subsample", 2000))
        self.background = int(c.get("background", 100))
        self.additivity_tol = float(c.get("additivity_tol", 1e-3))
        self.random_state = int(c.get("random_state", 42))
        self.out_path = Path(c.get("out_path", "artifacts/explainability/shap_values.pkl"))
        self.background_cache = Path(c.get("background_cache", "artifacts/explainability/shap_background.npz"))

        self.model_ = None
        self.feature_names_: list[str] = []
        self.explainer_ = None
        self.base_value_: Optional[float] = None
        self.result_: Optional[ShapResult] = None

    # ------------------------------------------------------------------ public
    def load(self) -> "SHAPExplainer":
        """Load the base LightGBM model and the 865 selected feature names."""
        self.model_ = joblib.load(self.model_path)
        meta = json.loads(self.feature_meta_path.read_text())
        self.feature_names_ = meta["feature_names"]
        if len(self.feature_names_) != self.model_.n_features_in_:
            raise ValueError(
                f"Name/model mismatch: {len(self.feature_names_)} names vs "
                f"{self.model_.n_features_in_} model features."
            )
        self.logger.info("Loaded %s (%d features).", type(self.model_).__name__,
                         self.model_.n_features_in_)
        return self

    def build_explainer(self, background: Optional[np.ndarray] = None) -> "SHAPExplainer":
        """Construct the interventional TreeSHAP explainer over a background sample.

        Parameters
        ----------
        background:
            Dense reference matrix. If ``None``, a ``self.background``-row random sample
            of the training matrix is used.
        """
        import shap  # heavy import, only when explaining

        if self.model_ is None:
            self.load()
        if background is None:
            background = self._background_rows()

        self.explainer_ = shap.TreeExplainer(
            self.model_,
            data=background,
            feature_perturbation="interventional",
            model_output="raw",           # log-odds margin; calibration is a squash on top
        )
        ev = self.explainer_.expected_value
        self.base_value_ = (
            float(ev[1]) if np.ndim(ev) > 0 and len(np.atleast_1d(ev)) > 1
            else float(np.atleast_1d(ev)[0])
        )
        self.logger.info("Interventional explainer built (bg=%d, base=%.4f).",
                         background.shape[0], self.base_value_)
        return self

    def explain(
        self,
        X_dense: np.ndarray,
        row_index: Optional[np.ndarray] = None,
        assert_additivity: bool = True,
    ) -> ShapResult:
        """Compute signed SHAP for the given DENSE rows (one or many).

        Used both for the global subsample and for a single campaign (Phase E).
        Raises if additivity is violated beyond ``additivity_tol``.
        """
        if self.explainer_ is None:
            self.build_explainer()
        X_dense = np.asarray(X_dense, dtype=float)
        if X_dense.ndim == 1:
            X_dense = X_dense.reshape(1, -1)

        sv = _positive_class_shap(self.explainer_.shap_values(X_dense))
        margin = self.model_.predict(X_dense, raw_score=True)
        add_err = float(np.abs(self.base_value_ + sv.sum(axis=1) - margin).max())
        if assert_additivity and add_err > self.additivity_tol:
            raise ValueError(
                f"SHAP additivity violated: max|base+Σshap-margin|={add_err:.3e} "
                f"> tol {self.additivity_tol:.1e}. Attributions untrustworthy."
            )

        return ShapResult(
            values=sv,
            base_value=self.base_value_,
            feature_names=self.feature_names_,
            row_index=row_index,
            additivity_error=add_err,
            meta={"model": type(self.model_).__name__, "space": "log_odds_margin",
                  "perturbation": "interventional", "n_rows": int(X_dense.shape[0])},
        )

    def explain_subsample(self, n: Optional[int] = None) -> ShapResult:
        """Explain a random ~n-row subsample of the training matrix (global work)."""
        n = n or self.subsample
        X_sub, idx = self._sample_rows(n)
        self.result_ = self.explain(X_sub, row_index=idx)
        self.logger.info("Explained %d-row subsample; additivity err=%.2e OK.",
                         X_sub.shape[0], self.result_.additivity_error)
        return self.result_

    def save(self, result: Optional[ShapResult] = None) -> Path:
        """Persist a ShapResult (defaults to the last subsample) to ``shap_values.pkl``."""
        res = result or self.result_
        if res is None:
            raise ValueError("Nothing to save — call explain_subsample() first.")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "values": res.values,
            "base_value": res.base_value,
            "feature_names": res.feature_names,
            "row_index": res.row_index,
            "additivity_error": res.additivity_error,
            "meta": res.meta,
        }
        with open(self.out_path, "wb") as fh:
            pickle.dump(payload, fh)
        self.logger.info("Saved SHAP values (%s) -> %s", res.values.shape, self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _background_rows(self) -> np.ndarray:
        """The fixed SHAP background: the same rows as ``_sample_rows(self.background)``, cached.

        Sampling 100 rows means loading the whole ~60 MB training matrix, which a server then keeps
        as resident memory after it is freed. The rows are saved once (~0.7 MB) and reused while the
        matrix file, the row count and the seed are unchanged, so explanations are identical.
        """
        src = self.fused_dir / "X_train_selected.npz"
        key = np.array([src.stat().st_size, self.background, self.random_state])
        if self.background_cache.exists():
            try:
                with np.load(self.background_cache) as z:
                    if np.array_equal(z["key"], key):
                        return z["rows"]
            except Exception:                           # unreadable cache -> rebuild
                pass
        rows = self._sample_rows(self.background)[0]
        try:
            self.background_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.background_cache, rows=rows, key=key)
        except OSError:                                 # read-only artifacts mount: just skip
            pass
        return rows

    def _sample_rows(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Densify a random ``n``-row sample of X_train_selected (never the full 66k)."""
        X = sp.load_npz(self.fused_dir / "X_train_selected.npz").tocsr()
        n = min(n, X.shape[0])
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(X.shape[0], size=n, replace=False)
        return X[idx].toarray(), idx
