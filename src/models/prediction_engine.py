"""Phase H — Prediction Engine Integration (Day-4, Architecture Layer 4).

The single entry point that turns a **raw campaign** into a full prediction. It
chains the artifacts every prior phase produced:

    raw campaign DataFrame
      → Day-3 FeaturePipeline.transform   (865-feature sparse matrix)
      → calibrated LightGBM .predict_proba (funding probability)
      → decision threshold                 (F1-optimal, from Phase F)
      → VentureScoreGenerator              (honest venture scores)
      → PredictionOutput

Wrapping the **Day-3 feature pipeline** (not re-implementing preprocessing) is
what guarantees train/serve consistency: the exact same transforms that produced
the training matrix are applied to a new campaign — one code path, no drift.

The engine expects raw campaigns with the Day-3 *feature* columns (name,
category, main_category, currency, deadline, goal, launched, country,
usd_goal_real, name_clean) — i.e. everything known at launch, minus the label.

Example
-------
>>> engine = PredictionEngine()
>>> out = engine.predict(raw_campaigns_df)[0]
>>> out.funding_probability, out.predicted_label
(0.72, 'successful')
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.features.pipeline import FeaturePipeline
from src.models.venture_scores import VentureScoreGenerator
from src.utils.config import load_config
from src.utils.logger import get_logger


@dataclass
class PredictionOutput:
    """A full prediction for one campaign."""

    funding_probability: float
    predicted_label: str          # "successful" | "failed"
    decision_threshold: float
    scores: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PredictionEngine:
    """Raw campaign → calibrated probability + venture scores (Layer 4)."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.models.prediction_engine")

        pe = self.config.get("day4", {}).get("prediction_engine", {})
        self.pipeline_path = pe.get("pipeline_path", "artifacts/feature_pipeline/feature_pipeline.joblib")
        self.model_path = pe.get("model_path", "artifacts/models/calibrated_model.joblib")
        self.calibration_path = pe.get("calibration_path", "artifacts/evaluations/calibration.json")
        self._threshold_override = pe.get("threshold", None)

        self.pipeline = FeaturePipeline.load(self.pipeline_path)
        self.model = joblib.load(self.model_path)
        self.scorer = VentureScoreGenerator(self.config, self.logger)
        self.threshold = self._resolve_threshold()
        self.logger.info(
            "PredictionEngine ready — model=%s, threshold=%.3f.",
            Path(self.model_path).name, self.threshold,
        )

    # ------------------------------------------------------------------ public
    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        """Calibrated funding probabilities for raw campaigns."""
        X = self.pipeline.transform(raw)               # Day-3 transforms (train/serve identical)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, raw: pd.DataFrame) -> list[PredictionOutput]:
        """Full prediction (probability + label + venture scores) per campaign."""
        proba = self.predict_proba(raw)
        outputs = []
        for p in proba:
            p = float(p)
            outputs.append(
                PredictionOutput(
                    funding_probability=round(p, 4),
                    predicted_label="successful" if p >= self.threshold else "failed",
                    decision_threshold=round(self.threshold, 4),
                    scores=self.scorer.score_campaign(p),
                )
            )
        return outputs

    # --------------------------------------------------------------- internals
    def _resolve_threshold(self) -> float:
        if self._threshold_override is not None:
            return float(self._threshold_override)
        p = Path(self.calibration_path)
        if p.exists():
            try:
                return float(json.loads(p.read_text())["threshold"]["f1_optimal"])
            except (KeyError, ValueError):
                pass
        self.logger.warning("No F1-optimal threshold found — defaulting to 0.5.")
        return 0.5
