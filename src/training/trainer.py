"""Day 16 · Phase G — Single-config trainer (fit → optional calibrate → evaluate).

The atom the reproducible pipeline is built from: train **one** model configuration end-to-end and
score it on the untouched test set. It **reuses** the Day-4 machinery — the incumbent path (calibrated
LightGBM) goes through :class:`ProbabilityCalibrator`, so re-running from the recorded ``best_config``
*reproduces* the production model rather than approximating it.

Two side-effect hazards in the reused calibrator are neutralised here (Day-16 C-5): its ``save()``
(which writes the **incumbent** ``calibrated_model.joblib``) is **never called**, and its plot path is
**redirected** to the versioned run directory. Evaluation is computed directly (not via
``ModelEvaluator``, which loads the incumbent) so nothing curated is touched.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.models.calibration import ProbabilityCalibrator
from src.models.data_split import ModelData
from src.utils.config import load_config
from src.utils.logger import get_logger

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _metrics_at_threshold(model, X, y, threshold: float) -> dict[str, float]:
    """The six metrics + Brier; label metrics at the chosen threshold, ranking metrics on probas."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "brier": round(float(brier_score_loss(y, proba)), 4),
    }


def build_estimator(family: str, params: Optional[dict[str, Any]] = None, random_state: int = 42):
    """Construct a candidate estimator (mirrors the Day-4 registries; used for the uncalibrated path)."""
    params = params or {}
    if family == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1, **params)
    if family == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=random_state, n_jobs=-1, tree_method="hist",
                             eval_metric="logloss", **params)
    if family == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=random_state, n_jobs=-1, **params)
    if family == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(solver="saga", max_iter=1000, random_state=random_state, **params)
    raise ValueError(f"unknown family {family!r}")


@dataclass
class TrainResult:
    family: str
    calibrated: bool
    threshold: float
    train_time_s: float
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)
    calibration: Optional[dict[str, Any]] = None
    model: Any = None


class SingleModelTrainer:
    """Train one configuration end-to-end; the incumbent path reproduces the calibrated LightGBM."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.training.trainer")
        self.random_state = int(self.config.get("day4", {}).get("tuning", {}).get("random_state", 42))

    def train(self, data: ModelData, *, family: str = "lightgbm", params: Optional[dict[str, Any]] = None,
              calibrate: bool = True, out_dir: Optional[Path] = None) -> TrainResult:
        """Fit (+ optionally calibrate) one config and score it on val + test."""
        t0 = time.perf_counter()

        if calibrate and family == "lightgbm":
            cal = ProbabilityCalibrator(self.config, self.logger)
            # C-5: neutralise the incumbent-clobbering side effects (never save(); redirect the plot).
            if out_dir is not None:
                cal.model_path = str(Path(out_dir) / "calibrated_model.joblib")
                cal.plot_path = str(Path(out_dir) / "calibration_curve.png")
                Path(out_dir).mkdir(parents=True, exist_ok=True)
            calres = cal.calibrate(data.X_fit, data.y_fit, data.X_val, data.y_val)
            model = cal.best_model_
            threshold = float(calres["threshold"]["f1_optimal"])
            used_params = params or {"base": "best_config.json", "calibration": calres["best_method"]}
            calibrated = True
        else:
            model = build_estimator(family, params, self.random_state).fit(data.X_fit, data.y_fit)
            threshold, calres, calibrated = 0.5, None, False
            used_params = params or {}

        val_metrics = _metrics_at_threshold(model, data.X_val, data.y_val, threshold)
        test_metrics = _metrics_at_threshold(model, data.X_test, data.y_test, threshold)
        dt = round(time.perf_counter() - t0, 1)
        self.logger.info("Trained %s (calibrated=%s) in %.1fs: test PR-AUC=%.4f ROC-AUC=%.4f @thr=%.3f",
                         family, calibrated, dt, test_metrics["pr_auc"], test_metrics["roc_auc"], threshold)
        return TrainResult(family=family, calibrated=calibrated, threshold=threshold, train_time_s=dt,
                           val_metrics=val_metrics, test_metrics=test_metrics, params=used_params,
                           calibration=(calres or {}).get("results") and {
                               "best_method": calres["best_method"],
                               "threshold": calres["threshold"]} or None,
                           model=model)
