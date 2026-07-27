"""Phase F — Probability Calibration (Day-4).

A ranking-good model isn't automatically a *probability*-good model. Boosting in
particular tends to push scores toward 0/1 (over-confident), so "0.7" may not
mean "70% of these succeed". The platform promises *"81% funding probability"*,
so the numbers must be **calibrated**: among campaigns scored p, roughly a
fraction p should actually succeed.

:class:`ProbabilityCalibrator` takes the tuned LightGBM (Phase E params) and:

- fits the uncalibrated base + two calibrated variants — **sigmoid** (Platt) and
  **isotonic** — using cross-fitted :class:`CalibratedClassifierCV` on the
  **fit** set (calibration learned from fit, never test/val leakage into fitting);
- measures calibration quality on **validation** with the **Brier score** and a
  **reliability curve**;
- runs a **threshold analysis** — calibration/AUC don't fix recall (they're
  monotonic, so ranking is unchanged); the recall lever is the decision
  threshold, chosen here to maximise F1.

Calibration is a *monotonic* re-map of scores → ROC-AUC / PR-AUC are unchanged;
only the probability values (and thus Brier) move.

Example
-------
>>> cal = ProbabilityCalibrator()
>>> result = cal.calibrate(X_fit, y_fit, X_val, y_val)
>>> result["best_method"], result["threshold"]["f1_optimal"]
('isotonic', 0.42)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib
matplotlib.use("Agg")  # headless: save PNGs, no display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)

from src.utils.config import load_config
from src.utils.logger import get_logger


class ProbabilityCalibrator:
    """Calibrate the tuned model's probabilities and pick the decision threshold."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.models.calibration")

        d4 = self.config.get("day4", {})
        c = d4.get("calibration", {})
        self.methods: list[str] = list(c.get("methods", ["sigmoid", "isotonic"]))
        self.cv: int = int(c.get("cv", 5))
        self.n_bins: int = int(c.get("n_bins", 10))
        self.plot_path = c.get("plot_path", "artifacts/evaluations/calibration_curve.png")
        self.model_path = c.get("model_path", "artifacts/models/calibrated_model.joblib")
        self._best_config_path = d4.get("tuning", {}).get(
            "best_config_path", "artifacts/models/best_config.json"
        )
        self.random_state = int(d4.get("tuning", {}).get("random_state", 42))

        self.best_model_ = None

    # --------------------------------------------------------------- base model
    def _base_estimator(self):
        from lightgbm import LGBMClassifier
        params = {}
        p = Path(self._best_config_path)
        if p.exists():
            params = json.loads(p.read_text()).get("best_params", {})
        self.logger.info("Base LightGBM params: %s", params or "(defaults)")
        return LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbose=-1, **params)

    # --------------------------------------------------------------- public
    def calibrate(self, X_fit, y_fit, X_val, y_val) -> dict[str, Any]:
        results: dict[str, Any] = {}

        # Uncalibrated base (fit on fit, scored on val).
        base = self._base_estimator().fit(X_fit, y_fit)
        results["uncalibrated"] = self._score(base, X_val, y_val)

        # Calibrated variants (cross-fitted on fit set).
        self._models = {"uncalibrated": base}
        for method in self.methods:
            cal = CalibratedClassifierCV(self._base_estimator(), method=method, cv=self.cv)
            cal.fit(X_fit, y_fit)
            self._models[method] = cal
            results[method] = self._score(cal, X_val, y_val)

        # Pick the lowest-Brier variant as the calibrated model to ship.
        best_method = min(results, key=lambda k: results[k]["brier"])
        self.best_model_ = self._models[best_method]

        # Threshold analysis on the best model's probabilities.
        proba = self.best_model_.predict_proba(X_val)[:, 1]
        threshold = self._threshold_analysis(y_val, proba)

        self._plot(results)
        out = {
            "results": results,
            "best_method": best_method,
            "threshold": threshold,
            "plot": str(self.plot_path),
        }
        self.logger.info(
            "Best calibration: %s (Brier %.4f). F1-optimal threshold %.2f "
            "(recall %.4f vs %.4f at 0.50).",
            best_method, results[best_method]["brier"],
            threshold["f1_optimal"], threshold["recall_at_optimal"], threshold["recall_at_0.5"],
        )
        return out

    def save(self) -> Path:
        path = Path(self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model_, path)
        self.logger.info("Saved calibrated model -> %s", path)
        return path

    # --------------------------------------------------------------- internals
    def _score(self, model, X, y) -> dict[str, float]:
        proba = model.predict_proba(X)[:, 1]
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=self.n_bins, strategy="quantile")
        return {
            "brier": round(float(brier_score_loss(y, proba)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
            "pr_auc": round(float(average_precision_score(y, proba)), 4),
            "_mean_pred": mean_pred.tolist(),
            "_frac_pos": frac_pos.tolist(),
        }

    def _threshold_analysis(self, y, proba) -> dict[str, float]:
        prec, rec, thr = precision_recall_curve(y, proba)
        # precision_recall_curve returns one extra prec/rec point without a threshold.
        f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        best = int(np.argmax(f1s))
        t = float(thr[best])
        pred_opt = (proba >= t).astype(int)
        pred_half = (proba >= 0.5).astype(int)
        return {
            "f1_optimal": round(t, 4),
            "f1_at_optimal": round(float(f1_score(y, pred_opt)), 4),
            "recall_at_optimal": round(float(recall_score(y, pred_opt)), 4),
            "f1_at_0.5": round(float(f1_score(y, pred_half)), 4),
            "recall_at_0.5": round(float(recall_score(y, pred_half)), 4),
        }

    def _plot(self, results) -> None:
        Path(self.plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k:", label="perfectly calibrated")
        for label, r in results.items():
            ax.plot(r["_mean_pred"], r["_frac_pos"], "s-", label=label, alpha=0.8)
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed fraction of successes")
        ax.set_title("Reliability curve — funding-success probabilities")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=120)
        plt.close(fig)
