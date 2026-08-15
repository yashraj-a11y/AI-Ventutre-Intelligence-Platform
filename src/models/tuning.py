"""Phase E — Hyperparameter Optimization (Day-4).

The Phase-D defaults were untuned. :class:`HyperparameterTuner` searches each
model's hyperparameters with **RandomizedSearchCV** over **StratifiedKFold**,
scoring **PR-AUC** (``average_precision`` — our primary metric), fit on the
**fit** set only. Random search (not exhaustive grid) samples a fixed budget of
configurations — far cheaper than the full grid, and provably near-optimal when
only a few hyperparameters matter.

Search spaces:
- **lightgbm**: n_estimators, learning_rate, num_leaves, max_depth,
  min_child_samples, subsample, colsample_bytree, reg_alpha/lambda, and
  ``scale_pos_weight`` (to lift the ~0.55 recall ceiling on the 40/60 split).
- **logreg**: C, l1_ratio, class_weight.

The best configuration + CV score is persisted; the tuned estimator is scored on
validation to confirm the CV gain generalises (test stays untouched until J).

Example
-------
>>> tuner = HyperparameterTuner()
>>> result = tuner.tune(X_fit, y_fit, X_val, y_val)
>>> result["best_params"]
{'num_leaves': 63, 'learning_rate': 0.05, ...}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.models.benchmarking import classifier_metrics
from src.utils.config import load_config
from src.utils.logger import get_logger


class HyperparameterTuner:
    """Randomized hyperparameter search (PR-AUC, stratified CV) for a chosen model."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.models.tuning")

        t = self.config.get("day4", {}).get("tuning", {})
        self.model_name: str = t.get("model", "lightgbm")
        self.n_iter: int = int(t.get("n_iter", 25))
        self.cv: int = int(t.get("cv", 3))
        self.scoring: str = t.get("scoring", "average_precision")
        self.random_state: int = int(t.get("random_state", 42))
        self.best_config_path = t.get("best_config_path", "artifacts/models/best_config.json")

        self.search_: Optional[RandomizedSearchCV] = None

    # --------------------------------------------------------------- spaces
    def _estimator_and_space(self):
        rs = self.random_state
        if self.model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            est = LGBMClassifier(random_state=rs, n_jobs=-1, verbose=-1)
            space = {
                "n_estimators": [200, 300, 500, 800],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "num_leaves": [15, 31, 63, 127],
                "max_depth": [-1, 5, 10, 20],
                "min_child_samples": [20, 50, 100, 200],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "reg_alpha": [0.0, 0.1, 1.0],
                "reg_lambda": [0.0, 0.1, 1.0],
                "scale_pos_weight": [1.0, 1.25, 1.5],  # 60/40 -> up-weight positives
            }
            return est, space
        if self.model_name == "logreg":
            from sklearn.linear_model import LogisticRegression
            est = LogisticRegression(solver="saga", max_iter=1000, random_state=rs)
            space = {
                "C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
                "l1_ratio": [0.0, 0.5, 1.0],
                "class_weight": [None, "balanced"],
            }
            return est, space
        raise ValueError(f"No search space for model '{self.model_name}'.")

    # --------------------------------------------------------------- public
    def tune(self, X_fit, y_fit, X_val, y_val) -> dict[str, Any]:
        est, space = self._estimator_and_space()
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        search = RandomizedSearchCV(
            est, param_distributions=space, n_iter=self.n_iter, scoring=self.scoring,
            cv=cv, n_jobs=-1, random_state=self.random_state, refit=True, verbose=0,
        )
        self.logger.info(
            "Tuning %s: %d candidates x %d folds = %d fits (scoring=%s)...",
            self.model_name, self.n_iter, self.cv, self.n_iter * self.cv, self.scoring,
        )
        t = time.perf_counter()
        search.fit(X_fit, y_fit)
        dt = time.perf_counter() - t
        self.search_ = search

        val_metrics = classifier_metrics(search.best_estimator_, X_val, y_val)
        result = {
            "model": self.model_name,
            "scoring": self.scoring,
            "best_cv_score": round(float(search.best_score_), 4),
            "best_params": search.best_params_,
            "val_metrics": val_metrics,
            "search_time_s": round(dt, 1),
            "n_fits": self.n_iter * self.cv,
        }
        self.logger.info(
            "Best CV %s=%.4f; val PR-AUC=%.4f; params=%s",
            self.scoring, search.best_score_, val_metrics["pr_auc"], search.best_params_,
        )
        self._save(result)
        return result

    def _save(self, result: dict[str, Any]) -> None:
        p = Path(self.best_config_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        self.logger.info("Saved best config -> %s", p)
