"""Phase C — Logistic Regression Baseline (Day-4, mandatory Week 6).

Logistic regression is the reference baseline for high-dimensional **sparse**
data: fast, interpretable (one coefficient per feature), and naturally
probabilistic. Before any tree model earns its complexity, it must beat a
well-regularised LogReg — and LogReg itself must beat the majority-class floor.

:class:`LogisticRegressionTrainer` fits, on the **fit** set and scores on
**validation**:

- a ``DummyClassifier`` (most-frequent) — the No-Information Rate floor (~0.596);
- **L2** LogReg (mandatory) — ridge penalty, shrinks all coefficients;
- **L1** LogReg — lasso penalty, drives weak coefficients to *exactly* zero
  (feature selection for free);
- **ElasticNet** — a mix of L1 and L2.

All three use ``solver="saga"`` — the only sklearn solver that handles L1 /
ElasticNet on large sparse matrices. Metrics reported: accuracy, precision,
recall, F1, ROC-AUC, and **PR-AUC** (primary, for the 40/60 imbalance).

Example
-------
>>> trainer = LogisticRegressionTrainer()
>>> results = trainer.train_all(X_fit, y_fit, X_val, y_val)
>>> results[0].metrics["pr_auc"]
0.61...
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config import load_config
from src.utils.logger import get_logger


@dataclass
class ModelResult:
    """Metrics + provenance for one fitted model."""

    name: str
    train_time_s: float
    metrics: dict[str, float]
    n_nonzero_coef: Optional[int] = None
    params: dict[str, Any] = field(default_factory=dict)


class LogisticRegressionTrainer:
    """Fit Dummy + L2/L1/ElasticNet LogReg and score them on validation.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.models.logistic_regression")

        c = self.config.get("day4", {}).get("logreg", {})
        self.penalties: list[str] = list(c.get("penalties", ["l2", "l1", "elasticnet"]))
        self.C: float = float(c.get("C", 1.0))
        self.l1_ratio: float = float(c.get("l1_ratio", 0.5))
        self.solver: str = c.get("solver", "saga")
        self.max_iter: int = int(c.get("max_iter", 1000))
        self.class_weight = c.get("class_weight", None)
        self.random_state: int = int(c.get("random_state", 42))
        self._feature_names_path = c.get(
            "feature_names_path", "artifacts/feature_pipeline/feature_metadata.json"
        )

        self.models_: dict[str, Any] = {}
        self.results_: list[ModelResult] = []

    # ------------------------------------------------------------------ public
    def train_all(
        self, X_fit, y_fit, X_val, y_val
    ) -> list[ModelResult]:
        """Fit the floor + each LogReg variant; score all on validation."""
        self.results_ = []

        # Floor: majority-class baseline (the No-Information Rate).
        self.results_.append(self._fit_dummy(X_fit, y_fit, X_val, y_val))

        for penalty in self.penalties:
            self.results_.append(self._fit_logreg(penalty, X_fit, y_fit, X_val, y_val))

        # Rank by PR-AUC (primary metric).
        self.results_.sort(key=lambda r: r.metrics["pr_auc"], reverse=True)
        self.logger.info("Best by PR-AUC: %s (%.4f).", self.results_[0].name,
                         self.results_[0].metrics["pr_auc"])
        return self.results_

    def top_coefficients(self, model_name: str = "logreg_l1", k: int = 15):
        """Largest-magnitude coefficients of a fitted model (Week-7 preview)."""
        model = self.models_[model_name]
        names = self._feature_names()
        coefs = model.coef_.ravel()
        order = np.argsort(np.abs(coefs))[::-1][:k]
        return [(names[i] if i < len(names) else f"f{i}", float(coefs[i])) for i in order]

    def save(self, model_name: str = "logreg_l2",
             path: str | Path = "artifacts/models/logistic_regression.joblib") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models_[model_name], path)
        self.logger.info("Saved %s -> %s", model_name, path)
        return path

    # --------------------------------------------------------------- internals
    def _fit_dummy(self, X_fit, y_fit, X_val, y_val) -> ModelResult:
        t = time.perf_counter()
        dummy = DummyClassifier(strategy="most_frequent").fit(X_fit, y_fit)
        dt = time.perf_counter() - t
        self.models_["dummy"] = dummy
        res = ModelResult("dummy_majority", dt, self._evaluate(dummy, X_val, y_val),
                          params={"strategy": "most_frequent"})
        self._log(res)
        return res

    def _fit_logreg(self, penalty, X_fit, y_fit, X_val, y_val) -> ModelResult:
        # sklearn >=1.8 deprecates `penalty=` in favour of `l1_ratio=`:
        #   L2 -> 0.0, L1 -> 1.0, ElasticNet -> mix in (0, 1).
        l1_ratio = {"l2": 0.0, "l1": 1.0, "elasticnet": self.l1_ratio}[penalty]
        kwargs = dict(
            l1_ratio=l1_ratio, C=self.C, solver=self.solver, max_iter=self.max_iter,
            class_weight=self.class_weight, random_state=self.random_state,
        )

        t = time.perf_counter()
        model = LogisticRegression(**kwargs).fit(X_fit, y_fit)
        dt = time.perf_counter() - t

        name = f"logreg_{penalty}"
        self.models_[name] = model
        nnz = int(np.count_nonzero(model.coef_))
        res = ModelResult(
            name=name, train_time_s=dt, metrics=self._evaluate(model, X_val, y_val),
            n_nonzero_coef=nnz,
            params={"penalty": penalty, "C": self.C,
                    **({"l1_ratio": self.l1_ratio} if penalty == "elasticnet" else {})},
        )
        self._log(res)
        return res

    def _evaluate(self, model, X, y) -> dict[str, float]:
        pred = model.predict(X)
        metrics = {
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
        }
        # Probabilistic metrics need predict_proba (dummy has it too).
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            # ROC/PR-AUC undefined if the score is constant (dummy) -> guard.
            if len(np.unique(proba)) > 1:
                metrics["roc_auc"] = roc_auc_score(y, proba)
                metrics["pr_auc"] = average_precision_score(y, proba)
            else:
                metrics["roc_auc"] = 0.5
                metrics["pr_auc"] = float(np.mean(y))  # PR-AUC floor = base rate
        return {k: round(float(v), 4) for k, v in metrics.items()}

    def _feature_names(self) -> list[str]:
        p = Path(self._feature_names_path)
        if p.exists():
            return json.loads(p.read_text()).get("feature_names", [])
        return []

    def _log(self, r: ModelResult) -> None:
        m = r.metrics
        self.logger.info(
            "%-16s  acc=%.4f  f1=%.4f  roc_auc=%.4f  pr_auc=%.4f  (%.1fs%s)",
            r.name, m["accuracy"], m["f1"], m.get("roc_auc", 0.5), m["pr_auc"],
            r.train_time_s,
            f", nnz={r.n_nonzero_coef}" if r.n_nonzero_coef is not None else "",
        )
