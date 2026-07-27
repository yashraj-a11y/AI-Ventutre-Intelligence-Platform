"""Phase D — Classical ML Model Benchmarking (Day-4).

The LogReg baseline (Phase C) set the bar: PR-AUC 0.626 on validation. A more
complex model only earns its place if it beats that. :class:`ModelBenchmark`
trains several tree-ensemble families on the **same** fit set and scores them on
the **same** validation set, comparing not just accuracy but **training time**
and the imbalance-aware metrics (PR-AUC / F1).

Models
------
- ``random_forest`` / ``extra_trees`` — bagging ensembles (parallel, low-variance).
- ``xgboost`` / ``lightgbm`` — gradient boosting (sequential, high-accuracy,
  histogram-based, sparse-native).
- ``hist_gradient_boosting`` — sklearn's fast gradient booster (needs **dense**
  input, so its block is densified just-in-time).

Everything is fit on train only and evaluated on validation — the Day-3 test set
stays untouched until Phase J.

Example
-------
>>> bench = ModelBenchmark()
>>> results = bench.run(X_fit, y_fit, X_val, y_val)
>>> results[0].name          # best by PR-AUC
'lightgbm'
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
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


def classifier_metrics(model, X, y) -> dict[str, float]:
    """Standard binary-classification metrics (label + probability based)."""
    pred = model.predict(X)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }
    proba = model.predict_proba(X)[:, 1]
    out["roc_auc"] = roc_auc_score(y, proba)
    out["pr_auc"] = average_precision_score(y, proba)
    return {k: round(float(v), 4) for k, v in out.items()}


@dataclass
class BenchResult:
    name: str
    train_time_s: float
    metrics: dict[str, float]
    needs_dense: bool = False
    params: dict[str, Any] = field(default_factory=dict)


class ModelBenchmark:
    """Train several model families and rank them by PR-AUC on validation."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.models.benchmarking")

        b = self.config.get("day4", {}).get("benchmark", {})
        self.model_names: list[str] = list(b.get("models", []))
        self.n_estimators: int = int(b.get("n_estimators", 200))
        self.n_jobs: int = int(b.get("n_jobs", -1))
        self.random_state: int = int(b.get("random_state", 42))

        self.models_: dict[str, Any] = {}
        self.results_: list[BenchResult] = []

    # ------------------------------------------------------------------ registry
    def _registry(self) -> dict[str, tuple[Any, bool]]:
        """name -> (estimator, needs_dense)."""
        rs, nj, ne = self.random_state, self.n_jobs, self.n_estimators
        reg: dict[str, tuple[Any, bool]] = {
            "random_forest": (
                RandomForestClassifier(n_estimators=ne, n_jobs=nj, random_state=rs), False,
            ),
            "extra_trees": (
                ExtraTreesClassifier(n_estimators=ne, n_jobs=nj, random_state=rs), False,
            ),
            "hist_gradient_boosting": (
                HistGradientBoostingClassifier(max_iter=ne, random_state=rs), True,
            ),
        }
        # Optional deps — only register if importable.
        try:
            from xgboost import XGBClassifier
            reg["xgboost"] = (
                XGBClassifier(n_estimators=ne, tree_method="hist", n_jobs=nj,
                              random_state=rs, eval_metric="logloss"), False,
            )
        except ImportError:
            self.logger.warning("xgboost not available — skipping.")
        try:
            from lightgbm import LGBMClassifier
            reg["lightgbm"] = (
                LGBMClassifier(n_estimators=ne, n_jobs=nj, random_state=rs, verbose=-1), False,
            )
        except ImportError:
            self.logger.warning("lightgbm not available — skipping.")
        return reg

    # ------------------------------------------------------------------ public
    def run(self, X_fit, y_fit, X_val, y_val) -> list[BenchResult]:
        registry = self._registry()
        X_fit_dense = X_val_dense = None  # lazily materialised for dense-only models

        for name in self.model_names:
            if name not in registry:
                self.logger.warning("Model '%s' not in registry — skipped.", name)
                continue
            estimator, needs_dense = registry[name]

            if needs_dense:
                if X_fit_dense is None:
                    self.logger.info("Densifying for %s (dense-only model)...", name)
                    X_fit_dense = X_fit.toarray() if sp.issparse(X_fit) else X_fit
                    X_val_dense = X_val.toarray() if sp.issparse(X_val) else X_val
                Xf, Xv = X_fit_dense, X_val_dense
            else:
                Xf, Xv = X_fit, X_val

            t = time.perf_counter()
            estimator.fit(Xf, y_fit)
            dt = time.perf_counter() - t

            self.models_[name] = estimator
            res = BenchResult(
                name=name, train_time_s=round(dt, 1),
                metrics=classifier_metrics(estimator, Xv, y_val),
                needs_dense=needs_dense,
                params={"n_estimators": self.n_estimators},
            )
            self.results_.append(res)
            m = res.metrics
            self.logger.info(
                "%-22s acc=%.4f f1=%.4f roc_auc=%.4f pr_auc=%.4f (%.1fs)",
                name, m["accuracy"], m["f1"], m["roc_auc"], m["pr_auc"], dt,
            )

        self.results_.sort(key=lambda r: r.metrics["pr_auc"], reverse=True)
        if self.results_:
            self.logger.info("Best by PR-AUC: %s (%.4f).",
                             self.results_[0].name, self.results_[0].metrics["pr_auc"])
        return self.results_

    # --------------------------------------------------------------- reporting
    def save_json(self, extra: Optional[dict[str, Any]] = None,
                  path: str | Path = "artifacts/evaluations/benchmark.json") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {r.name: asdict(r) for r in self.results_}
        if extra:
            payload["_context"] = extra
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def render_report(self, baseline: Optional[dict[str, Any]] = None,
                      path: str | Path = "docs/day4_benchmark_report.md") -> Path:
        rows = list(self.results_)
        # Merge in the LogReg baseline row (from Phase C) for comparison.
        merged = []
        if baseline:
            merged.append(("logreg_l2 (baseline)", baseline["train_time_s"], baseline["metrics"]))
        for r in rows:
            merged.append((r.name, r.train_time_s, r.metrics))
        merged.sort(key=lambda t: t[2]["pr_auc"], reverse=True)

        lines = ["# Day 4 — Phase D: Model Benchmark\n",
                 "_Validation set (53,035 rows). Ranked by PR-AUC. Baseline: LogReg L2._\n",
                 "| model | acc | precision | recall | F1 | ROC-AUC | PR-AUC | train time |",
                 "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for name, t, m in merged:
            lines.append(f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                         f"{m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | "
                         f"**{m['pr_auc']:.4f}** | {t:.1f}s |")
        best = merged[0]
        lines.append(f"\n**Best by PR-AUC: `{best[0]}` (PR-AUC {best[2]['pr_auc']:.4f}).**")
        path = Path(path)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
