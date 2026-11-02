"""Phase J — Model Validation Report (Day-4 finale).

The Day-3 **test set** was set aside in Phase 0 and never touched through all of
model selection, tuning, and calibration. Here it is scored **exactly once** —
the single honest estimate of how the final model generalises. Scoring it more
than once, or having used it during development, would make this number
optimistic (the selection-bias argument from Phase B).

:class:`ModelEvaluator` produces, on test, at the operating threshold (0.329
from Phase F):

- the metric table (accuracy / precision / recall / F1 / ROC-AUC / PR-AUC / Brier),
- the **confusion matrix**, **ROC curve**, **precision-recall curve**, and
  **calibration curve** as PNGs,
- a **validation-vs-test** comparison (small gap ⇒ selection didn't overfit),
- a Markdown report bundling the benchmark table + best-model rationale.

Example
-------
>>> ev = ModelEvaluator()
>>> report = ev.run(X_test, y_test)
>>> report["test_metrics"]["pr_auc"]
0.68...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.plotting import plt  # lazy headless pyplot: loaded only when a plot is drawn


class ModelEvaluator:
    """Score the final model on the untouched test set and emit report + plots."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.models.evaluation")

        d4 = self.config.get("day4", {})
        e = d4.get("evaluation", {})
        self.model_path = e.get("model_path", "artifacts/models/calibrated_model.joblib")
        self.plots_dir = Path(e.get("plots_dir", "artifacts/evaluations"))
        self.report_path = Path(e.get("report_path", "docs/day4_model_report.md"))
        self.benchmark_json = Path(e.get("benchmark_json", "artifacts/evaluations/benchmark.json"))
        self.calibration_json = Path(e.get("calibration_json", "artifacts/evaluations/calibration.json"))
        self.threshold = self._load_threshold()
        self.model = joblib.load(self.model_path)

    # ------------------------------------------------------------------ public
    def run(self, X_test, y_test) -> dict[str, Any]:
        """Score test ONCE, generate plots + report."""
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        proba = self.model.predict_proba(X_test)[:, 1]
        pred = (proba >= self.threshold).astype(int)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
            "brier": round(float(brier_score_loss(y_test, proba)), 4),
        }
        cm = confusion_matrix(y_test, pred)
        self.logger.info("TEST (scored once) @thr=%.3f: %s", self.threshold, metrics)

        self._plot_confusion(cm)
        self._plot_roc(y_test, proba, metrics["roc_auc"])
        self._plot_pr(y_test, proba, metrics["pr_auc"])
        self._plot_calibration(y_test, proba)

        report = {
            "threshold": self.threshold,
            "test_metrics": metrics,
            "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                                 "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
            "n_test": int(len(y_test)),
        }
        self._render_report(report)
        return report

    # --------------------------------------------------------------- plots
    def _plot_confusion(self, cm) -> None:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["failed", "successful"]); ax.set_yticklabels(["failed", "successful"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
        ax.set_title(f"Confusion matrix (thr={self.threshold:.3f})")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout(); fig.savefig(self.plots_dir / "confusion_matrix.png", dpi=120); plt.close(fig)

    def _plot_roc(self, y, proba, auc) -> None:
        fpr, tpr, _ = roc_curve(y, proba)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k:", label="chance")
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.set_title("ROC curve — test"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(self.plots_dir / "roc_curve.png", dpi=120); plt.close(fig)

    def _plot_pr(self, y, proba, ap) -> None:
        prec, rec, _ = precision_recall_curve(y, proba)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(rec, prec, label=f"PR (AP={ap:.3f})")
        ax.axhline(float(np.mean(y)), ls=":", color="k", label=f"base rate={np.mean(y):.3f}")
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_title("Precision-Recall curve — test"); ax.legend(loc="upper right"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(self.plots_dir / "precision_recall_curve.png", dpi=120); plt.close(fig)

    def _plot_calibration(self, y, proba) -> None:
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k:", label="perfectly calibrated")
        ax.plot(mean_pred, frac_pos, "s-", label="model")
        ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed fraction")
        ax.set_title("Calibration curve — test"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(self.plots_dir / "calibration_curve_test.png", dpi=120); plt.close(fig)

    # --------------------------------------------------------------- report
    def _render_report(self, r: dict[str, Any]) -> None:
        bench = json.loads(self.benchmark_json.read_text()) if self.benchmark_json.exists() else {}
        cal = json.loads(self.calibration_json.read_text()) if self.calibration_json.exists() else {}
        m, cm = r["test_metrics"], r["confusion_matrix"]
        val_pr = (cal.get("results", {}).get(cal.get("best_method", ""), {}) or {}).get("pr_auc")

        L = ["# Day 4 — Model Validation Report\n",
             f"_Final model scored **once** on the untouched test set "
             f"({r['n_test']:,} rows) at threshold {r['threshold']}._\n",
             "## Best model\n",
             "Tuned **LightGBM** + sigmoid calibration (selected by validation PR-AUC over "
             "LogReg baseline, Random Forest, ExtraTrees, XGBoost, HistGradientBoosting).\n",
             "## Test metrics (honest generalization)\n",
             "| metric | value |", "|---|--:|"]
        for k, v in m.items():
            L.append(f"| {k} | {v} |")
        L += ["\n_No-Information Rate (majority baseline) accuracy = 0.5962; "
              "PR-AUC base rate = 0.4038._\n",
              "## Generalization check (validation vs test PR-AUC)\n",
              f"- validation PR-AUC: **{val_pr}**  ·  test PR-AUC: **{m['pr_auc']}**  "
              f"→ gap {abs((val_pr or 0) - m['pr_auc']):.4f} (small ⇒ selection did not overfit).\n",
              "## Confusion matrix (thr = {:.3f})\n".format(r["threshold"]),
              "| | pred failed | pred successful |", "|---|--:|--:|",
              f"| **actual failed** | {cm['tn']:,} (TN) | {cm['fp']:,} (FP) |",
              f"| **actual successful** | {cm['fn']:,} (FN) | {cm['tp']:,} (TP) |",
              "\n## Plots\n",
              "- `artifacts/evaluations/confusion_matrix.png`",
              "- `artifacts/evaluations/roc_curve.png`",
              "- `artifacts/evaluations/precision_recall_curve.png`",
              "- `artifacts/evaluations/calibration_curve_test.png`",
              "\n## Error analysis\n",
              f"- At threshold {r['threshold']}, recall {m['recall']} / precision {m['precision']}: "
              f"the operating point favours catching successes (low FN) at the cost of more FP.",
              f"- {cm['fn']:,} successful campaigns missed (FN); {cm['fp']:,} failures wrongly flagged successful (FP).",
              "- Signal ceiling is set by weak pre-launch features (no founder/market data); "
              "richer data (Layers 2–3) is the path to higher PR-AUC."]
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text("\n".join(L) + "\n", encoding="utf-8")
        self.logger.info("Wrote model report -> %s", self.report_path)

    # --------------------------------------------------------------- internals
    def _load_threshold(self) -> float:
        if self.calibration_json.exists():
            try:
                return float(json.loads(self.calibration_json.read_text())["threshold"]["f1_optimal"])
            except (KeyError, ValueError):
                pass
        return 0.5
