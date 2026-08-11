"""Day 16 · Phase H — Automated model selection (fair comparison, explicit promotion).

Pick the best model *by evidence*, and promote it **only if it genuinely beats the incumbent** — never
automatically. Two honesty guarantees:

- **Fair F1.** Comparing F1 is only fair if each model is scored at *its own* F1-optimal threshold
  (tuned on validation) — otherwise the calibrated incumbent (threshold 0.329) would be compared
  against challengers stuck at 0.5. So every candidate gets its threshold tuned before comparison;
  PR-AUC / ROC-AUC are threshold-free and inherently fair.
- **The baseline is a floor, not a candidate.** The mandatory Logistic-Regression baseline is reported
  as the floor every model must clear; it is never itself a promotion target.

Ranking: primary **F1**, then PR-AUC, recall, then (secondary, ascending) inference latency and model
size. Promotion to Production is **explicit** (`apply`) and gated on a real margin over the incumbent —
if no challenger clears it, the honest recommendation is to **keep the incumbent**.
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score

from src.models.data_split import ModelData, ModelDataSplitter
from src.mlops.model_registry import ModelRegistry
from src.training.trainer import SingleModelTrainer, _metrics_at_threshold
from src.utils.config import load_config
from src.utils.logger import get_logger

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RANK_KEYS = ("f1", "pr_auc", "recall")           # primary → tie-breaks (all higher-is-better)


def _f1_optimal_threshold(model, X_val, y_val) -> float:
    """The decision threshold that maximises F1 on validation (fair per-model comparison)."""
    proba = model.predict_proba(X_val)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_val, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


class ModelSelector:
    """Evaluate candidates on a common basis, rank them, and recommend keep-or-promote vs the incumbent."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.training.selector")

        s = self.config.get("day16", {}).get("selection", {})
        self.min_f1_margin = float(s.get("min_f1_margin", 0.005))
        self.latency_sample = int(s.get("latency_sample", 2000))
        self.slot = self.config.get("day16", {}).get("training", {}).get("slot", "funding_success_classifier")
        self.out_path = _REPO_ROOT / s.get("out_path", "artifacts/selection/model_selection.json")
        self.trainer = SingleModelTrainer(self.config, self.logger)

    # ------------------------------------------------------------------ footprint
    def _footprint(self, model, X) -> dict[str, Any]:
        """Real inference latency (ms per 1000 rows) + serialized model size (MB)."""
        n = min(self.latency_sample, X.shape[0])
        Xs = X[:n]
        model.predict_proba(Xs[:1])                     # warm any lazy init
        t = time.perf_counter()
        model.predict_proba(Xs)
        latency_ms_per_1k = round((time.perf_counter() - t) / n * 1000 * 1000, 3)
        buf = io.BytesIO(); joblib.dump(model, buf)
        return {"latency_ms_per_1k": latency_ms_per_1k, "size_mb": round(len(buf.getvalue()) / (1 << 20), 2)}

    # ------------------------------------------------------------------ evaluate
    def evaluate_family(self, data: ModelData, family: str, *, calibrate: Optional[bool] = None,
                        role: str = "candidate") -> dict[str, Any]:
        """Train one family, tune its F1-optimal threshold, score on test, measure footprint."""
        calibrate = (family == "lightgbm") if calibrate is None else calibrate
        res = self.trainer.train(data, family=family, calibrate=calibrate)
        model = res.model
        # calibrated models already carry an F1-optimal threshold; tune it for the rest (fairness)
        threshold = res.threshold if res.calibrated else _f1_optimal_threshold(model, data.X_val, data.y_val)
        metrics = _metrics_at_threshold(model, data.X_test, data.y_test, threshold)
        fp = self._footprint(model, data.X_test)
        self.logger.info("%-14s F1=%.4f pr_auc=%.4f recall=%.4f @thr=%.3f | %.2fMB %.3fms/1k",
                         family, metrics["f1"], metrics["pr_auc"], metrics["recall"], threshold,
                         fp["size_mb"], fp["latency_ms_per_1k"])
        return {"name": family, "family": family, "role": role, "calibrated": res.calibrated,
                "threshold": round(threshold, 4), "metrics": metrics, "footprint": fp}

    # ------------------------------------------------------------------ rank + recommend
    def rank(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rank by F1, then PR-AUC, recall (desc), then latency, size (asc)."""
        return sorted(candidates, key=lambda c: (
            *[-c["metrics"][k] for k in _RANK_KEYS],
            c["footprint"]["latency_ms_per_1k"], c["footprint"]["size_mb"]))

    def recommend(self, candidates: list[dict[str, Any]], incumbent: dict[str, Any],
                  baseline: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Recommend keep-or-promote: promote only if the best challenger beats the incumbent's F1."""
        challengers = [c for c in candidates if c.get("role") != "baseline"]
        ranked = self.rank(challengers)
        best = ranked[0] if ranked else None
        inc_f1 = incumbent["metrics"]["f1"]
        beats = bool(best and best["metrics"]["f1"] > inc_f1 + self.min_f1_margin
                     and best["metrics"]["pr_auc"] >= incumbent["metrics"]["pr_auc"])
        clears_baseline = bool(best and baseline and best["metrics"]["f1"] >= baseline["metrics"]["f1"])
        rec = "promote" if beats else "keep_incumbent"
        rationale = (
            f"Best challenger {best['name']} F1={best['metrics']['f1']:.4f} vs incumbent "
            f"{inc_f1:.4f} (margin required {self.min_f1_margin}). "
            + ("Beats incumbent → promote." if beats else
               "Does NOT beat the incumbent by the required margin → keep incumbent."))
        return {"recommendation": rec, "beats_incumbent": beats, "clears_baseline": clears_baseline,
                "best_challenger": best, "incumbent": incumbent, "baseline": baseline,
                "ranked": ranked, "rationale": rationale, "primary": "f1",
                "min_f1_margin": self.min_f1_margin,
                "created_utc": datetime.now(timezone.utc).isoformat()}

    def save(self, decision: dict[str, Any]) -> Path:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(decision, indent=2, default=str))
        return self.out_path

    # ------------------------------------------------------------------ explicit promotion
    def apply(self, decision: dict[str, Any], *, version: Optional[str] = None,
              registry: Optional[ModelRegistry] = None, note: Optional[str] = None) -> dict[str, Any]:
        """Act on a decision: promote ONLY if recommended (explicit, gated). Otherwise a no-op."""
        if decision["recommendation"] != "promote":
            self.logger.info("Selection: keeping incumbent (no challenger beat it). No promotion applied.")
            return {"promoted": False, "reason": "recommendation is keep_incumbent"}
        if version is None:
            raise ValueError("promotion requires the registry `version` of the winning challenger")
        reg = registry or ModelRegistry(self.config, self.logger)
        entry = reg.promote(self.slot, version, "Production",
                            note=note or f"auto-selected: {decision['rationale']}")
        self.logger.info("Selection: promoted %s v%s to Production.", self.slot, version)
        return {"promoted": True, "version": version, "status": entry["status"]}

    # ------------------------------------------------------------------ convenience
    def run(self, families: list[str], incumbent_family: str = "lightgbm",
            data: Optional[ModelData] = None) -> dict[str, Any]:
        """Evaluate the incumbent family + challengers + baseline, then recommend."""
        data = data or ModelDataSplitter(self.config, self.logger).load()
        incumbent = self.evaluate_family(data, incumbent_family, calibrate=True, role="incumbent")
        baseline = self.evaluate_family(data, "logreg", calibrate=False, role="baseline")
        challengers = [self.evaluate_family(data, f, role="candidate")
                       for f in families if f not in (incumbent_family, "logreg")]
        decision = self.recommend(challengers, incumbent, baseline)
        self.save(decision)
        self.logger.info("Selection → %s. %s", decision["recommendation"], decision["rationale"])
        return decision
