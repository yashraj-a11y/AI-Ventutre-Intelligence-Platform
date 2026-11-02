"""Phase C — Funding Probability Simulator (Day-8, Layer 7).

Measure how a candidate change moves the funding probability by **applying its raw-input
edits and re-running the real Day-4 calibrated model** (which re-runs the Day-3 pipeline, so
every coupled feature recomputes). Two views:

- **Single-feature Δ** (:meth:`simulate_all`) — each candidate's effect *in isolation* vs the
  baseline. Feeds the ranking (Phase E).
- **Honest cumulative plan** (:meth:`greedy_plan`) — greedily stack the best changes, but
  **re-simulate the jointly-modified campaign at every step** rather than summing Δs (features
  interact; probabilities saturate). At most one change per feature (a goal change and a
  *different* goal change can't both be in a plan).

The simulator is a pure query over the existing model — no new training. It uses only
``PredictionEngine`` (lightgbm), so there is **no faiss/torch** and no OpenMP concern.

Example
-------
>>> sim = ProbabilitySimulator()
>>> plan = sim.greedy_plan(raw, candidates)
>>> plan["baseline_p"], plan["final_p"], [s["feature"] for s in plan["steps"]]
(0.31, 0.55, ['goal', 'campaign_duration'])
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any, Optional

import pandas as pd

from src.counterfactual.candidate_generator import Candidate
from src.models.prediction_engine import PredictionEngine
from src.utils.config import load_config
from src.utils.logger import get_logger


class ProbabilitySimulator:
    """Perturb a campaign and re-run the Day-4 model to measure Δ funding probability.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    engine:
        Optional pre-built :class:`PredictionEngine` (share one warm calibrated model).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 engine: Optional[PredictionEngine] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.cf.simulator")

        c = self.config.get("day8", {}).get("simulator", {})
        self.min_gain = float(c.get("min_gain", 0.005))
        self.cols = list(c.get("predict_cols", [
            "name", "category", "main_category", "currency", "deadline", "goal",
            "launched", "country", "usd_goal_real", "name_clean"]))

        self.engine = engine or PredictionEngine(self.config, self.logger)
        self._n_calls = 0
        # Memo of campaign -> probability. The model is deterministic, and one request re-scores the
        # same campaigns many times (baseline + every greedy step), so hits are exact reuse.
        self._cache: OrderedDict[tuple, float] = OrderedDict()
        self._cache_size = int(c.get("prob_cache_size", 4096))
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def baseline(self, raw: dict[str, Any]) -> float:
        """Calibrated funding probability of a raw campaign."""
        return self._prob(raw)

    def simulate_all(self, raw: dict[str, Any],
                     candidates: list[Candidate]) -> list[dict[str, Any]]:
        """Single-feature Δp for each candidate (independent, vs baseline)."""
        base = self.baseline(raw)
        out = []
        probs = self._probs([{**raw, **c.raw_changes} for c in candidates])   # one batched model call
        for c, new_p in zip(candidates, probs):
            out.append({
                "feature": c.feature, "current": c.current, "proposed": c.proposed,
                "tier": c.tier, "rationale_key": c.rationale_key,
                "baseline_p": round(base, 4), "new_p": round(new_p, 4),
                "delta": round(new_p - base, 4),
                "raw_changes": c.raw_changes,
            })
        out.sort(key=lambda d: d["delta"], reverse=True)
        return out

    def greedy_plan(self, raw: dict[str, Any], candidates: list[Candidate],
                    min_gain: Optional[float] = None) -> dict[str, Any]:
        """Build a cumulative plan; each step re-simulates the joint campaign (never sums Δ)."""
        min_gain = self.min_gain if min_gain is None else min_gain
        base_p = self.baseline(raw)
        current_raw = dict(raw)
        current_p = base_p
        used: set[str] = set()
        steps: list[dict[str, Any]] = []

        while True:
            best_c, best_p = None, current_p
            open_cands = [c for c in candidates if c.feature not in used]   # one change per feature
            # RE-SIMULATE each joint campaign — scored together in one batch, picked in the same order
            probs = self._probs([{**current_raw, **c.raw_changes} for c in open_cands])
            for c, p in zip(open_cands, probs):
                if p > best_p + 1e-9:
                    best_p, best_c = p, c
            if best_c is None or (best_p - current_p) < min_gain:
                break
            marginal = best_p - current_p
            current_raw = {**current_raw, **best_c.raw_changes}
            used.add(best_c.feature)
            steps.append({
                "feature": best_c.feature, "current": best_c.current,
                "proposed": best_c.proposed, "tier": best_c.tier,
                "rationale_key": best_c.rationale_key,
                "cumulative_p": round(best_p, 4),
                "marginal_gain": round(marginal, 4),
                "raw_changes": best_c.raw_changes,
            })
            current_p = best_p

        return {
            "baseline_p": round(base_p, 4),
            "final_p": round(current_p, 4),
            "total_gain": round(current_p - base_p, 4),
            "n_steps": len(steps),
            "steps": steps,
            "n_model_calls": self._n_calls,
            "note": ("Cumulative probabilities are re-simulated jointly through the calibrated "
                     "model, not summed. Model-simulated estimate, not a causal guarantee."),
        }

    # --------------------------------------------------------------- internals
    def _prob(self, campaign: dict[str, Any]) -> float:
        return self._probs([campaign])[0]

    def _probs(self, campaigns: list[dict[str, Any]]) -> list[float]:
        """Probabilities for several campaigns: cache hits reused, misses scored in ONE model call.

        Batched scoring is bitwise-identical to row-by-row (the pipeline and model are row-wise).
        ``_n_calls`` still counts every requested evaluation, so telemetry is unchanged.
        """
        self._n_calls += len(campaigns)
        rows = [{k: c.get(k) for k in self.cols} for c in campaigns]
        keys = [self._key(r) for r in rows]
        out: list[Optional[float]] = [None] * len(rows)
        with self._lock:
            for i, k in enumerate(keys):
                if k is not None and k in self._cache:
                    self._cache.move_to_end(k)
                    out[i] = self._cache[k]
        miss = [i for i, p in enumerate(out) if p is None]
        if miss:
            # one row per distinct campaign; duplicates within the batch share the result
            uniq: dict[Any, int] = {}
            for i in miss:
                uniq.setdefault(keys[i] if keys[i] is not None else ("_nokey", i), i)
            # pandas infers ONE date format per column from its first value, so rows are batched only
            # with rows whose date strings have the same shape ("2011-08-17" vs "2011-08-02 19:08:09"
            # would otherwise mis-parse); within a shape group, scores equal row-by-row exactly.
            groups: dict[tuple, list[tuple[Any, int]]] = {}
            for k, i in uniq.items():
                groups.setdefault(self._date_shape(rows[i]), []).append((k, i))
            by_key: dict[Any, float] = {}
            for members in groups.values():
                probs = self.engine.predict_proba(pd.DataFrame([rows[i] for _, i in members]))
                by_key.update({k: float(p) for (k, _), p in zip(members, probs)})
            with self._lock:
                for i in miss:
                    k = keys[i] if keys[i] is not None else ("_nokey", i)
                    out[i] = by_key[k]
                    if keys[i] is not None:
                        self._cache[keys[i]] = out[i]
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return [float(p) for p in out]

    @staticmethod
    def _date_shape(row: dict[str, Any]) -> tuple:
        """Type + digit-masked text of each date field: rows that share it parse identically."""
        return tuple((type(row.get(k)).__name__, re.sub(r"\d", "0", str(row.get(k))))
                     for k in ("launched", "deadline"))

    @staticmethod
    def _key(row: dict[str, Any]) -> Optional[tuple]:
        """Hashable cache key for a model input row (None if a value isn't hashable)."""
        try:
            key = tuple((k, type(v).__name__, v) for k, v in row.items())
            hash(key)
            return key
        except TypeError:
            return None
