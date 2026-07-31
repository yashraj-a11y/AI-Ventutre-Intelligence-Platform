"""Phase E — Improvement Ranking Engine (Day-8, Layer 7).

Not every recommendation is equal. :class:`RankingEngine` scores each simulated,
evidence-backed recommendation and ranks them, so the report leads with the highest-value,
best-supported change.

Score = **predicted probability gain** (the real, model-measured Δp — primary) ×
**tier weight** (strong / moderate / suggestive — causal-plausibility prior from Phase B) ×
**evidence factor** (historical lift, clipped) × **feasibility** (per-feature heuristic).
Negligible or negative-gain changes are dropped. Each surviving recommendation gets a
**confidence label** (high / medium / low) combining gain + evidence + tier.

Honesty: ``cost``/``time`` are **heuristic labels** (no real cost data) — for pre-launch
parameters they are genuinely low/immediate (you just set a different value before launch),
but they are flagged as heuristic, not measured.

Example
-------
>>> ranked = RankingEngine().rank(validated_recommendations)
>>> ranked[0]["feature"], ranked[0]["priority"], ranked[0]["confidence"]
('goal', 1, 'high')
"""

from __future__ import annotations

from typing import Any, Optional

from src.utils.config import load_config
from src.utils.logger import get_logger


class RankingEngine:
    """Score and rank evidence-backed recommendations.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.cf.ranking")

        c = self.config.get("day8", {}).get("ranking", {})
        self.min_delta = float(c.get("min_delta", 0.005))
        self.tier_weights = dict(c.get("tier_weights",
                                       {"strong": 1.0, "moderate": 0.85, "suggestive": 0.7}))
        self.feasibility = dict(c.get("feasibility", {}))
        self.lift_cap = float(c.get("evidence_lift_cap", 1.8))

    # ------------------------------------------------------------------ public
    def rank(self, recommendations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score, filter, and priority-rank recommendations (highest value first)."""
        scored = []
        for r in recommendations:
            delta = float(r.get("delta", 0.0))
            if delta < self.min_delta:                       # drop negligible / negative
                continue
            ev = r.get("evidence", {}) or {}
            tier = r.get("tier", "suggestive")

            tier_w = self.tier_weights.get(tier, 0.7)
            lift = ev.get("lift", 1.0) if ev.get("available") else 1.0
            ev_factor = max(0.5, min(self.lift_cap, float(lift)))   # reward >1, cap outliers
            feas = float(self.feasibility.get(r.get("feature"), 1.0))
            score = delta * tier_w * ev_factor * feas

            scored.append({
                **r,
                "score": round(score, 5),
                "feasibility": round(feas, 2),
                "confidence": self._confidence(delta, ev, tier),
                "cost_label": "low (pre-launch setting)",     # heuristic
                "time_label": "immediate (set before launch)",  # heuristic
                "labels_are_heuristic": True,
            })

        scored.sort(key=lambda r: r["score"], reverse=True)
        for i, r in enumerate(scored, start=1):
            r["priority"] = i
        self.logger.info("Ranked %d recommendations (dropped %d below Δ %.3f).",
                         len(scored), len(recommendations) - len(scored), self.min_delta)
        return scored

    # --------------------------------------------------------------- internals
    def _confidence(self, delta: float, ev: dict[str, Any], tier: str) -> str:
        """High/medium/low, combining model gain + historical evidence + tier."""
        lift = ev.get("lift", 0.0) if ev.get("available") else 0.0
        n = ev.get("n", 0)
        if delta >= 0.02 and lift >= 1.15 and n >= 500 and tier in ("strong", "moderate"):
            return "high"
        if delta >= 0.01 and (lift >= 1.05 or tier != "suggestive"):
            return "medium"
        return "low"
