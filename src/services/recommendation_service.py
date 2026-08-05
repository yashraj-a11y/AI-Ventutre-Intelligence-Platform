"""Day 14 · Phase E — Recommendation service (Day-11 RecommendationPipeline)."""
from __future__ import annotations
from typing import Any
from src.services.base import campaign_of


def recommend(registry: Any, req: Any) -> dict[str, Any]:
    """Unified venture decision (readiness + prioritized actions + confidence)."""
    d = registry.rec.run(campaign_of(req))
    return {k: d[k] for k in ("campaign", "investment_readiness", "confidence",
                              "priority_improvements", "investor_strategy", "final_strategy",
                              "deferred", "honesty_note")}
