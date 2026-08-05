"""Day 14 · Phase E — Counterfactual service (Day-8 CounterfactualPipeline)."""
from __future__ import annotations
from typing import Any
from src.services.base import campaign_of, summary


def counterfactual(registry: Any, req: Any) -> dict[str, Any]:
    """What-if improvements over controllable features."""
    camp = campaign_of(req)
    a = registry.cf.analyze(camp)
    plan = {k: v for k, v in a["optimization_plan"].items() if k != "n_model_calls"}  # drop telemetry
    return {"campaign": summary(camp),
            "result": {"baseline_p": a["baseline_p"],
                       "ranked_recommendations": a["ranked_recommendations"],
                       "optimization_plan": plan},
            "honesty_note": (a.get("meta") or {}).get("note")}
