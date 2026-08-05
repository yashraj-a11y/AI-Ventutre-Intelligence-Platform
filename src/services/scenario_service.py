"""Day 14 · Phase E — Scenario service (Day-10 ScenarioPipeline)."""
from __future__ import annotations
from typing import Any
from src.services.base import campaign_of, summary


def scenario(registry: Any, req: Any) -> dict[str, Any]:
    """Multi-variable what-if scenarios, scored + compared."""
    camp = campaign_of(req)
    s = registry.scenario.run(camp)
    return {"campaign": summary(camp), "result": s, "honesty_note": s.get("honesty_note")}
