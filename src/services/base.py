"""Day 14 · Phase E — shared service helpers.

Services take the :class:`~src.core.model_loader.ModelRegistry` + a request and return a plain dict.
They accept either a :class:`CampaignRequest` (from the API) or a raw campaign dict (from tests) —
so they're unit-testable without HTTP.
"""

from __future__ import annotations

from typing import Any


def campaign_of(req: Any) -> dict[str, Any]:
    """Normalize a request into the raw campaign dict the pipelines consume."""
    return req.to_campaign() if hasattr(req, "to_campaign") else dict(req)


def summary(camp: dict[str, Any]) -> dict[str, Any]:
    """A compact campaign identity block for response envelopes."""
    return {k: camp.get(k) for k in ("name", "main_category", "usd_goal_real")}
