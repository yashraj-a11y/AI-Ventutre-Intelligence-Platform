"""Day 14 · Phase E — Report service (Day-13 founder / investor report generators)."""

from __future__ import annotations

from typing import Any

from src.services.base import campaign_of


def founder(registry: Any, req: Any) -> dict[str, Any]:
    """Founder Intelligence Report (Day-13)."""
    return registry.reporting.founder.generate(campaign_of(req))


def investor(registry: Any, req: Any) -> dict[str, Any]:
    """Investor Due Diligence Report (Day-13)."""
    return registry.reporting.investor.generate(campaign_of(req))
