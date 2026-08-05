"""Day 14 · Phase E/G — Pipeline service (the ``/analyze`` orchestrator, Day-13 ReportingPipeline).

Builds the ReportBundle **once** and generates all four reports + the copilot from it (the Day-13
bundle-once discipline), and additionally surfaces the Day-12 committee verdict + confidence at the
top level for the ``AnalysisResponse``.
"""

from __future__ import annotations

from typing import Any

from src.services.base import campaign_of

_CAPTION = ("Full venture analysis reformatted from the Days 4-12 outputs on one warm stack (no new "
            "analysis). Deterministic, no LLM; deferred items are appended, not fabricated.")


def analyze(registry: Any, req: Any) -> dict[str, Any]:
    """The complete venture analysis: committee verdict + the full report suite + copilot."""
    rp = registry.reporting
    camp = campaign_of(req)
    bundle = rp.build_bundle(camp)                       # ONE bundle -> everything below is pure
    reports = {
        "founder": rp.founder.generate(camp, bundle=bundle),
        "investor": rp.investor.generate(camp, bundle=bundle),
        "memo": rp.memo.generate(camp, bundle=bundle),
        "executive": rp.executive.generate(camp, bundle=bundle),
    }
    copilot = rp.copilot.converse(bundle=bundle)
    committee = bundle["committee"]
    return {
        "campaign": bundle["campaign"],
        "committee_recommendation": committee["committee_recommendation"],
        "overall_confidence": committee["overall_confidence"],
        "reports": reports,
        "copilot": copilot,
        "honesty_note": _CAPTION,
    }
