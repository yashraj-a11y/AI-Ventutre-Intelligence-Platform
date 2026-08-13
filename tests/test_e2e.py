"""Day 17 · Phase H — end-to-end integration (one campaign through the whole platform).

All `heavy`. Exercises the Day-14 `/analyze` orchestrator, which builds one ReportBundle spanning every
layer (prediction → SHAP drivers → retrieval cohort → counterfactual → scenario → evolution → copilot →
Day-11 decision → Day-12 committee → four reports). We assert every stage produced output **and** that
the honesty `deferred_*` fields survive from the deep pipeline all the way to the report output — the
platform never fabricates what the data can't support, even end to end.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.heavy

CAMPAIGN = {
    "name": "Artificer - Board Game", "main_category": "Games", "usd_goal_real": 30000.0,
    "launched": "2016-05-01 12:00:00", "deadline": "2016-05-31 12:00:00",
}


@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    with TestClient(app) as c:
        yield c


def test_analyze_http_end_to_end(client):
    """The full HTTP path: request → normalize → orchestrate 13 layers → response."""
    r = client.post("/analyze", json=CAMPAIGN)
    assert r.status_code == 200
    j = r.json()
    assert set(j["reports"]) == {"founder", "investor", "memo", "executive"}
    assert isinstance(j["committee_recommendation"], str) and j["committee_recommendation"]
    assert 0.0 <= j["overall_confidence"] <= 1.0
    assert j["honesty_note"]
    # honesty survives to every report's output
    for kind, rep in j["reports"].items():
        assert "deferred_appendix" in rep, f"{kind} lost its deferred appendix"


def test_bundle_every_stage_executed(client):
    """Inspect the ReportBundle: each layer must have produced output (reuses the warm registry)."""
    from src.api.schemas.campaign import CampaignRequest
    reg = client.app.state.registry
    camp = CampaignRequest(**CAMPAIGN).to_campaign()
    bundle = reg.reporting.build_bundle(camp)
    vi, decision, committee = bundle["vi"], bundle["decision"], bundle["committee"]

    for stage in ["prediction", "drivers", "counterfactual", "scenario", "evolution", "cohort", "copilot"]:
        assert stage in vi and vi[stage], f"missing/empty stage: vi.{stage}"
    assert "funding_probability" in vi["prediction"]
    assert 0.0 <= vi["prediction"]["funding_probability"] <= 1.0

    for key in ["investment_readiness", "confidence", "recommendations", "priority_improvements"]:
        assert key in decision, f"missing decision.{key}"
    assert committee["committee_recommendation"]


def test_deferred_honesty_survives_end_to_end(client):
    """The deferred (never-fabricated) fields are present from the deep pipeline to the decision."""
    from src.api.schemas.campaign import CampaignRequest
    reg = client.app.state.registry
    bundle = reg.reporting.build_bundle(CampaignRequest(**CAMPAIGN).to_campaign())
    vi, decision = bundle["vi"], bundle["decision"]

    assert vi.get("deferred"), "vi lost its top-level deferred honesty boundary"
    assert decision.get("deferred"), "decision lost its deferred honesty boundary"
    # a deep deferred field (Day-9 copilot risk analysis) must still be present
    assert "deferred_risks" in vi["copilot"]["risk_analysis"]
