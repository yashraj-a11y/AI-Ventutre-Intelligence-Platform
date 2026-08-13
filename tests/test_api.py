"""Day 14 · Phase I — API test suite (FastAPI TestClient).

Integration tests over the live app: status codes, response-schema shapes, the /analyze structure,
determinism, and the error taxonomy (422 / 503). The client is module-scoped so the warm registry
loads once (~seconds) for the whole suite.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Every API test loads the warm registry (models + 407 MB FAISS index) via the app lifespan — all
# artifact-dependent, so the whole module is `heavy` and runs locally, not in the CI fast lane (Day-17).
pytestmark = pytest.mark.heavy

CAMPAIGN = {
    "name": "Artificer - Board Game", "main_category": "Games", "usd_goal_real": 30000.0,
    "launched": "2016-05-01 12:00:00", "deadline": "2016-05-31 12:00:00",
}


@pytest.fixture(scope="module")
def client():
    """Load the app + warm registry once for the module (lifespan runs on __enter__)."""
    from src.api.main import app
    with TestClient(app) as c:
        yield c


# ----------------------------------------------------------------- system
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["loaded"] is True


def test_docs_and_openapi(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    for path in ("/predict", "/explain", "/retrieve", "/analyze", "/report/founder"):
        assert path in spec["paths"]


def test_request_id_header(client):
    r = client.post("/predict", json=CAMPAIGN)
    assert "x-request-id" in {k.lower() for k in r.headers}


# ----------------------------------------------------------------- endpoints (200 + shape)
def test_predict(client):
    r = client.post("/predict", json=CAMPAIGN)
    assert r.status_code == 200
    j = r.json()
    assert 0.0 <= j["funding_probability"] <= 1.0
    assert j["predicted_label"] in ("successful", "failed")
    assert 0.0 < j["decision_threshold"] < 1.0


def test_explain(client):
    j = client.post("/explain", json=CAMPAIGN).json()
    assert len(j["top_positive"]) > 0 and len(j["top_negative"]) > 0


def test_retrieve(client):
    r = client.post("/retrieve?k=5", json=CAMPAIGN)
    assert r.status_code == 200
    assert len(r.json()["similar"]) == 5


def test_counterfactual(client):
    j = client.post("/counterfactual", json=CAMPAIGN).json()
    assert "ranked_recommendations" in j["result"]


def test_scenario(client):
    j = client.post("/scenario", json=CAMPAIGN).json()
    assert "best_scenario" in j["result"]


def test_recommend(client):
    j = client.post("/recommend", json=CAMPAIGN).json()
    assert "investment_readiness" in j and "priority_improvements" in j


@pytest.mark.parametrize("audience", ["founder", "investor"])
def test_report(client, audience):
    r = client.post(f"/report/{audience}", json=CAMPAIGN)
    assert r.status_code == 200
    j = r.json()
    assert len(j["sections"]) > 0
    assert all(s["source"] for s in j["sections"])          # provenance total (no-new-claims)
    assert "deferred_appendix" in j


def test_analyze(client):
    r = client.post("/analyze", json=CAMPAIGN)
    assert r.status_code == 200
    j = r.json()
    assert set(j["reports"]) == {"founder", "investor", "memo", "executive"}
    assert isinstance(j["committee_recommendation"], str)
    assert 0.0 <= j["overall_confidence"] <= 1.0


# ----------------------------------------------------------------- validation (422)
def test_reject_fabricated_field(client):
    r = client.post("/predict", json={**CAMPAIGN, "mrr": 20000, "founder_experience": "expert"})
    assert r.status_code == 422
    assert r.json()["detail"][0]["type"] == "extra_forbidden"


def test_reject_missing_field(client):
    r = client.post("/predict", json={"main_category": "Games", "usd_goal_real": 5000,
                                      "launched": "2016-05-01", "deadline": "2016-06-01"})
    assert r.status_code == 422


def test_reject_bad_date_order(client):
    r = client.post("/predict", json={**CAMPAIGN, "launched": "2016-06-01", "deadline": "2016-05-01"})
    assert r.status_code == 422


def test_reject_nonpositive_goal(client):
    r = client.post("/predict", json={**CAMPAIGN, "usd_goal_real": -5})
    assert r.status_code == 422


# ----------------------------------------------------------------- determinism + errors
def test_determinism(client):
    a = client.post("/predict", json=CAMPAIGN).json()
    b = client.post("/predict", json=CAMPAIGN).json()
    assert a == b


def test_503_when_artifact_unavailable(client):
    """ArtifactError -> 503 with a typed body + request id (restore the retriever afterward)."""
    reg = client.app.state.registry
    saved = reg.retriever
    reg.retriever = None
    try:
        r = client.post("/retrieve", json=CAMPAIGN)
        assert r.status_code == 503
        j = r.json()
        assert j["error"] == "artifact_unavailable"
        assert "request_id" in j
    finally:
        reg.retriever = saved


# ----------------------------------------------------------------- response time (Day-17 Phase G)
# Warm bounds ~3x measured (health 0.8 / predict 12 / explain 31 / retrieve 20 / recommend 1120 /
# analyze 3226 ms) — loose enough to never flake, tight enough to catch a hung/regressed endpoint.
_TIME_BUDGET_MS = {"/health": 1000, "/predict": 2000, "/explain": 3000,
                   "/retrieve": 3000, "/recommend": 10000, "/analyze": 30000}


def test_response_times_within_budget(client):
    import json as _json
    import time
    from pathlib import Path

    def _call(ep):
        return client.get(ep) if ep == "/health" else client.post(ep, json=CAMPAIGN)

    measured, failures = {}, []
    for ep, budget in _TIME_BUDGET_MS.items():
        _call(ep)                                        # warm
        t = time.perf_counter()
        r = _call(ep)
        dt = (time.perf_counter() - t) * 1000
        measured[ep] = round(dt, 1)
        assert r.status_code == 200
        if dt > budget:
            failures.append(f"{ep} {dt:.0f}ms > {budget}ms")

    out = Path(__file__).resolve().parents[1] / "artifacts/testing/api_response_times.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps({"measured_ms": measured, "budget_ms": _TIME_BUDGET_MS,
                                "passed": not failures}, indent=2))
    assert not failures, failures
