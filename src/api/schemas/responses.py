"""Day 14 · Phase C — Response schemas (typed, honesty-aware output contracts).

Typed envelopes for each endpoint. Heavy nested content (report sections, agent votes) is carried as
``dict``/``list`` — the pipelines already produce validated, honesty-tiered structures, so the API
types the *envelope* and passes the body through unchanged (no re-shaping = no drift). Every envelope
that can defer carries a ``deferred_appendix``/``honesty_note`` so the boundary never hides the
platform's limits.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """Day-4 funding prediction."""
    campaign: dict[str, Any]
    funding_probability: float
    predicted_label: str
    decision_threshold: float


class ExplanationResponse(BaseModel):
    """Day-5 SHAP local explanation (top +/- real-feature drivers)."""
    funding_probability: float
    predicted_label: str
    top_positive: list[dict[str, Any]]
    top_negative: list[dict[str, Any]]
    base_value: Optional[float] = None   # expected log-odds margin (phi0); waterfall start
    margin: Optional[float] = None       # this campaign's log-odds (phi0 + sum of contributions)
    honesty_note: Optional[str] = None


class RetrievalResponse(BaseModel):
    """Day-6 semantic retrieval (similar campaigns from the knowledge base)."""
    query: dict[str, Any]
    similar: list[dict[str, Any]]
    similar_successful: list[dict[str, Any]] = []
    similar_failed: list[dict[str, Any]] = []
    honesty_note: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Day-11 unified venture decision (readiness + prioritized actions + confidence)."""
    campaign: dict[str, Any]
    investment_readiness: dict[str, Any]
    confidence: dict[str, Any]
    priority_improvements: list[dict[str, Any]]
    investor_strategy: dict[str, Any]
    final_strategy: str
    deferred: dict[str, Any]
    honesty_note: str


class ReportResponse(BaseModel):
    """A Day-13 report (founder / investor) — typed envelope, sections passed through."""
    report_type: str
    campaign: dict[str, Any]
    sections: list[dict[str, Any]]
    deferred_appendix: dict[str, Any]
    honesty_note: str


class AnalysisResponse(BaseModel):
    """The ``/analyze`` orchestrator — committee verdict + the full report suite + copilot."""
    campaign: dict[str, Any]
    committee_recommendation: str
    overall_confidence: float
    reports: dict[str, Any]
    copilot: dict[str, Any]
    honesty_note: str


class GenericResponse(BaseModel):
    """Fallback envelope for endpoints whose body is a pipeline dict (scenario / counterfactual)."""
    campaign: dict[str, Any]
    result: dict[str, Any]
    honesty_note: Optional[str] = None
