"""Day 14 — API schemas (pydantic request/response contracts).

- :class:`~src.api.schemas.campaign.CampaignRequest` — the honest input (real fields only; fabricated
  fields are 422-rejected).
- :mod:`~src.api.schemas.responses` — typed response envelopes (honesty-aware).
"""

from src.api.schemas.campaign import CampaignRequest
from src.api.schemas.responses import (
    AnalysisResponse, ExplanationResponse, GenericResponse, PredictionResponse,
    RecommendationResponse, ReportResponse, RetrievalResponse,
)

__all__ = ["CampaignRequest", "PredictionResponse", "ExplanationResponse", "RetrievalResponse",
           "RecommendationResponse", "ReportResponse", "AnalysisResponse", "GenericResponse"]
