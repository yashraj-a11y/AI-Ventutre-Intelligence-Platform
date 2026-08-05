"""Day 14 · Phase F — POST /explain (Day-5 SHAP)."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from src.api.dependencies import get_registry
from src.api.schemas import CampaignRequest, ExplanationResponse
from src.services import explainability_service

router = APIRouter(tags=["intelligence"])


@router.post("/explain", response_model=ExplanationResponse)
def explain(campaign: CampaignRequest, registry=Depends(get_registry)):
    return explainability_service.explain(registry, campaign)
