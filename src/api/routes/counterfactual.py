"""Day 14 · Phase F — POST /counterfactual (Day-8)."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from src.api.dependencies import get_registry
from src.api.schemas import CampaignRequest, GenericResponse
from src.services import counterfactual_service

router = APIRouter(tags=["intelligence"])


@router.post("/counterfactual", response_model=GenericResponse)
def counterfactual(campaign: CampaignRequest, registry=Depends(get_registry)):
    return counterfactual_service.counterfactual(registry, campaign)
