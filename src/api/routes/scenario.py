"""Day 14 · Phase F — POST /scenario (Day-10)."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from src.api.dependencies import get_registry
from src.api.schemas import CampaignRequest, GenericResponse
from src.services import scenario_service

router = APIRouter(tags=["intelligence"])


@router.post("/scenario", response_model=GenericResponse)
def scenario(campaign: CampaignRequest, registry=Depends(get_registry)):
    return scenario_service.scenario(registry, campaign)
