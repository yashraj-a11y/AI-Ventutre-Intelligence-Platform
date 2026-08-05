"""Day 14 · Phase F/G — POST /analyze (the full orchestrator, Day-13)."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from src.api.dependencies import get_registry
from src.api.schemas import CampaignRequest, AnalysisResponse
from src.services import pipeline_service

router = APIRouter(tags=["intelligence"])


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(campaign: CampaignRequest, registry=Depends(get_registry)):
    return pipeline_service.analyze(registry, campaign)
