"""Day 14 · Phase F — POST /recommend (Day-11)."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from src.api.dependencies import get_registry
from src.api.schemas import CampaignRequest, RecommendationResponse
from src.services import recommendation_service

router = APIRouter(tags=["intelligence"])


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(campaign: CampaignRequest, registry=Depends(get_registry)):
    return recommendation_service.recommend(registry, campaign)
