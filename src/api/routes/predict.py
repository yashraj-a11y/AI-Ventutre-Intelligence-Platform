"""Day 14 · Phase F — POST /predict (Day-4)."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from src.api.dependencies import get_registry
from src.api.schemas import CampaignRequest, PredictionResponse
from src.services import prediction_service

router = APIRouter(tags=["intelligence"])


@router.post("/predict", response_model=PredictionResponse)
def predict(campaign: CampaignRequest, registry=Depends(get_registry)):
    return prediction_service.predict(registry, campaign)
