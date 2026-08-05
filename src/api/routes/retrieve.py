"""Day 14 · Phase F — POST /retrieve (Day-6 faiss)."""
from __future__ import annotations
from fastapi import APIRouter, Depends, Query
from src.api.dependencies import get_registry
from src.api.schemas import CampaignRequest, RetrievalResponse
from src.services import retrieval_service

router = APIRouter(tags=["intelligence"])


@router.post("/retrieve", response_model=RetrievalResponse)
def retrieve(campaign: CampaignRequest, k: int = Query(5, ge=1, le=25), registry=Depends(get_registry)):
    return retrieval_service.retrieve(registry, campaign, k=k)
