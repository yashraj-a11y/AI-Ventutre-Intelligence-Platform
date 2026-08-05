"""Day 14 · Phase F — POST /report/founder, /report/investor (Day-13)."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from src.api.dependencies import get_registry
from src.api.schemas import CampaignRequest, ReportResponse
from src.services import report_service

router = APIRouter(prefix="/report", tags=["reports"])


@router.post("/founder", response_model=ReportResponse)
def founder(campaign: CampaignRequest, registry=Depends(get_registry)):
    return report_service.founder(registry, campaign)


@router.post("/investor", response_model=ReportResponse)
def investor(campaign: CampaignRequest, registry=Depends(get_registry)):
    return report_service.investor(registry, campaign)
