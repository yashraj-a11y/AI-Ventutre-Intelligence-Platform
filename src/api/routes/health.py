"""Day 14 · Phase F — system routes (/health)."""
from __future__ import annotations
from typing import Any
from fastapi import APIRouter, Request

router = APIRouter(tags=["system"])


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Readiness probe — reports registry status without requiring it to be ready."""
    reg = getattr(request.app.state, "registry", None)
    status = reg.status() if reg is not None else {"loaded": False}
    return {"status": "ok" if (reg is not None and reg.ready()) else "loading", **status}
