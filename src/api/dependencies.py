"""Day 14 · Phase F — FastAPI dependency providers.

Dependency injection wiring: routes declare ``registry = Depends(get_registry)`` and receive the
lifespan-loaded singleton from ``app.state`` — never importing a global. If the registry isn't ready
(startup still running / load failed), requests get a clean ``503`` instead of an ``AttributeError``.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from src.core.model_loader import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    """Provide the warm model registry from app state (503 until ready)."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None or not registry.ready():
        raise HTTPException(status_code=503, detail="Model registry is not ready yet.")
    return registry
