"""Day 14 · Phase B/H — FastAPI application entrypoint.

The single backend that exposes Architecture V1.0 over REST.

**CRITICAL — the first two lines set ``OMP_NUM_THREADS=1`` BEFORE any heavy import** (numpy /
lightgbm / faiss / torch read it at import time). Together with the model registry's load order
(lightgbm before faiss, Phase D) this is what lets ``/analyze`` (lightgbm) and ``/retrieve`` (faiss)
coexist in one process. Do not move or remove these lines.

Boot:  ``OMP_NUM_THREADS=1 PYTHONPATH=. ./.venv/bin/uvicorn src.api.main:app --port 8000``
Docs:  ``GET /docs`` (Swagger UI, auto-generated) · ``GET /health`` (readiness).
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # MUST precede numpy/lightgbm/faiss/torch imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from contextlib import asynccontextmanager       # noqa: E402
from typing import Any                            # noqa: E402

from fastapi import FastAPI                        # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from src.utils.config import load_config           # noqa: E402
from src.utils.logger import get_logger            # noqa: E402

logger = get_logger("avip.api")
_cfg = load_config()
_api = _cfg.get("day14", {}).get("api", {})


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the warm stack ONCE at startup (lightgbm before faiss); tear down on shutdown."""
    logger.info("API starting — OMP_NUM_THREADS=%s", os.environ.get("OMP_NUM_THREADS"))
    app.state.config = _cfg
    from src.core.model_loader import get_registry     # deferred: import AFTER OMP is set
    app.state.registry = get_registry(_cfg)            # singleton warm stack (~seconds, once)
    app.state.ready = True
    logger.info("API ready — registry loaded.")
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title=_api.get("title", "AI Venture Intelligence Platform API"),
    version=_api.get("version", "1.0.0"),
    description=("Production backend over Architecture V1.0 (Days 1-13). Deterministic, honesty-tiered "
                "outputs; requests accept real Kickstarter campaign fields only."),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_api.get("cors_origins", ["*"]),
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Phase H — request-id middleware + typed exception handlers (422 / 503 / 500).
from src.core.exceptions import setup_error_handling   # noqa: E402

setup_error_handling(app)


@app.get("/", tags=["system"])
def root() -> dict[str, Any]:
    """Service banner + where to find the docs."""
    return {"service": _api.get("title"), "docs": "/docs", "health": "/health"}


# Phase F — include every endpoint router (predict / explain / retrieve / … / analyze / report).
from src.api.routes import routers   # noqa: E402  (import after app + OMP set)

for _router in routers:
    app.include_router(_router)
