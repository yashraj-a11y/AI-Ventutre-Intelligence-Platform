"""Day 14 · Phase H — Error taxonomy, handlers, and request-scoped logging.

A small typed error hierarchy so failure modes map to the right HTTP status, plus FastAPI handlers
that turn any error into a clean JSON envelope with a **request id** (never a leaked traceback), and
a middleware that tags every request with an id and logs method/path/status/latency.

    ValidationError (fabricated / malformed input)  -> 422  (pydantic, automatic)
    ArtifactError   (model / retriever unavailable)  -> 503
    InferenceError  (a pipeline failed)              -> 500
    <anything else unexpected>                       -> 500  (logged, generic message out)
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.utils.logger import get_logger

logger = get_logger("avip.api")


class APIError(Exception):
    """Base backend error carrying an HTTP status + a stable error code."""
    status_code = 500
    error = "internal_error"

    def __init__(self, detail: str = "internal error") -> None:
        self.detail = detail
        super().__init__(detail)


class ArtifactError(APIError):
    """A required model/artifact is unavailable (e.g. the faiss retriever isn't loaded)."""
    status_code = 503
    error = "artifact_unavailable"


class InferenceError(APIError):
    """A pipeline/inference call failed while processing a valid request."""
    status_code = 500
    error = "inference_error"


def setup_error_handling(app: FastAPI) -> None:
    """Attach the request-id middleware + exception handlers to the app."""

    @app.middleware("http")
    async def _request_context(request: Request, call_next):
        rid = uuid.uuid4().hex[:8]
        request.state.request_id = rid
        t = time.time()
        try:
            response = await call_next(request)
        except Exception:                                     # log the 500 path, then let handlers map it
            logger.info("[%s] %s %s -> 500 (%dms)", rid, request.method, request.url.path,
                        int((time.time() - t) * 1000))
            raise
        response.headers["X-Request-ID"] = rid
        logger.info("[%s] %s %s -> %s (%dms)", rid, request.method, request.url.path,
                    response.status_code, int((time.time() - t) * 1000))
        return response

    @app.exception_handler(APIError)
    async def _handle_api_error(request: Request, exc: APIError):
        rid = getattr(request.state, "request_id", "-")
        logger.warning("[%s] %s: %s", rid, exc.error, exc.detail)
        return JSONResponse(status_code=exc.status_code,
                            content={"error": exc.error, "detail": exc.detail, "request_id": rid})

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "-")
        logger.exception("[%s] unhandled error: %s", rid, exc)   # full traceback server-side only
        return JSONResponse(status_code=500,
                            content={"error": "inference_error",
                                     "detail": "An internal error occurred while processing the request.",
                                     "request_id": rid})
