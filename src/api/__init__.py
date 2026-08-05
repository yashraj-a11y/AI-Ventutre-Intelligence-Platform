"""Day 14 — Production Backend / API layer.

A single FastAPI backend that exposes the whole Architecture-V1.0 platform (Days 1-13) over REST.
Thin routes validate a request and call a service; services are thin wrappers over the EXISTING
pipelines (Day-4 prediction … Day-13 reporting); a singleton model registry loads one warm stack
once at startup.

**Critical runtime rule:** ``OMP_NUM_THREADS=1`` must be set at the very top of the entrypoint
(before any import), and the lightgbm stack must be loaded *before* the Day-6 faiss retriever —
otherwise lightgbm + faiss+torch collide in one process (segfault/hang). Preflight-verified: with the
mitigation, ``/analyze`` (lightgbm) and ``/retrieve`` (faiss) coexist.

Subpackages
-----------
- ``schemas`` — pydantic request/response contracts (real Kickstarter fields only).
- ``routes``  — one thin router per endpoint group.
- ``main``    — the FastAPI app (OMP set, lifespan-loaded registry, CORS, exception handlers).
- ``dependencies`` — FastAPI DI providers (get the registry / services).
"""
