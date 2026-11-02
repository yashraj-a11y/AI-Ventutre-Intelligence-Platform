"""Day 14 · Phase F + Day 18 · Phase F — system routes (/health, /live, /ready).

Three probes, each with a distinct job (the orchestrator-friendly split):

- **/live** — liveness: is the process up? Always 200 while running, even during the ~tens-of-seconds
  model load — so an orchestrator does not kill a container that is merely still warming up.
- **/ready** — readiness: is the warm stack loaded and able to serve? 200 when ``reg.ready()``, else
  **503** — so a load balancer routes traffic only once the models + FAISS index are in memory.
- **/health** — the human/diagnostic view: registry component status + disk/memory/CPU via psutil.

There is no separate "vector DB" or "MLflow" health check: FAISS is in-process (its readiness IS
``reg.ready()``), and MLflow is a training-time local store, not a serving dependency.
"""
from __future__ import annotations

import shutil
from typing import Any

import psutil
from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["system"])


def _system_diagnostics() -> dict[str, Any]:
    """Cheap host/process resource snapshot for the diagnostic /health view."""
    vm = psutil.virtual_memory()
    du = shutil.disk_usage("/")
    proc = psutil.Process()
    return {
        "process_rss_mb": round(proc.memory_info().rss / (1 << 20), 1),
        "memory_percent": vm.percent,
        "memory_available_mb": round(vm.available / (1 << 20), 1),
        "disk_percent": round(du.used / du.total * 100, 1),
        "disk_free_gb": round(du.free / (1 << 30), 2),
        "cpu_percent": psutil.cpu_percent(interval=None),
    }


@router.get("/live")
def live() -> dict[str, Any]:
    """Liveness — the process is up. Always 200 (does not depend on the model load)."""
    return {"status": "alive"}


@router.get("/ready")
def ready(request: Request, response: Response) -> dict[str, Any]:
    """Readiness — the warm stack is loaded. 200 when ready, 503 while loading (probe-friendly)."""
    reg = getattr(request.app.state, "registry", None)
    is_ready = reg is not None and reg.ready()
    if not is_ready:
        response.status_code = 503
    return {"status": "ready" if is_ready else "loading"}


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Diagnostic health — registry component status + host/process resource usage."""
    reg = getattr(request.app.state, "registry", None)
    status = reg.status() if reg is not None else {"loaded": False}
    return {"status": "ok" if (reg is not None and reg.ready()) else "loading",
            **status, "system": _system_diagnostics()}
