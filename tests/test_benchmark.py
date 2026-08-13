"""Day 17 · Phase I — performance benchmark (latency percentiles + memory/CPU).

`heavy`. Measures warm per-endpoint latency (p50/p95 over N runs) for the key layers, plus the process
resident memory and a CPU snapshot via `psutil`. Numbers are recorded to `artifacts/testing/benchmark.json`;
the assertions are generous sanity bounds (catch a hang or a memory blow-up, don't flake on a busy box).
Retrieval latency was characterised in depth on Day 15 (warm p50 ~14 ms end-to-end) — re-measured here
through the HTTP layer.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.heavy

CAMPAIGN = {
    "name": "Artificer - Board Game", "main_category": "Games", "usd_goal_real": 30000.0,
    "launched": "2016-05-01 12:00:00", "deadline": "2016-05-31 12:00:00",
}
_BUDGET_MS = {"predict": 2000, "explain": 3000, "retrieve": 3000, "report_founder": 15000, "analyze": 30000}


def test_performance_benchmark():
    proc = psutil.Process()
    with TestClient(__import__("src.api.main", fromlist=["app"]).app) as c:
        rss_mb = round(proc.memory_info().rss / (1 << 20), 1)          # after warm load
        ops = {
            "predict": lambda: c.post("/predict", json=CAMPAIGN),
            "explain": lambda: c.post("/explain", json=CAMPAIGN),
            "retrieve": lambda: c.post("/retrieve?k=5", json=CAMPAIGN),
            "report_founder": lambda: c.post("/report/founder", json=CAMPAIGN),
            "analyze": lambda: c.post("/analyze", json=CAMPAIGN),
        }
        latency: dict[str, dict] = {}
        proc.cpu_percent(None)                                         # prime the CPU counter
        for name, call in ops.items():
            call()                                                    # warm
            n = 5 if name == "analyze" else 20                        # analyze is ~3 s/call
            samples = np.empty(n)
            for i in range(n):
                t = time.perf_counter()
                r = call()
                samples[i] = (time.perf_counter() - t) * 1000
                assert r.status_code == 200
            latency[name] = {"p50": round(float(np.percentile(samples, 50)), 1),
                             "p95": round(float(np.percentile(samples, 95)), 1),
                             "mean": round(float(samples.mean()), 1), "n": n}
        cpu_percent = proc.cpu_percent(None)                          # utilisation over the run

    report = {
        "latency_ms": latency, "memory_rss_mb": rss_mb, "cpu_percent": cpu_percent,
        "n_cpus": psutil.cpu_count(),
        "notes": {"retrieval": "Day-15 characterised warm retrieval at p50 ~14 ms end-to-end "
                               "(SBERT encode ~4.4 ms + FAISS ~6.4 ms); here it's through the HTTP layer",
                  "omp": "OMP_NUM_THREADS=1 (faiss + torch coexistence)"},
        "budgets_ms": _BUDGET_MS,
        "passed": all(latency[k]["p50"] < _BUDGET_MS[k] for k in _BUDGET_MS),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = Path(__file__).resolve().parents[1] / "artifacts/testing/benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))

    over = [f"{k} p50 {latency[k]['p50']}ms > {_BUDGET_MS[k]}ms" for k in _BUDGET_MS
            if latency[k]["p50"] >= _BUDGET_MS[k]]
    assert not over, over
    assert rss_mb < 8000, f"process RSS {rss_mb} MB unexpectedly high"
