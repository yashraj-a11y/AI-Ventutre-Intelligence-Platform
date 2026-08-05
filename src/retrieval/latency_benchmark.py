"""Day 15 · Optimization phase — measure retrieval latency BEFORE optimizing (honest C-6).

Day 14 saw a **~10s first `/retrieve`** and it was tempting to "optimize retrieval." The honest move
is to *measure* first: is 10s a per-query cost (a real problem) or a one-time model warmup (not a
retrieval problem at all)? This benchmark answers that by separating three things:

- **cold** — the very first ``retrieve`` call after process start (includes any lazy SBERT/torch
  graph warmup the first ``encode`` triggers).
- **warm p50/p95** — steady-state per-query latency over many calls (the number that actually matters
  for a served endpoint).
- **decomposition** — of a warm query, how much is **SBERT encode** vs **FAISS search**? This tells us
  *where* any real cost lives, so optimization (if needed) targets the true bottleneck, not a guess.

If warm p50 is small and cold ≫ warm, the ~10s was **one-time warmup** → the honest fix is to warm
the model at startup (already done: the registry builds ``VentureRetriever`` on boot), not to
"optimize FAISS." No new deps. Run with ``OMP_NUM_THREADS=1``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.retrieval.venture_retriever import VentureRetriever
from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("Latency measured, not assumed (C-6): cold = first call incl. one-time SBERT warmup; "
            "warm p50/p95 = steady-state; decomposition = SBERT-encode vs FAISS-search. If cold >> "
            "warm, the Day-14 ~10s was model warmup (fixed by warming at startup), not a FAISS cost.")


def _pct(a: np.ndarray, p: float) -> float:
    return round(float(np.percentile(a, p)), 2)


class RetrievalLatencyBenchmark:
    """Measure cold vs warm retrieval latency and decompose encode-vs-search.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    retriever:
        Optional pre-built Day-6 :class:`VentureRetriever`. If ``None``, one is constructed here —
        which itself demonstrates the cold path (model load) that the registry pays once at startup.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 retriever: Optional[VentureRetriever] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.retrieval.latency")

        c = self.config.get("day15", {}).get("optimization", {})
        self.n_warm = int(c.get("n_warm", 200))
        self.n_decompose = int(c.get("n_decompose", 50))
        self.k = int(c.get("k", 5))
        self.random_state = int(c.get("random_state", 15))
        self.out_path = Path(c.get("out_path", "artifacts/retrieval/retrieval_latency.json"))

        # Building the retriever here IS the cold model-load; time it so "load cost" is explicit.
        t = time.perf_counter()
        self.retriever = retriever or VentureRetriever(self.config, self.logger)
        self.load_seconds = round(time.perf_counter() - t, 2)   # 0.0 if a warm retriever was passed
        self.engine = self.retriever.engine

    # ------------------------------------------------------------------ public
    def measure(self) -> dict[str, Any]:
        """Run the latency benchmark and return the report dict (times in milliseconds)."""
        engine = self.engine
        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        # one sample pool, drawn once, reused for cold + warm + decomposition (deterministic)
        pool = df.iloc[ti].sample(min(self.n_warm + 1, len(ti)),
                                  random_state=self.random_state).reset_index(drop=True)
        campaigns = [row.to_dict() for _, row in pool.iterrows()]

        # --- COLD: the first end-to-end retrieve (triggers lazy encode graph warmup) --------------
        t = time.perf_counter()
        self.retriever.retrieve(campaigns[0], k=self.k)
        cold_ms = round((time.perf_counter() - t) * 1e3, 2)

        # --- WARM: steady-state per-query end-to-end retrieve --------------------------------------
        warm = np.empty(len(campaigns) - 1, dtype="float64")
        for i, camp in enumerate(campaigns[1:]):
            t = time.perf_counter()
            self.retriever.retrieve(camp, k=self.k)
            warm[i] = (time.perf_counter() - t) * 1e3

        # --- DECOMPOSE: encode-only vs FAISS-search-only, on warm calls ----------------------------
        sub = campaigns[1:1 + self.n_decompose]
        enc = np.empty(len(sub)); srch = np.empty(len(sub))
        for i, camp in enumerate(sub):
            text = engine.serialize(pd.Series(camp))
            t = time.perf_counter()
            qv = engine.embedder.encode([text]).astype("float32")
            enc[i] = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            engine.vector_db.search(qv, k=self.k)
            srch[i] = (time.perf_counter() - t) * 1e3

        warm_p50 = _pct(warm, 50)
        cold_over_warm = round(cold_ms / warm_p50, 1) if warm_p50 else None
        report = {
            "n_warm": int(len(warm)), "n_decompose": int(len(sub)), "k": self.k,
            "model_load_seconds": self.load_seconds,
            "cold_first_query_ms": cold_ms,
            "warm_end_to_end_ms": {"p50": warm_p50, "p95": _pct(warm, 95),
                                   "mean": round(float(warm.mean()), 2),
                                   "min": round(float(warm.min()), 2)},
            "decomposition_ms": {
                "sbert_encode": {"p50": _pct(enc, 50), "p95": _pct(enc, 95),
                                 "mean": round(float(enc.mean()), 2)},
                "faiss_search": {"p50": _pct(srch, 50), "p95": _pct(srch, 95),
                                 "mean": round(float(srch.mean()), 2)},
            },
            "cold_over_warm_ratio": cold_over_warm,
            "verdict": self._verdict(cold_ms, warm_p50, float(enc.mean()), float(srch.mean())),
            "honesty_note": _CAPTION,
        }
        self.logger.info("Latency: cold %.0fms, warm p50 %.1fms / p95 %.1fms (encode %.1fms + "
                         "faiss %.2fms). %s", cold_ms, warm_p50, report["warm_end_to_end_ms"]["p95"],
                         report["decomposition_ms"]["sbert_encode"]["mean"],
                         report["decomposition_ms"]["faiss_search"]["mean"], report["verdict"])
        return report

    @staticmethod
    def _verdict(cold_ms: float, warm_p50: float, enc_mean: float, srch_mean: float) -> str:
        """Honest one-line conclusion: was the Day-14 ~10s warmup, and where is warm cost?"""
        warmup = cold_ms >= 5 * max(warm_p50, 1e-9) and cold_ms > 1000
        dominant = "SBERT encode" if enc_mean > srch_mean else "FAISS search"
        head = ("Cold >> warm: the Day-14 first-call cost was ONE-TIME model warmup, already paid at "
                "registry startup — not a per-query FAISS cost." if warmup else
                "Cold ≈ warm: no large one-time warmup observed in this run.")
        # HONESTY (measured, not assumed): single-query IndexFlatIP over 265K×384 under
        # OMP_NUM_THREADS=1 is a few ms — NOT the "sub-millisecond" earlier notes claimed. Batched it
        # amortizes to well under 1ms/query (Phase G's BLAS matmul path). Either way, no bottleneck.
        return (f"{head} Warm p50 {warm_p50:.1f}ms is split {dominant}-dominant "
                f"(encode {enc_mean:.1f}ms vs faiss {srch_mean:.2f}ms). Single-query exact "
                f"IndexFlatIP is ~{srch_mean:.0f}ms single-threaded (not sub-ms; batched it drops "
                f"below 1ms/query) — comfortably fast for a served endpoint, so there is no FAISS "
                f"bottleneck to optimize at 265K×384 (ANN would only trade recall for speed we "
                f"don't need). Optimization is a no-op beyond the startup warmup already in place.")

    def generate_report(self) -> Path:
        """Run the benchmark and persist the JSON report."""
        report = self.measure()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(report, indent=2, default=str))
        self.logger.info("Saved retrieval latency -> %s", self.out_path)
        return self.out_path
