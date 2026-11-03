"""Day 14 · Phase D — Model & Artifact Loader (the singleton warm-stack registry).

Load every serialized component **once** at startup and hold it in one ``ModelRegistry``, in the
**mandatory load order**: the lightgbm stack (Days 4-13) FIRST, the Day-6 faiss retriever LAST. With
``OMP_NUM_THREADS=1`` (set by the API entrypoint) this is what lets lightgbm + faiss+torch coexist in
one process (preflight-verified).

The elegance: the whole lightgbm stack is reachable from **one** Day-8 ``CounterfactualPipeline`` —
the Day-13 ``ReportingPipeline`` composes Day-11/12 on it, and the Day-11 aggregator already composed
Day-5 SHAP, Day-9 copilot, Day-10 evolution/scenario on the *same* cf. So one cf → the entire stack,
one model in memory.

Usage
-----
>>> reg = get_registry()          # builds once, cached (singleton)
>>> reg.engine.predict_proba(...) # Day-4 ; reg.reporting.run(...) ; reg.retriever.retrieve(...)
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # belt-and-suspenders if imported directly

import time                                             # noqa: E402
from typing import Any, Optional                        # noqa: E402

from src.utils.config import load_config                # noqa: E402
from src.utils.logger import get_logger                 # noqa: E402


class ModelRegistry:
    """One warm stack: shared cf → Day-11/12/13 pipelines, then the Day-6 faiss retriever.

    Parameters
    ----------
    config, logger:
        Standard config + logger (reused from ``src.utils``).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.core.registry")
        self.load_retriever = bool(self.config.get("day14", {}).get("api", {}).get("load_retriever", True))
        self._loaded = False
        # handles (populated by load())
        self.cf = self.reporting = self.rec = self.consensus = self.agg = None
        self.engine = self.explainer = self.scenario = self.evolution = self.retriever = None

    # ------------------------------------------------------------------ public
    def load(self) -> "ModelRegistry":
        """Build the warm stack once, in the mandatory order (lightgbm before faiss)."""
        if self._loaded:
            return self
        t0 = time.time()

        # ---- 1. lightgbm stack FIRST (one cf -> the whole Day 4-13 stack) --------------------
        from src.counterfactual.pipeline import CounterfactualPipeline
        from src.reporting.pipeline import ReportingPipeline
        self.cf = CounterfactualPipeline(self.config, self.logger)          # Day-8 (model + item frame)
        self.reporting = ReportingPipeline(self.config, self.logger, cf_pipeline=self.cf)  # Day-13
        self.rec = self.reporting.rec                                       # Day-11 RecommendationPipeline
        self.consensus = self.reporting.consensus                          # Day-12 ConsensusEngine
        self.agg = self.rec.agg                                            # Day-11 IntelligenceAggregator
        self.engine = self.cf.simulator.engine                            # Day-4 PredictionEngine
        self.explainer = self.agg.lx                                       # Day-5 LocalExplainer
        self.scenario = self.agg.scn                                       # Day-10 ScenarioPipeline
        self.evolution = self.agg.evo                                      # Day-10 EvolutionPipeline
        t_light = time.time() - t0

        # ---- 2. faiss retriever LAST (deferred import -> load order enforced) ----------------
        if self.load_retriever:
            from src.retrieval.venture_retriever import VentureRetriever    # faiss + SBERT/torch
            self.retriever = VentureRetriever(self.config, self.logger)

        self._loaded = True
        _release_freed_memory()
        self.logger.info("ModelRegistry ready in %.1fs (lightgbm %.1fs, retriever=%s, OMP=%s).",
                         time.time() - t0, t_light, bool(self.retriever),
                         os.environ.get("OMP_NUM_THREADS"))
        return self

    def ready(self) -> bool:
        return self._loaded

    def status(self) -> dict[str, Any]:
        """Which components are loaded (for /health)."""
        return {"loaded": self._loaded, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "components": {"prediction": self.engine is not None,
                               "reporting": self.reporting is not None,
                               "recommendation": self.rec is not None,
                               "committee": self.consensus is not None,
                               "retriever": self.retriever is not None}}


def _release_freed_memory() -> None:
    """Hand memory freed during startup back to the OS (glibc/Linux only; a no-op elsewhere).

    Loading the stack creates large temporaries that Python frees, but glibc keeps the pages in its
    heap, so the process still *looks* that big to the host's memory limit. ``malloc_trim(0)`` only
    returns already-free pages — it never touches live objects.
    """
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):                  # not glibc (macOS, musl): nothing to do
        pass


# ------------------------------------------------------------------ module singleton
_REGISTRY: Optional[ModelRegistry] = None


def get_registry(config: Optional[dict[str, Any]] = None) -> ModelRegistry:
    """Return the process-wide registry, building it once (singleton)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ModelRegistry(config).load()
    return _REGISTRY
