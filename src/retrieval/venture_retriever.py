"""Phase F — Historical Venture Retriever (Day-6, Architecture Layer 3 core).

The heart of the Venture Intelligence Engine. Given a query campaign it returns, all
grounded in **real** retrieved data:

- **similar** campaigns (any outcome),
- **similar successful** and **similar failed** campaigns (filtered by the real ``target``),
- a **funding benchmark** — goal statistics of the nearest neighbours, split by outcome,
  plus their success rate (a k-NN estimate of P(fund), the "retrieval-based prior"),
- a **category / industry benchmark** — success rate + goal medians for the query's
  ``main_category`` across the whole knowledge base,
- a **reason** per neighbour, built only from **shared real attributes**
  ("same category, same country, similar goal size") — never traction / founder profile.

**Honesty tiers** (same discipline as Day-4 scores / Day-5 copilot): founder, market,
valuation, and exit benchmarks have **no data** in this dataset, so they are returned as
explicit ``deferred`` placeholders — never fabricated.

Efficiency: one FAISS search of a ``benchmark_pool`` of neighbours feeds *all* of the
lists and the funding benchmark; the category benchmark is a cached KB aggregate.

Example
-------
>>> vr = VentureRetriever()
>>> ev = vr.retrieve({"name": "Handcrafted wooden board game", "category": "Tabletop Games",
...                   "main_category": "Games", "country": "US", "usd_goal_real": 8000})
>>> ev["funding_benchmark"]["success_rate"], ev["similar_successful"][0]["title"]
(0.46, '...')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from src.knowledge_base.venture_database import goal_band
from src.retrieval.semantic_search import SemanticSearchEngine
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED_REASON = ("no data basis in this dataset (no founder / team / market / valuation "
                    "/ exit signals) — awaits Layers 2–3 richer data; not fabricated")
_DEFERRED_BENCHMARKS = ["founder_benchmark", "market_benchmark",
                        "valuation_benchmark", "exit_benchmark"]


class VentureRetriever:
    """Retrieve similar campaigns, outcome-split neighbours, and honest benchmarks.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    engine:
        Optional pre-built :class:`SemanticSearchEngine` (share a warm index/model).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 engine: Optional[SemanticSearchEngine] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.retrieval.venture_retriever")

        c = self.config.get("day6", {}).get("venture_retriever", {})
        self.k = int(c.get("k", 5))
        self.benchmark_pool = int(c.get("benchmark_pool", 50))
        self.examples_path = Path(c.get("examples_path",
                                        "artifacts/retrieval/retrieval_examples.json"))

        self.engine = engine or SemanticSearchEngine(self.config, self.logger)
        self._category_stats: Optional[pd.DataFrame] = None   # cached KB aggregate

    # ------------------------------------------------------------------ public
    def retrieve(self, query: Union[dict, pd.Series], k: Optional[int] = None,
                 exclude_campaign_ids: Optional[list[int]] = None) -> dict[str, Any]:
        """Full evidence bundle for one query campaign (one search feeds all)."""
        k = k or self.k
        qa = self._query_attrs(query)
        pool = self.engine.search(query, k=self.benchmark_pool,
                                  exclude_campaign_ids=exclude_campaign_ids)
        for n in pool:                                        # attach real-attribute reasons
            n["reason"] = self._reason(qa, n)

        successful = [n for n in pool if n["outcome"] == "successful"]
        failed = [n for n in pool if n["outcome"] == "failed"]

        return {
            "query": qa,
            "similar": pool[:k],
            "similar_successful": successful[:k],
            "similar_failed": failed[:k],
            "funding_benchmark": self._funding_benchmark(pool, qa),
            "category_benchmark": self._category_benchmark(qa),
            "deferred_benchmarks": {b: {"status": "deferred", "reason": _DEFERRED_REASON}
                                    for b in _DEFERRED_BENCHMARKS},
        }

    def similar(self, query, k=None, **kw):
        """Top-k similar campaigns (any outcome)."""
        return self._with_reasons(query, self.engine.search(query, k=k or self.k, **kw))

    def similar_successful(self, query, k=None, **kw):
        """Top-k similar campaigns that funded."""
        return self._with_reasons(query, self.engine.search(
            query, k=k or self.k, filters={"outcome": "successful"}, **kw))

    def similar_failed(self, query, k=None, **kw):
        """Top-k similar campaigns that failed."""
        return self._with_reasons(query, self.engine.search(
            query, k=k or self.k, filters={"outcome": "failed"}, **kw))

    def generate_examples(self, n: int = 4, random_state: int = 5) -> Path:
        """Retrieve for a spread of real campaigns and save the deliverable JSON."""
        feat = ["name", "category", "main_category", "currency", "deadline",
                "goal", "launched", "country", "usd_goal_real", "name_clean"]
        processed = self.config.get("day6", {}).get("knowledge_base", {}).get(
            "processed_path", "data/processed/kickstarter_processed.csv")
        df = pd.read_csv(processed)
        sample = df.sample(n, random_state=random_state).reset_index(drop=True)
        bundles = [self.retrieve(row.to_dict()) for _, row in sample[feat].iterrows()]
        self.examples_path.parent.mkdir(parents=True, exist_ok=True)
        self.examples_path.write_text(json.dumps(bundles, indent=2, default=_json_safe))
        self.logger.info("Saved %d retrieval bundles -> %s", len(bundles), self.examples_path)
        return self.examples_path

    # --------------------------------------------------------------- internals
    def _with_reasons(self, query, neighbors):
        qa = self._query_attrs(query)
        for n in neighbors:
            n["reason"] = self._reason(qa, n)
        return neighbors

    def _query_attrs(self, query: Union[dict, pd.Series]) -> dict[str, Any]:
        q = dict(query)
        usd = float(q.get("usd_goal_real", q.get("goal", 0.0)))
        return {
            "title": str(q.get("name", q.get("title", ""))).strip(),
            "category": q.get("category"),
            "main_category": q.get("main_category"),
            "country": q.get("country"),
            "usd_goal_real": usd,
            "goal_band": goal_band(usd),
        }

    def _reason(self, qa: dict[str, Any], neighbor: dict[str, Any]) -> str:
        """A reason string from shared REAL attributes only."""
        shared = []
        if qa["category"] and neighbor["category"] == qa["category"]:
            shared.append(f"same category ({neighbor['category']})")
        elif qa["main_category"] and neighbor["main_category"] == qa["main_category"]:
            shared.append(f"same category group ({neighbor['main_category']})")
        if qa["country"] and neighbor["country"] == qa["country"]:
            shared.append(f"same country ({neighbor['country']})")
        if goal_band(neighbor["usd_goal_real"]) == qa["goal_band"]:
            shared.append(f"similar goal size ({qa['goal_band']})")
        if not shared:
            shared.append("semantically similar title")
        return ", ".join(shared)

    def _funding_benchmark(self, pool: list[dict], qa: dict[str, Any]) -> dict[str, Any]:
        """Goal stats + success rate (k-NN prior) over the neighbour pool."""
        if not pool:
            return {"n_neighbors": 0}
        goals = np.array([n["usd_goal_real"] for n in pool], dtype=float)
        funded = np.array([n["target"] for n in pool])
        succ = goals[funded == 1]
        fail = goals[funded == 0]

        def stats(a):
            return {"n": int(a.size),
                    "median_goal_usd": round(float(np.median(a)), 2) if a.size else None,
                    "mean_goal_usd": round(float(np.mean(a)), 2) if a.size else None}

        succ_med = float(np.median(succ)) if succ.size else None
        return {
            "n_neighbors": len(pool),
            "success_rate": round(float(funded.mean()), 4),      # retrieval-based prior (k-NN P(fund))
            "all": stats(goals),
            "successful": stats(succ),
            "failed": stats(fail),
            "query_goal_usd": round(qa["usd_goal_real"], 2),
            "query_goal_vs_successful_median": (
                None if succ_med is None else
                ("above" if qa["usd_goal_real"] > succ_med else "at or below")),
        }

    def _category_benchmark(self, qa: dict[str, Any]) -> dict[str, Any]:
        """Whole-KB success rate + goal medians for the query's main_category."""
        main = qa["main_category"]
        stats = self._category_table()
        if main not in stats.index:
            return {"main_category": main, "available": False,
                    "reason": "category not present in KB"}
        r = stats.loc[main]
        succ_med = r["median_goal_successful"]
        return {
            "main_category": main,
            "n_in_category": int(r["n"]),
            "success_rate": round(float(r["success_rate"]), 4),
            "median_goal_successful_usd": None if pd.isna(succ_med) else round(float(succ_med), 2),
            "median_goal_failed_usd": None if pd.isna(r["median_goal_failed"]) else round(float(r["median_goal_failed"]), 2),
            "query_goal_usd": round(qa["usd_goal_real"], 2),
            "query_goal_vs_successful_median": (
                None if pd.isna(succ_med) else
                ("above" if qa["usd_goal_real"] > succ_med else "at or below")),
        }

    def _category_table(self) -> pd.DataFrame:
        """Cache per-main_category aggregates over the whole KB metadata."""
        if self._category_stats is None:
            md = self.engine.metadata_
            g = md.groupby("main_category")
            self._category_stats = pd.DataFrame({
                "n": g.size(),
                "success_rate": g["target"].mean(),
                "median_goal_successful": md[md["target"] == 1].groupby("main_category")["usd_goal_real"].median(),
                "median_goal_failed": md[md["target"] == 0].groupby("main_category")["usd_goal_real"].median(),
            })
        return self._category_stats


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)
