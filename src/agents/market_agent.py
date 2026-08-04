"""Phase B — Market Analyst Agent (Day-12, Layer 12).

The market specialist. It scores **category attractiveness & market timing** — the honest "market"
for a Kickstarter campaign is its **category cohort**, not an addressable market. It reads Day-9
``category_analysis`` (funding rate, lift, crowding, year-trend, timing) and emits a uniform
``AgentReport``.

Score (0-100) = 50 + lift + timing + trend + crowding adjustments. Confidence rises with cohort size
and year-trend consistency (this agent's OWN signal strength, not the global Day-11 confidence).

**Deferred:** market size / TAM / CAGR / industry growth rate — none of that exists in the dataset;
we observe the Kickstarter category, not an addressable market. Deterministic, no faiss.

Example
-------
>>> ma = MarketAgent()
>>> r = ma.analyze(raw_campaign)
>>> r["agent"], r["stance"], 0 <= r["score"] <= 100
('MarketAgent', 'positive', True)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.agents.base import agent_report, clip, clip01
from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.intelligence_aggregator import IntelligenceAggregator
from src.utils.config import load_config
from src.utils.logger import get_logger

# Structural scoring maps (points added to the neutral 50 baseline).
_TIMING = {"Favorable": 10, "Neutral": 0, "Unfavorable": -10}
_TREND = {"rising": 8, "flat": 0, "falling": -8}
_CROWD = {"Crowded": -5, "Moderate": 0, "Sparse": 5}
_DEFERRED = {
    b: {"status": "deferred", "reason": "no addressable-market data in a Kickstarter category"}
    for b in ["market_size_tam", "industry_cagr", "addressable_market", "industry_growth_rate"]
}
_CAPTION = ("Market = the Kickstarter category cohort (funding rate, crowding, trend, timing), not an "
            "addressable market. Model-simulated, not causal; TAM / market size / CAGR are deferred.")


class MarketAgent:
    """Score category attractiveness & market timing from Day-9 category analysis.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    aggregator:
        Optional pre-built Day-11 :class:`IntelligenceAggregator` (share the warm stack).
    cf_pipeline:
        Optional pre-built Day-8 pipeline (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 aggregator: Optional[IntelligenceAggregator] = None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.agents.market")

        c = self.config.get("day12", {}).get("market", {})
        self.positive_at = float(c.get("positive_at", 60))
        self.negative_at = float(c.get("negative_at", 40))
        self.n_full = float(c.get("n_full_confidence", 5000))
        self.out_path = Path(c.get("out_path", "artifacts/agents/market_analysis.json"))

        self.agg = aggregator or IntelligenceAggregator(self.config, self.logger, cf_pipeline=cf_pipeline)

    # ------------------------------------------------------------------ public
    def analyze(self, raw: dict[str, Any],
                intelligence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Market AgentReport for one campaign (reuses an aggregated profile if given)."""
        vi = intelligence if intelligence is not None else self.agg.aggregate(raw)
        ca = vi["copilot"]["category_analysis"]

        score = self._score(ca)
        confidence = self._confidence(ca)
        rate = float(ca.get("category_funding_rate", 0.0))
        lift = float(ca.get("category_lift", 1.0))
        base = rate / lift if lift else rate
        cat = ca.get("category") or vi["campaign"].get("main_category")
        direction = (ca.get("year_trend") or {}).get("direction", "flat")
        n = int(ca.get("n_campaigns", 0))

        findings = [
            f"{cat} campaigns fund at {rate:.0%} vs a {base:.0%} platform baseline (lift {lift:.2f}).",
            f"Market timing is {str(ca.get('market_timing','?')).lower()}; the {cat} trend is "
            f"{direction} ({n:,} campaigns).",
            f"The category is {str(ca.get('crowding','?')).lower()}.",
        ]
        report = agent_report(
            "MarketAgent", "Category attractiveness & market timing", score, confidence,
            findings, {"name": "category_funding_rate", "value": round(rate, 4)},
            _DEFERRED, _CAPTION, self.positive_at, self.negative_at)
        self.logger.info("MarketAgent: %s %d (conf %.2f) — %s.", report["stance"], report["score"],
                         report["confidence"], cat)
        return report

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Market analyses for a spread of test campaigns -> deliverable JSON."""
        df = pd.read_csv(self.config.get("day6", {}).get("knowledge_base", {})
                         .get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(self.config.get("day6", {}).get("knowledge_base", {})
                          .get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.analyze(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved market analyses -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _score(self, ca: dict[str, Any]) -> float:
        lift = float(ca.get("category_lift", 1.0))
        lift_pts = clip((lift - 1.0) * 100, -25, 25)         # lift 1.25 -> +25 (capped)
        timing = _TIMING.get(ca.get("market_timing"), 0)
        trend = _TREND.get((ca.get("year_trend") or {}).get("direction"), 0)
        crowd = _CROWD.get(ca.get("crowding"), 0)
        return clip(50 + lift_pts + timing + trend + crowd, 0, 100)

    def _confidence(self, ca: dict[str, Any]) -> float:
        n = int(ca.get("n_campaigns", 0))
        n_comp = clip01(n / self.n_full)                     # big cohort -> reliable
        rates = [float(y["funding_rate"]) for y in (ca.get("year_trend") or {}).get("by_year", [])]
        consistency = clip01(1 - float(np.std(rates)) / 0.10) if len(rates) >= 3 else 0.5
        return min(0.99, max(0.5, 0.5 + 0.49 * (0.6 * n_comp + 0.4 * consistency)))
