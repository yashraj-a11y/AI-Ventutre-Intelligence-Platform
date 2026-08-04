"""Phase H — Unified Recommendation Pipeline (Day-11, Layer 11).

The single entry point that assembles the complete **VentureDecision** — the platform's final,
audience-split, evidence-backed decision object. It runs the whole Layer-11 chain on ONE warm
faiss-free stack, aggregating the intelligence **once** and threading that one profile (and the one
confidence score) through every sub-engine so nothing recomputes:

    aggregate (B) -> generate (C) -> prioritize (D) -> confidence (G)
                  -> founder view (E, +confidence) -> investor view (F, +confidence)
                  -> final strategy synthesis

Because confidence is computed once and fed into both audience views, a low-confidence campaign can
downgrade its investor band — the decision object is *self-aware* of its own reliability. Founder
and investor strategies are two projections of the same prioritized intelligence, so they cannot
contradict each other. Deterministic, no LLM.

Example
-------
>>> pipe = RecommendationPipeline()
>>> d = pipe.run(raw_campaign)
>>> d["investor_strategy"]["recommendation"], round(d["confidence"]["score"], 2)
('Proceed with Caution', 0.83)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.confidence import ConfidenceScorer
from src.recommendation.founder_strategy import FounderStrategyEngine
from src.recommendation.intelligence_aggregator import IntelligenceAggregator
from src.recommendation.investor_strategy import InvestorStrategyEngine
from src.recommendation.prioritization import RecommendationPrioritizer
from src.recommendation.recommendation_generator import RecommendationGenerator
from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("A unified venture decision aggregated from Days 4-10 on one warm stack: readiness + "
            "prioritized evidence-backed actions + transparent confidence + founder & investor "
            "views (two projections of one intelligence). Model-simulated over controllable "
            "features, not causal; product / team / traction / financial items are deferred.")


class RecommendationPipeline:
    """Assemble the complete VentureDecision on one warm stack, one aggregation, one confidence.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    cf_pipeline:
        Optional pre-built Day-8 :class:`CounterfactualPipeline` (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reco.pipeline")

        c = self.config.get("day11", {}).get("pipeline", {})
        self.out_path = Path(c.get("out_path", "artifacts/recommendation/unified_recommendations.json"))

        # ONE aggregator (=> one warm cf), shared into every sub-engine.
        self.agg = IntelligenceAggregator(self.config, self.logger, cf_pipeline=cf_pipeline)
        self.generator = RecommendationGenerator(self.config, self.logger, aggregator=self.agg)
        self.prioritizer = RecommendationPrioritizer(self.config, self.logger, generator=self.generator)
        self.founder = FounderStrategyEngine(self.config, self.logger, prioritizer=self.prioritizer)
        self.investor = InvestorStrategyEngine(self.config, self.logger, aggregator=self.agg)
        self.confidence = ConfidenceScorer(self.config, self.logger, aggregator=self.agg)
        self.logger.info("RecommendationPipeline ready (one warm faiss-free stack).")

    # ------------------------------------------------------------------ public
    def run(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Build the complete VentureDecision for one campaign."""
        vi = self.agg.aggregate(raw)                        # ONE aggregation, threaded everywhere
        recs = self.generator.generate(raw, intelligence=vi)
        prioritized = self.prioritizer.prioritize(raw, intelligence=vi)
        conf = self.confidence.score(raw, intelligence=vi)  # ONE confidence, fed into both views
        founder = self.founder.generate(raw, intelligence=vi, prioritized=prioritized, confidence=conf)
        investor = self.investor.generate(raw, intelligence=vi, confidence=conf)

        stage = vi["evolution"]["stage"]
        p, opt, t = (stage["funding_probability"], stage["optimized_probability"], stage["threshold"])
        decision = {
            "campaign": vi["campaign"],
            "investment_readiness": {
                "stage": stage["stage"], "funding_probability": p,
                "optimized_probability": opt, "threshold": t,
                "readiness_index": round(100 * 0.5 * (p + opt)),   # model-derived: mean(current, reachable)
            },
            "improvement_headroom": round(opt - p, 4),
            "confidence": {"score": conf["score"], "factors": conf["factors"], "reason": conf["reason"]},
            "priority_improvements": prioritized["prioritized_actions"],
            "recommendations": recs["recommendations"],
            "category_positioning": recs["category_positioning"],
            "risks": {"overall_risk_level": vi["copilot"]["risk_analysis"].get("overall_risk_level"),
                      "top_risks": vi["copilot"]["risk_analysis"].get("risks", [])[:3]},
            "opportunities": self._opportunities(vi, p, opt),
            "founder_strategy": founder,
            "investor_strategy": investor,
            "final_strategy": self._final_strategy(investor, founder, stage, conf),
            "deferred": {**vi["deferred"], **recs["deferred_categories"],
                         **founder["deferred_founder_advice"], **investor["deferred_diligence"]},
            "honesty_note": _CAPTION,
        }
        self.logger.info("VentureDecision: %s | %s | readiness %d | confidence %.2f.",
                         vi["campaign"]["name"], investor["recommendation"],
                         decision["investment_readiness"]["readiness_index"], conf["score"])
        return decision

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Full venture decisions for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.run(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved unified recommendations -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    @staticmethod
    def _opportunities(vi: dict[str, Any], p: float, opt: float) -> list[dict[str, Any]]:
        opps = [{"type": "reachable_upside",
                 "detail": f"Priority changes can lift funding odds from {p:.0%} to ~{opt:.0%}."}]
        cohort = vi["cohort"]
        if cohort["cohort_funding_rate"] >= cohort["platform_base_rate"]:
            opps.append({"type": "category",
                         "detail": (f"{cohort['main_category']} funds at "
                                    f"{cohort['cohort_funding_rate']:.0%} vs the "
                                    f"{cohort['platform_base_rate']:.0%} baseline — a favorable category.")})
        for d in vi["drivers"]["top_positive"][:2]:
            opps.append({"type": "strength", "detail": f"Working in your favor: {d['descriptor']}."})
        return opps

    @staticmethod
    def _final_strategy(investor: dict[str, Any], founder: dict[str, Any],
                        stage: dict[str, Any], conf: dict[str, Any]) -> str:
        p, opt = stage["funding_probability"], stage["optimized_probability"]
        focus = "; ".join(founder["focus_first"]).lower()
        base = (f"{investor['recommendation']}: {stage['stage']} at {p:.0%} "
                f"(reachable ~{opt:.0%}), confidence {conf['score']:.0%}.")
        if focus:
            base += f" Founder priority: {focus}."
        return base
