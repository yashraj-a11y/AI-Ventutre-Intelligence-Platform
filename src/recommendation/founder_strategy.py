"""Phase E — Founder Strategy Engine (Day-11, Layer 11).

The **founder-facing view** of the prioritized intelligence: a pre-launch action plan tied to the
funding-readiness stage — "here's where you stand, here's what to do first, in order." It's one
projection of the shared intelligence object (the investor view, Phase F, is the other), so the two
can never contradict each other.

    Fixable before launch (24% -> ~66%). Do these before you go live:
      1. Shorten your campaign from 29 to 15 days   (Critical, +30 pts)
      2. Lower your funding goal to $5,000          (High, +12 pts — investors flag this)

Deterministic templates, no LLM. Real levers only (goal / duration / timing / title). Founder
advice that needs data we don't have — "improve retention", "validate pricing", "hire a technical
co-founder", "delay fundraising three months", "grow MRR" — is **deferred**, not emitted.

Example
-------
>>> fs = FounderStrategyEngine()
>>> plan = fs.generate(raw_campaign)
>>> plan["readiness"]["stage"], plan["action_plan"][0]["step"]
('Fixable', 1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.prioritization import RecommendationPrioritizer
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED_FOUNDER = {
    b: {"status": "deferred", "reason": "needs data the dataset doesn't have"}
    for b in ["improve_retention_before_fundraising", "validate_pricing_strategy",
              "hire_technical_cofounder", "delay_fundraising_timing", "grow_monthly_revenue"]
}
_CAPTION = ("Founder-facing pre-launch action plan over controllable levers, ordered by the Day-11 "
            "cross-layer priority and tied to the funding-readiness stage. Deterministic, "
            "model-simulated (not causal). Product / team / pricing / retention advice is deferred.")


class FounderStrategyEngine:
    """Project the prioritized intelligence into a founder pre-launch action plan.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    cf_pipeline:
        Optional pre-built Day-8 pipeline (share the warm stack).
    prioritizer:
        Optional pre-built Phase-D :class:`RecommendationPrioritizer` (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None,
                 prioritizer: Optional[RecommendationPrioritizer] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reco.founder")

        c = self.config.get("day11", {}).get("founder", {})
        self.focus_first_n = int(c.get("focus_first_n", 2))
        self.out_path = Path(c.get("out_path", "artifacts/recommendation/founder_strategy.json"))

        self.prioritizer = prioritizer or RecommendationPrioritizer(
            self.config, self.logger, cf_pipeline=cf_pipeline)
        self.agg = self.prioritizer.agg                     # share the warm aggregator

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any], intelligence: Optional[dict[str, Any]] = None,
                 prioritized: Optional[dict[str, Any]] = None,
                 confidence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Founder pre-launch action plan for one campaign.

        ``confidence`` (Phase G) is optional — threaded in by the Phase-H pipeline when available.
        """
        vi = intelligence if intelligence is not None else self.agg.aggregate(raw)
        pr = prioritized if prioritized is not None else self.prioritizer.prioritize(raw, intelligence=vi)
        stage = vi["evolution"]["stage"]

        action_plan = [{
            "step": a["rank"],
            "action": a["action"],
            "category": a["category"],
            "priority_label": a["priority_label"],
            "expected_impact_points": a["expected_impact_points"],
            "why": a["reason"],
            "evidence": a["evidence"],
            "addresses_investor_concern": a["factors"]["investor_overlap"] > 0,
        } for a in pr["prioritized_actions"]]

        out = {
            "campaign": vi["campaign"],
            "readiness": {"stage": stage["stage"],
                          "funding_probability": stage["funding_probability"],
                          "optimized_probability": stage["optimized_probability"],
                          "threshold": stage["threshold"]},
            "headline": self._headline(stage),
            "focus_first": [a["action"] for a in action_plan[:self.focus_first_n]],
            "action_plan": action_plan,
            "deferred_founder_advice": _DEFERRED_FOUNDER,
            "honesty_note": _CAPTION,
        }
        if confidence is not None:
            out["confidence"] = confidence
        self.logger.info("Founder strategy: %s, %d actions (focus: %s).", stage["stage"],
                         len(action_plan), ", ".join(out["focus_first"]) or "-")
        return out

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Founder strategies for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.generate(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved founder strategies -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    @staticmethod
    def _headline(stage: dict[str, Any]) -> str:
        s = stage["stage"]
        p, opt, t = (stage["funding_probability"], stage["optimized_probability"], stage["threshold"])
        if s == "Strong":
            return (f"Strong pre-launch position ({p:.0%}) — comfortably above the {t:.0%} funding "
                    f"bar. The steps below are optional polish.")
        if s == "Ready":
            return (f"You're already fundable ({p:.0%}, above the {t:.0%} bar). These low-cost "
                    f"changes strengthen the campaign further.")
        if s == "Fixable":
            return (f"Fixable before launch. As-is the model estimates {p:.0%}, but the actions "
                    f"below can reach ~{opt:.0%} — above the {t:.0%} bar. Do them before you go live.")
        return (f"The fundamentals need rework. Even optimized the model estimates ~{opt:.0%}, "
                f"below the {t:.0%} bar — reconsider the goal, scope, or category before launching.")
