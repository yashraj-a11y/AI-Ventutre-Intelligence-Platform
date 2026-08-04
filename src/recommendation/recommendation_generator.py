"""Phase C — Recommendation Generator (Day-11, Layer 11).

Turn the aggregated :class:`VentureIntelligence` into **strategic recommendations** in the REAL
categories the platform can actually support:

- **Fundraising Strategy** — the funding goal.
- **Campaign Strategy** — duration / launch timing / title.
- **Category Positioning** — an *observational* read of the category cohort + Day-9 competition
  (context, not an action).

Each actionable recommendation reuses the Day-8 counterfactual as its source of truth (no
re-derivation) and carries: action, category, priority label, expected impact, a strategic reason,
and **Day-7 historical evidence**. One recommendation per controllable feature (best-ranked).

Product / Team / Financial / Technology / Go-To-Market strategies are **deferred** — they need data
(retention, MRR, team, moat, pricing) the dataset doesn't have. Deterministic, no LLM. No faiss.

Example
-------
>>> gen = RecommendationGenerator()
>>> recs = gen.generate(raw_campaign)["recommendations"]
>>> recs[0]["category"], recs[0]["priority_label"]
('Campaign Strategy', 'Critical')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.explanation_generator import action_phrase
from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.intelligence_aggregator import IntelligenceAggregator
from src.utils.config import load_config
from src.utils.logger import get_logger

# Controllable feature -> real strategic category.
_CATEGORY = {
    "goal": "Fundraising Strategy",
    "campaign_duration": "Campaign Strategy",
    "launch_weekday": "Campaign Strategy",
    "title": "Campaign Strategy",
}
# Deterministic strategic "why" per lever (the reason, distinct from the historical evidence).
_REASON = {
    "goal": ("The funding goal is the single strongest lever on fundability — right-sizing it to a "
             "category-proven level most improves the odds."),
    "campaign_duration": ("Campaign length materially shifts funding odds; a tighter window "
                          "concentrates backer momentum."),
    "launch_weekday": ("Launch timing is a near-zero-cost lever — the funded cohort clusters on "
                       "specific weekdays."),
    "title": ("Title framing is a cheap signal the model associates with funded campaigns."),
}
# Strategy categories we refuse to fabricate (no data).
_DEFERRED_CATEGORIES = {
    c: {"status": "deferred", "reason": "requires data the dataset doesn't have"}
    for c in ["Product Strategy", "Team Strategy", "Financial Strategy",
              "Technology Strategy", "Go-To-Market Strategy"]
}
_CAPTION = ("Strategic recommendations over controllable features only (goal / duration / timing / "
            "title), each backed by Day-7 historical evidence. Category positioning is observational "
            "context, not an action. Product / team / financial / tech / GTM strategies are deferred "
            "(no data). Model-simulated, not causal.")


class RecommendationGenerator:
    """Reframe the aggregated intelligence as categorized strategic recommendations.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    cf_pipeline:
        Optional pre-built Day-8 pipeline (share the warm stack).
    aggregator:
        Optional pre-built Phase-B :class:`IntelligenceAggregator` (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None,
                 aggregator: Optional[IntelligenceAggregator] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reco.generator")

        c = self.config.get("day11", {}).get("generator", {})
        self.critical_pts = int(c.get("critical_impact_points", 15))
        self.high_pts = int(c.get("high_impact_points", 5))
        self.out_path = Path(c.get("out_path", "artifacts/recommendation/strategic_recommendations.json"))

        self.agg = aggregator or IntelligenceAggregator(self.config, self.logger, cf_pipeline=cf_pipeline)

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any],
                 intelligence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Strategic recommendations for one campaign (reuses an aggregated profile if given)."""
        vi = intelligence if intelligence is not None else self.agg.aggregate(raw)
        recs_src = vi["counterfactual"]["ranked_recommendations"]
        # one recommendation per controllable feature — best-ranked (source is priority-sorted).
        seen: set[str] = set()
        best = [r for r in recs_src if not (r["feature"] in seen or seen.add(r["feature"]))]
        recommendations = [self._reframe(r) for r in best]

        self.logger.info("Generated %d strategic recommendations (%d deferred categories).",
                         len(recommendations), len(_DEFERRED_CATEGORIES))
        return {
            "campaign": vi["campaign"],
            "recommendations": recommendations,
            "category_positioning": self._positioning(vi),
            "deferred_categories": _DEFERRED_CATEGORIES,
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Recommendations for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.generate(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved strategic recommendations -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _reframe(self, rec: dict[str, Any]) -> dict[str, Any]:
        feature = rec["feature"]
        pts = round(float(rec.get("delta", 0)) * 100)
        return {
            "feature": feature,
            "category": _CATEGORY.get(feature, "Campaign Strategy"),
            "action": action_phrase(feature, rec.get("current"), rec.get("proposed")),
            "priority_label": self._priority(pts),
            "expected_impact_points": pts,
            "funding_influence": round(float(rec.get("delta", 0)), 4),
            "reason": _REASON.get(feature, "A controllable pre-launch lever the model favors."),
            "evidence": (rec.get("evidence") or {}).get("statement"),
            "confidence": rec.get("confidence"),
            "source_priority": rec.get("priority"),
        }

    def _priority(self, pts: int) -> str:
        if pts >= self.critical_pts:
            return "Critical"
        if pts >= self.high_pts:
            return "High"
        return "Medium"

    @staticmethod
    def _positioning(vi: dict[str, Any]) -> dict[str, Any]:
        """Observational category positioning (not an action) from cohort + Day-9 analysis."""
        cat = vi["copilot"]["category_analysis"]
        comp = vi["copilot"]["competition_analysis"]
        cohort = vi["cohort"]
        rate, base = cohort["cohort_funding_rate"], cohort["platform_base_rate"]
        stance = "favorable" if rate >= base else "below-average"
        return {
            "category": cohort["main_category"],
            "cohort_funding_rate": rate,
            "platform_base_rate": base,
            "market_opportunity": cat.get("market_opportunity"),
            "crowding": cat.get("crowding"),
            "competitive_intensity": comp.get("competitive_intensity"),
            "insight": (f"{cohort['main_category']} campaigns fund at {rate:.0%} vs the {base:.0%} "
                        f"platform baseline — a {stance} category to launch in "
                        f"(n={cohort['n_similar']:,})."),
            "note": "Observational positioning, not an action — from the category cohort + Day-9 analysis.",
        }
