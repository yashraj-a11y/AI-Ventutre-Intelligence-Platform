"""Phase D — Recommendation Prioritization Engine (Day-11, Layer 11).

Rank the strategic recommendations by a **transparent weighted score** that fuses signals from
*multiple* layers — the genuine value over Day-8's single-layer ranking. Six factors, each in
``[0, 1]``, combined as a normalized weighted mean:

    impact           (Day-8 predicted funding gain)
    evidence         (Day-7 historical lift x sample size)
    investor_overlap (Day-9: does this lever answer a raised objection?)
    scenario_gain    (Day-10 marginal value — the joint re-simulation cross-check)
    feasibility      (Day-8)
    confidence       (Day-8 label)

The per-factor breakdown and the weights are returned with every action, so the ranking is fully
auditable — nothing is a black box. Deterministic, no LLM, no faiss.

Example
-------
>>> pr = RecommendationPrioritizer()
>>> acts = pr.prioritize(raw_campaign)["prioritized_actions"]
>>> acts[0]["rank"], acts[0]["priority_score"], list(acts[0]["factors"])
(1, 0.78, ['impact', 'evidence', 'investor_overlap', 'scenario_gain', 'feasibility', 'confidence'])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.intelligence_aggregator import IntelligenceAggregator
from src.recommendation.recommendation_generator import RecommendationGenerator
from src.utils.config import load_config
from src.utils.logger import get_logger

# scenario "change" key -> controllable feature.
_FEAT_OF_CHANGE = {"goal_usd": "goal", "duration_days": "campaign_duration",
                   "weekday": "launch_weekday", "add_number": "title"}
_SEV = {"High": 1.0, "Medium": 0.6, "Low": 0.3}
_CONF = {"high": 1.0, "medium": 0.6, "low": 0.3}
_CAPTION = ("Ranked by a transparent weighted mean of six cross-layer factors (each shown). This "
            "ADDS investor-concern overlap and scenario-gain to Day-8's ranking — it doesn't "
            "override the model. Model-simulated, not causal.")


class RecommendationPrioritizer:
    """Rank recommendations by a transparent, auditable cross-layer score.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    cf_pipeline:
        Optional pre-built Day-8 pipeline (share the warm stack).
    generator:
        Optional pre-built Phase-C :class:`RecommendationGenerator` (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None,
                 generator: Optional[RecommendationGenerator] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reco.prioritization")

        c = self.config.get("day11", {}).get("prioritization", {})
        self.weights = dict(c.get("weights", {}))
        self.impact_scale = float(c.get("impact_scale_points", 25))
        self.lift_span = float(c.get("evidence_lift_span", 0.6))
        self.min_n = int(c.get("min_evidence_n", 100))
        self.scenario_scale = float(c.get("scenario_gain_scale", 0.25))
        self.out_path = Path(c.get("out_path", "artifacts/recommendation/prioritized_actions.json"))

        self.generator = generator or RecommendationGenerator(
            self.config, self.logger, cf_pipeline=cf_pipeline)
        self.agg = self.generator.agg                       # share the warm aggregator

    # ------------------------------------------------------------------ public
    def prioritize(self, raw: dict[str, Any],
                   intelligence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Rank the recommendations for one campaign by the cross-layer score."""
        vi = intelligence if intelligence is not None else self.agg.aggregate(raw)
        recs = self.generator.generate(raw, intelligence=vi)["recommendations"]

        day8 = self._day8_by_feature(vi)
        marg = {_FEAT_OF_CHANGE[m["change"]]: float(m["marginal_gain"])
                for m in vi["scenario"]["marginal_values"] if m["change"] in _FEAT_OF_CHANGE}
        obj = self._objection_overlap(vi)

        enriched = []
        for rec in recs:
            f = rec["feature"]
            d8 = day8.get(f, {})
            factors = {
                "impact": _clip(rec["expected_impact_points"] / self.impact_scale),
                "evidence": self._evidence_strength(d8.get("evidence", {})),
                "investor_overlap": obj.get(f, 0.0),
                "scenario_gain": _clip(marg.get(f, 0.0) / self.scenario_scale),
                "feasibility": float(d8.get("feasibility", 0.0)),
                "confidence": _CONF.get(str(rec.get("confidence")), 0.6),
            }
            score = self._weighted(factors)
            enriched.append({**rec, "priority_score": round(score, 4),
                             "factors": {k: round(v, 3) for k, v in factors.items()}})

        enriched.sort(key=lambda r: r["priority_score"], reverse=True)
        for i, r in enumerate(enriched, start=1):
            r["rank"] = i

        self.logger.info("Prioritized %d actions (top: %s, score %.2f).", len(enriched),
                         enriched[0]["feature"] if enriched else "-",
                         enriched[0]["priority_score"] if enriched else 0.0)
        return {
            "campaign": vi["campaign"],
            "weights": self.weights,
            "prioritized_actions": enriched,
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Prioritized actions for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.prioritize(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved prioritized actions -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _weighted(self, factors: dict[str, float]) -> float:
        w = self.weights
        tot = sum(w.get(k, 0.0) for k in factors)
        return sum(w.get(k, 0.0) * v for k, v in factors.items()) / tot if tot else 0.0

    def _evidence_strength(self, ev: dict[str, Any]) -> float:
        lift = float(ev.get("lift", 1.0)); n = int(ev.get("n", 0))
        s = _clip((lift - 1.0) / self.lift_span)
        return s * 0.5 if n < self.min_n else s

    @staticmethod
    def _day8_by_feature(vi: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in vi["counterfactual"]["ranked_recommendations"]:
            out.setdefault(r["feature"], r)                 # best-ranked (source is priority-sorted)
        return out

    @staticmethod
    def _objection_overlap(vi: dict[str, Any]) -> dict[str, float]:
        """feature -> strongest severity weight of an investor objection about that lever."""
        out: dict[str, float] = {}
        for o in vi["copilot"]["investor_objections"]:
            f = _feat_of_dimension(str(o.get("dimension", "")))
            if f:
                out[f] = max(out.get(f, 0.0), _SEV.get(str(o.get("severity", "Low")), 0.3))
        return out


def _clip(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _feat_of_dimension(dim: str) -> Optional[str]:
    d = dim.lower()
    if "goal" in d:
        return "goal"
    if "duration" in d or "length" in d:
        return "campaign_duration"
    if "weekday" in d or "timing" in d or "launch" in d:
        return "launch_weekday"
    if "title" in d:
        return "title"
    return None
