"""Phase B — Intelligence Aggregation Engine (Day-11, Layer 11).

Collect the outputs of every prior intelligence layer into ONE structured ``VentureIntelligence``
profile — the raw material the recommendation generator, prioritizer, confidence scorer, and
audience strategies all read from. Nothing is recomputed downstream; this is the single collection
point.

    prediction (D4) + drivers (D5) + counterfactual (D8) + copilot brief (D9)
    + evolution plan (D10) + scenario table (D10) + same-category cohort  ->  VentureIntelligence

**One warm faiss-free stack.** A single :class:`CounterfactualPipeline` (model + item frame) is
shared into the Day-9 copilot (via its ``cf_pipeline=`` hook), the Day-10 evolution and scenario
pipelines, and this aggregator's own ``cf.analyze`` / SHAP explainer. "Similar startups" is the
**same-category cohort in the shared pandas item frame** — not Day-6 semantic retrieval — so there
is no faiss, no torch, no OpenMP concern.

Deferred (never fabricated): retention / revenue / moat / team / pricing / growth-trajectory — the
aggregator preserves the honesty boundary of the layers beneath it.

Example
-------
>>> agg = IntelligenceAggregator()
>>> vi = agg.aggregate(raw_campaign)
>>> vi["prediction"]["funding_probability"], vi["evolution"]["stage"]["stage"]
(0.24, 'Fixable')
"""

from __future__ import annotations

import copy
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.copilot.pipeline import VentureCopilotPipeline
from src.counterfactual.pipeline import CounterfactualPipeline
from src.evolution.pipeline import EvolutionPipeline
from src.scenario.pipeline import ScenarioPipeline
from src.utils.config import load_config
from src.utils.logger import get_logger

# The honesty boundary carried at the aggregate level (merged with each layer's own deferrals).
_DEFERRED = {
    b: {"status": "deferred", "reason": "not observable in a single pre-launch Kickstarter snapshot"}
    for b in ["customer_retention", "monthly_revenue", "revenue_growth", "unit_economics",
              "competitive_moat", "named_competitors", "founder_experience", "team_size",
              "pricing_strategy", "market_size_tam", "product_traction", "funding_rounds",
              "growth_trajectory"]
}
_CAPTION = ("A unified venture-intelligence profile aggregated from the real Day 4-10 layers on one "
            "warm stack. Similar startups = same-category cohort in our own data (no semantic "
            "retrieval). Model-simulated over controllable features, not causal; product / team / "
            "traction / financial signals are deferred (the dataset can't observe them).")


class IntelligenceAggregator:
    """Merge every prior-layer output into one ``VentureIntelligence`` object.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    cf_pipeline:
        Optional pre-built Day-8 :class:`CounterfactualPipeline` (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reco.aggregator")

        c = self.config.get("day11", {}).get("aggregator", {})
        self.top_drivers = int(c.get("top_drivers", 4))
        self.out_path = Path(c.get("out_path", "artifacts/recommendation/venture_intelligence.json"))

        # ONE warm stack, shared into every prior-layer pipeline.
        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)
        self.copilot = VentureCopilotPipeline(self.config, self.logger, cf_pipeline=self.cf)  # IMP-1
        self.evo = EvolutionPipeline(self.config, self.logger, cf_pipeline=self.cf)
        self.scn = ScenarioPipeline(self.config, self.logger, cf_pipeline=self.cf)
        self.lx = self.copilot.lx                          # reuse the copilot's warm SHAP explainer
        self.frame = self.cf.evidence.frame                # item frame = the cohort source (no faiss)
        self.base = float(self.cf.evidence.base)
        # Profiles are deterministic per campaign (Phase-I content hash), and one request — or the
        # several endpoints the dashboard calls for one campaign — asks for the same profile repeatedly.
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_size = int(c.get("cache_size", 64))
        self._lock = threading.Lock()
        self.logger.info("IntelligenceAggregator ready (one warm faiss-free stack).")

    # ------------------------------------------------------------------ public
    def aggregate(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Build the unified VentureIntelligence profile for one campaign (memoized per campaign).

        Each caller gets its own deep copy, so a caller that edits the profile can't affect another.
        """
        key = json.dumps(raw, sort_keys=True, default=str)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return copy.deepcopy(self._cache[key])
        vi = self._aggregate(raw)
        with self._lock:
            self._cache[key] = vi
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return copy.deepcopy(vi)

    def _aggregate(self, raw: dict[str, Any]) -> dict[str, Any]:
        raw = dict(raw)
        analysis = self.cf.analyze(raw)                    # Day-8 (dict evidence — for generator/prioritizer)
        brief = self.copilot.analyze(raw)                  # Day-9
        evo = self.evo.run(raw)                            # Day-10 evolution
        scn = self.scn.run(raw)                            # Day-10 scenario
        threshold = float(self.cf.simulator.engine.threshold)
        base_p = analysis["baseline_p"]
        # Drop the engine's cumulative telemetry counter — it's not campaign intelligence and would
        # make the profile non-deterministic across calls (breaking the Phase-I content hash).
        opt_plan = {k: v for k, v in analysis["optimization_plan"].items() if k != "n_model_calls"}

        vi = {
            "campaign": {k: raw.get(k) for k in ("name", "main_category", "usd_goal_real", "country")},
            "prediction": {"funding_probability": base_p,
                           "predicted_label": brief["predicted_label"],
                           "threshold": round(threshold, 4)},
            "drivers": self._drivers(raw),                                        # Day 5
            "counterfactual": {"ranked_recommendations": analysis["ranked_recommendations"],
                               "optimization_plan": opt_plan},                    # Day 8
            "copilot": {k: brief[k] for k in ("investor_objections", "risk_analysis", "swot",
                                              "category_analysis", "competition_analysis",
                                              "recommendations")},                # Day 9
            "evolution": {k: evo[k] for k in ("stage", "roadmap", "milestones", "timeline")},  # Day 10
            "scenario": {k: scn[k] for k in ("ranked_scenarios", "marginal_values",
                                             "best_scenario", "most_efficient_scenario",
                                             "recommendation")},                  # Day 10
            "cohort": self._cohort(raw),                                          # "similar startups" (no faiss)
            "deferred": {**_DEFERRED, **evo.get("deferred", {})},
            "honesty_note": _CAPTION,
        }
        self.logger.info("Aggregated intelligence: %s | %.0f%% | stage %s | best scenario %.0f%%.",
                         vi["campaign"]["name"], base_p * 100, evo["stage"]["stage"],
                         scn["best_scenario"]["probability"] * 100)
        return vi

    def generate_examples(self, n: int = 2, random_state: int = 4) -> Path:
        """Aggregated profiles for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.aggregate(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved venture intelligence -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _drivers(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Top +/- real-feature drivers of the prediction (Day-5 SHAP)."""
        ex = self.lx.explain_raw(pd.DataFrame([raw]))[0].to_dict()
        pick = lambda lst: [{"feature": c["feature"], "contribution": round(c["contribution"], 4),
                             "descriptor": c["descriptor"]} for c in lst[:self.top_drivers]]
        return {"top_positive": pick(ex["top_positive"]), "top_negative": pick(ex["top_negative"])}

    def _cohort(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Similar startups = same-category cohort in the item frame (no semantic retrieval)."""
        main = str(raw.get("main_category"))
        sub = self.frame[self.frame["main_category"] == main]
        n = int(len(sub))
        rate = float(sub["target"].mean()) if n else self.base
        return {"main_category": main, "n_similar": n,
                "cohort_funding_rate": round(rate, 4), "platform_base_rate": round(self.base, 4),
                "note": "Similar = same-category campaigns in our own data (frame cohort, no faiss)."}
