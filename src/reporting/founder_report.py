"""Phase B — Founder Intelligence Report Generator (Day-13, Layer 13).

Assemble the founder-facing report — **presentation only, no new analysis**. Every section is a
reformatting of a real upstream output, tagged with its ``source``:

    Executive Summary      <- Day-11 synthesis (final strategy)
    Scorecard              <- Days 4/10/11 (readiness index, probability, stage, confidence)
    Strengths / Weaknesses <- Day-5 SHAP positive / negative drivers
    Priority Improvements  <- Day-11 prioritization (with Day-7 evidence)
    Evolution Roadmap      <- Day-10 preparation roadmap
    Scenario Analysis      <- Day-10 scenario simulator
    Next Milestones        <- Day-10 milestones
    Deferred appendix      <- what the platform can't assess (no data)

Deterministic, no LLM, no faiss. A pure function of the ReportBundle ({campaign, vi, decision}).

Example
-------
>>> gen = FounderReportGenerator()
>>> rep = gen.generate(raw_campaign)
>>> rep["report_type"], [s["id"] for s in rep["sections"]][:3]
('Founder Intelligence Report', ['executive_summary', 'scorecard', 'strengths'])
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.pipeline import RecommendationPipeline
from src.reporting.base import feature_prose, section
from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("Founder Intelligence Report — a deterministic reformatting of the Day 4-12 analysis "
            "(no new analysis). Every section names its source. Model-simulated over controllable "
            "features, not causal; founder / team / revenue / moat items are deferred (no data).")


class FounderReportGenerator:
    """Render the founder intelligence report from the ReportBundle.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    cf_pipeline:
        Optional pre-built Day-8 pipeline (share the warm stack).
    rec_pipeline:
        Optional pre-built Day-11 :class:`RecommendationPipeline` (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None,
                 rec_pipeline: Optional[RecommendationPipeline] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reporting.founder")

        c = self.config.get("day13", {}).get("founder", {})
        self.top_drivers = int(c.get("top_drivers", 4))
        self.top_scenarios = int(c.get("top_scenarios", 5))
        self.out_path = Path(c.get("out_path", "artifacts/reports/founder_report.json"))

        self.rec = rec_pipeline or RecommendationPipeline(self.config, self.logger, cf_pipeline=cf_pipeline)
        self.agg = self.rec.agg                              # share the warm aggregator

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any], bundle: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Founder intelligence report for one campaign (uses a prebuilt bundle if given)."""
        if bundle is not None:
            vi, decision = bundle["vi"], bundle["decision"]
        else:
            vi = self.agg.aggregate(raw)
            decision = self.rec.run(raw)

        ir = decision["investment_readiness"]
        drivers = vi["drivers"]
        sections = [
            section("executive_summary", "Executive Summary", "Day-11 synthesis", "text",
                    decision["final_strategy"]),
            section("scorecard", "Scorecard", "Days 4 / 10 / 11", "metrics", {
                "Startup score (readiness index)": ir["readiness_index"],
                "Funding probability": f"{ir['funding_probability']:.0%}",
                "Investment readiness": f"{ir['stage']} (reachable {ir['optimized_probability']:.0%})",
                "Confidence": f"{decision['confidence']['score']:.0%}"}),
            section("strengths", "Strengths", "Day-5 SHAP (positive drivers)", "list",
                    [f"{feature_prose(d['feature'], d.get('descriptor'))} works in your favor."
                     for d in drivers["top_positive"][:self.top_drivers]]),
            section("weaknesses", "Weaknesses", "Day-5 SHAP (negative drivers)", "list",
                    [f"{feature_prose(d['feature'], d.get('descriptor'))} weighs against you."
                     for d in drivers["top_negative"][:self.top_drivers]]),
            section("priority_improvements", "Priority Improvements",
                    "Day-11 prioritization (Day-7 evidence)", "table", {
                        "columns": ["#", "Action", "Impact", "Evidence"],
                        "rows": [[a["rank"], a["action"], f"+{a['expected_impact_points']} pts",
                                  a["evidence"] or "-"] for a in decision["priority_improvements"]]}),
            section("evolution_roadmap", "Evolution Roadmap", "Day-10 preparation roadmap", "table", {
                "columns": ["Step", "Action", "Stage after", "Crosses bar"],
                "rows": [[s["order"], s["action"], s["stage_after"], "yes" if s["crosses_bar"] else "no"]
                         for s in vi["evolution"]["roadmap"]["steps"]]}),
            section("scenario_analysis", "Scenario Analysis", "Day-10 scenario simulator", "table", {
                "columns": ["Scenario", "Probability", "Stage"],
                "rows": [[s["name"], f"{s['probability']:.0%}", s["stage"]]
                         for s in vi["scenario"]["ranked_scenarios"][:self.top_scenarios]]}),
            section("next_milestones", "Next Milestones", "Day-10 milestones", "table", {
                "columns": ["Priority", "Milestone", "Action"],
                "rows": [[m["priority"], m["milestone"], m["action"]]
                         for m in vi["evolution"]["milestones"]["milestones"]]}),
        ]

        report = {
            "report_type": "Founder Intelligence Report",
            "campaign": decision["campaign"],
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "sections": sections,
            "deferred_appendix": {
                "note": "Not covered — the dataset has no data for these; not fabricated.",
                "items": sorted(decision["deferred"].keys())},
            "honesty_note": _CAPTION,
        }
        self.logger.info("Founder report: %s (%d sections, readiness %d).",
                         decision["campaign"]["name"], len(sections), ir["readiness_index"])
        return report

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Founder reports for a spread of test campaigns -> deliverable JSON."""
        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.generate(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved founder reports -> %s", self.out_path)
        return self.out_path
