"""Phase C — Investor Due Diligence Report Generator (Day-13, Layer 13).

Assemble the investor-facing report — **presentation only, no new analysis**. It's largely a
rendering of the Day-12 committee: the recommendation, the six specialist agents' verdicts, the
risks/red-flags, and the vote table — each section tagged with its ``source``.

    Investment Recommendation  <- Day-12 committee (band + confidence + agreement)
    Market / Competitive / Founder / Financial assessment  <- Day-12 specialist agents
    Similar startups           <- Day-9 category cohort (our data)
    Risks & red flags          <- Day-12 risk agent + Day-9 investor objections
    Opportunities              <- Day-11
    Investment decision        <- Day-12 committee (votes + veto)
    Deferred appendix          <- expected return / comparable rounds / valuation (no data)

Deterministic, no LLM, no faiss. A pure function of the ReportBundle ({campaign, vi, decision,
committee}).

Example
-------
>>> gen = InvestorReportGenerator()
>>> gen.generate(raw_campaign)["report_type"]
'Investor Due Diligence Report'
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.agents.consensus import ConsensusEngine
from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.pipeline import RecommendationPipeline
from src.reporting.base import section
from src.utils.config import load_config
from src.utils.logger import get_logger

# Investor-side items the report refuses to fabricate (no data).
_DEFERRED_INVESTOR = ["expected_return", "irr", "valuation", "comparable_funding_rounds", "exit_multiple"]
_CAPTION = ("Investor Due Diligence Report — a deterministic reformatting of the Day 4-12 analysis "
            "(no new analysis). Every section names its source; agents are correlated views of one "
            "signal set, not independent experts. Model-simulated, not causal; expected return / "
            "comparable rounds / valuation / founder background / financials are deferred (no data).")


class InvestorReportGenerator:
    """Render the investor due-diligence report from the ReportBundle.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    cf_pipeline:
        Optional pre-built Day-8 pipeline (share the warm stack).
    rec_pipeline, consensus:
        Optional pre-built Day-11 pipeline / Day-12 consensus engine (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None,
                 rec_pipeline: Optional[RecommendationPipeline] = None,
                 consensus: Optional[ConsensusEngine] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reporting.investor")

        c = self.config.get("day13", {}).get("investor", {})
        self.out_path = Path(c.get("out_path", "artifacts/reports/investor_report.json"))

        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)
        self.rec = rec_pipeline or RecommendationPipeline(self.config, self.logger, cf_pipeline=self.cf)
        self.consensus = consensus or ConsensusEngine(self.config, self.logger, cf_pipeline=self.cf)
        self.agg = self.rec.agg

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any], bundle: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Investor due-diligence report for one campaign (uses a prebuilt bundle if given)."""
        if bundle is not None:
            vi, decision, committee = bundle["vi"], bundle["decision"], bundle["committee"]
        else:
            vi = self.agg.aggregate(raw)
            decision = self.rec.run(raw)
            committee = self.consensus.run(raw)

        agents = {r["agent"]: r for r in committee["agent_reports"]}
        cohort = vi["cohort"]
        objections = vi["copilot"].get("investor_objections", [])

        sections = [
            section("executive_summary", "Executive Summary", "Day-12 committee", "text",
                    committee["rationale"]),
            section("investment_recommendation", "Investment Recommendation", "Day-12 committee",
                    "metrics", {
                        "Recommendation": committee["committee_recommendation"],
                        "Overall confidence": f"{committee['overall_confidence']:.0%}",
                        "Agreement": f"{committee['agreement_level']:.0%}",
                        "Committee score": f"{committee['committee_score']:+.2f}"}),
            self._agent_section("market_analysis", "Market Analysis", "Day-12 market agent",
                                agents.get("MarketAgent")),
            self._agent_section("competitive_landscape", "Competitive Landscape",
                                "Day-12 competition agent", agents.get("CompetitionAgent")),
            self._agent_section("founder_assessment", "Founder Assessment (creator execution)",
                                "Day-12 founder agent", agents.get("FounderAgent")),
            self._agent_section("financial_assessment", "Financial Assessment (funding viability)",
                                "Day-12 financial agent", agents.get("FinancialAgent")),
            section("similar_startups", "Similar Startups", "Day-9 category cohort (our data)",
                    "metrics", {
                        "Category": cohort["main_category"],
                        "Similar campaigns": f"{cohort['n_similar']:,}",
                        "Category funding rate": f"{cohort['cohort_funding_rate']:.0%}",
                        "Platform baseline": f"{cohort['platform_base_rate']:.0%}"}),
            section("risks_and_red_flags", "Risks & Red Flags", "Day-12 risk agent + Day-9 objections",
                    "list",
                    (agents.get("RiskAgent", {}).get("findings", []) +
                     [f"Investor objection ({o.get('severity')}): {o.get('concern')}" for o in objections])),
            section("opportunities", "Opportunities", "Day-11", "list",
                    [o["detail"] for o in decision["opportunities"]]),
            section("investment_decision", "Investment Decision", "Day-12 committee", "table", {
                "columns": ["Agent", "Vote", "Confidence", "Weight"],
                "rows": [[v["agent"], v["stance"], f"{v['confidence']:.2f}", v["weight"]]
                         for v in committee["agent_votes"]]}),
        ]

        report = {
            "report_type": "Investor Due Diligence Report",
            "campaign": decision["campaign"],
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "recommendation": committee["committee_recommendation"],
            "risk_veto": committee["risk_veto"],
            "sections": sections,
            "deferred_appendix": {
                "note": "Not covered — the dataset has no data for these; not fabricated.",
                "items": sorted(set(decision["deferred"].keys()) | set(_DEFERRED_INVESTOR))},
            "honesty_note": _CAPTION,
        }
        self.logger.info("Investor report: %s (%s, %d sections).", decision["campaign"]["name"],
                         committee["committee_recommendation"], len(sections))
        return report

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Investor reports for a spread of test campaigns -> deliverable JSON."""
        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.generate(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved investor reports -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    @staticmethod
    def _agent_section(id: str, title: str, source: str, agent: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Render one agent report as a verdict line + its findings."""
        if not agent:
            return section(id, title, source, "list", ["(agent unavailable)"])
        head = (f"Verdict: {agent['stance']} (score {agent['score']}/100, "
                f"confidence {agent['confidence']:.0%}).")
        return section(id, title, source, "list", [head] + list(agent["findings"]))
