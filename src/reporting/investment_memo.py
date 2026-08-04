"""Phase D — Investment Memo Generator (Day-13, Layer 13).

A VC-style **narrative** memo — the same bundle as the investor report, but rendered as prose
paragraphs culminating in a deterministic **investment thesis**. Presentation only, no new analysis;
every section names its ``source``.

    Executive Summary -> Startup Overview -> Market Opportunity -> Competitive Position
    -> Founder (creator) Analysis -> Financial (funding) Analysis -> Risk Assessment
    -> Investment Thesis (deterministic synthesis) -> Recommendation

The **investment thesis** is a deterministic synthesis: the committee band + confidence, the
strongest agent as the bull case, the weakest agent as the bear case, and the readiness framing.
Deterministic, no LLM, no faiss.

Example
-------
>>> gen = InvestmentMemoGenerator()
>>> gen.generate(raw_campaign)["report_type"]
'Investment Memo'
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

_DEFERRED_MEMO = ["expected_return", "irr", "valuation", "comparable_funding_rounds",
                  "exit_multiple", "founder_background", "cap_table"]
_CAPTION = ("Investment Memo — a deterministic narrative reformatting of the Day 4-12 analysis (no "
            "new analysis). The investment thesis is a synthesis of the committee + readiness + the "
            "bull/bear agents. Model-simulated, not causal; expected return / valuation / comparable "
            "rounds / founder background / financials are deferred (no data).")


class InvestmentMemoGenerator:
    """Render a VC-style investment memo from the ReportBundle.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    cf_pipeline, rec_pipeline, consensus:
        Optional pre-built collaborators (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None,
                 rec_pipeline: Optional[RecommendationPipeline] = None,
                 consensus: Optional[ConsensusEngine] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reporting.memo")

        c = self.config.get("day13", {}).get("memo", {})
        self.out_path = Path(c.get("out_path", "artifacts/reports/investment_memo.json"))

        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)
        self.rec = rec_pipeline or RecommendationPipeline(self.config, self.logger, cf_pipeline=self.cf)
        self.consensus = consensus or ConsensusEngine(self.config, self.logger, cf_pipeline=self.cf)
        self.agg = self.rec.agg

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any], bundle: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Investment memo for one campaign (uses a prebuilt bundle if given)."""
        if bundle is not None:
            vi, decision, committee = bundle["vi"], bundle["decision"], bundle["committee"]
        else:
            vi = self.agg.aggregate(raw)
            decision = self.rec.run(raw)
            committee = self.consensus.run(raw)

        agents = {r["agent"]: r for r in committee["agent_reports"]}
        camp = decision["campaign"]
        ir = decision["investment_readiness"]
        band = committee["committee_recommendation"]; conf = committee["overall_confidence"]

        sections = [
            section("executive_summary", "Executive Summary", "Day-12 committee", "text",
                    f"{camp['name']} ({camp.get('main_category')}, goal ${camp['usd_goal_real']:,.0f}). "
                    f"The committee recommends {band} at {conf:.0%} confidence."),
            section("startup_overview", "Startup Overview", "Days 4 / 10", "metrics", {
                "Name": camp["name"], "Category": camp.get("main_category"),
                "Funding goal": f"${camp['usd_goal_real']:,.0f}",
                "Funding probability": f"{ir['funding_probability']:.0%}",
                "Readiness stage": ir["stage"]}),
            section("market_opportunity", "Market Opportunity", "Day-12 market agent", "text",
                    self._prose(agents.get("MarketAgent"))),
            section("competitive_position", "Competitive Position", "Day-12 competition agent", "text",
                    self._prose(agents.get("CompetitionAgent"))),
            section("founder_analysis", "Founder Analysis (creator execution)", "Day-12 founder agent",
                    "text", self._prose(agents.get("FounderAgent"))),
            section("financial_analysis", "Financial Analysis (funding viability)",
                    "Day-12 financial agent", "text", self._prose(agents.get("FinancialAgent"))),
            section("risk_assessment", "Risk Assessment", "Day-12 risk agent", "text",
                    self._prose(agents.get("RiskAgent"))),
            section("investment_thesis", "Investment Thesis", "Day-12 committee + Days 10/11 synthesis",
                    "text", self._thesis(committee, decision, agents)),
            section("recommendation", "Recommendation", "Day-12 committee", "metrics", {
                "Recommendation": band, "Overall confidence": f"{conf:.0%}",
                "Committee score": f"{committee['committee_score']:+.2f}",
                "Risk veto applied": committee["risk_veto"]["applied"]}),
        ]

        report = {
            "report_type": "Investment Memo",
            "campaign": camp,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "recommendation": band,
            "sections": sections,
            "deferred_appendix": {
                "note": "Not covered — the dataset has no data for these; not fabricated.",
                "items": sorted(set(decision["deferred"].keys()) | set(_DEFERRED_MEMO))},
            "honesty_note": _CAPTION,
        }
        self.logger.info("Investment memo: %s (%s, %d sections).", camp["name"], band, len(sections))
        return report

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Investment memos for a spread of test campaigns -> deliverable JSON."""
        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.generate(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved investment memos -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    @staticmethod
    def _prose(agent: Optional[dict[str, Any]]) -> str:
        """Render one agent report as a prose paragraph (verdict + findings)."""
        if not agent:
            return "(agent unavailable)"
        return (f"{agent['stance'].capitalize()} (score {agent['score']}/100). "
                + " ".join(agent["findings"]))

    @staticmethod
    def _thesis(committee: dict[str, Any], decision: dict[str, Any],
                agents: dict[str, Any]) -> str:
        """Deterministic investment thesis: bull (best agent) + bear (worst agent) + readiness."""
        reports = committee["agent_reports"]
        best = max(reports, key=lambda r: r["score"])
        worst = min(reports, key=lambda r: r["score"])
        band, conf = committee["committee_recommendation"], committee["overall_confidence"]
        ir = decision["investment_readiness"]
        stage, base, opt = ir["stage"], ir["funding_probability"], ir["optimized_probability"]

        lead = f"The committee recommends {band} at {conf:.0%} confidence."
        bull = f"Bull case ({best['agent'].replace('Agent', '')}): {best['findings'][0]}"
        bear = f"Bear case ({worst['agent'].replace('Agent', '')}): {worst['findings'][0]}"
        if stage == "Fixable":
            read = (f"As presented the campaign is not fundable ({base:.0%}), but it is fixable — "
                    f"reachable to ~{opt:.0%} with the priority pre-launch changes.")
            concl = "Conditional on those fixes the thesis could clear the bar; unchanged, pass."
        elif stage in ("Ready", "Strong"):
            read = f"The campaign is already fundable ({base:.0%}, reachable ~{opt:.0%})."
            concl = "The thesis is supportable, subject to standard diligence (deferred here)."
        else:                                                # Not ready
            read = (f"Even optimized the campaign falls short (~{opt:.0%} vs the bar); the "
                    f"fundamentals need rework.")
            concl = "The thesis does not hold as presented."
        return " ".join([lead, bull, bear, read, concl])
