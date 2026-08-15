"""Phase G — Investment Analyst Agent (Day-12, Layer 12).

The investment partner — the sixth agent, whose job is to **synthesize the other five**. It reads
the market / founder / financial / competition / risk reports plus the Day-10 readiness stage, forms
a **confidence-weighted investment score**, and issues a preliminary recommendation band:

    Strong Invest | Proceed to Due Diligence | Monitor Progress | Do Not Invest

This is *one agent's synthesized opinion*, computed as a **confidence-weighted mean of the specialist
scores** — a deliberately different mechanism from the Phase-I consensus, which does a **stance vote
with a risk veto**. Comparing the two is informative (do the score-mean and the stance-vote agree?).

**Deferred:** expected return / IRR / valuation — no financial-return data. Deterministic, no faiss.

Example
-------
>>> ia = InvestmentAgent()
>>> ia.analyze(raw_campaign)["key_metric"]
{'name': 'preliminary_recommendation', 'value': 'Do Not Invest'}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.agents.base import agent_report, clip
from src.agents.competition_agent import CompetitionAgent
from src.agents.financial_agent import FinancialAgent
from src.agents.founder_agent import FounderAgent
from src.agents.market_agent import MarketAgent
from src.agents.risk_agent import RiskAgent
from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.intelligence_aggregator import IntelligenceAggregator
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED = {
    b: {"status": "deferred", "reason": "no financial-return data in a Kickstarter snapshot"}
    for b in ["expected_return", "irr", "valuation", "exit_multiple"]
}
_CAPTION = ("Investment synthesis = confidence-weighted mean of the five specialist agents + "
            "readiness. One agent's opinion (a score-mean), distinct from the Phase-I consensus "
            "(a stance-vote with veto). Model-simulated, not causal; expected return / IRR / "
            "valuation are deferred.")


class InvestmentAgent:
    """Synthesize the five specialist agents into a preliminary investment recommendation.

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
        self.logger = logger or get_logger("avip.agents.investment")

        c = self.config.get("day12", {}).get("investment", {})
        self.positive_at = float(c.get("positive_at", 60))
        self.negative_at = float(c.get("negative_at", 40))
        self.strong_at = float(c.get("strong_invest_at", 70))
        self.proceed_at = float(c.get("proceed_dd_at", 55))
        self.monitor_at = float(c.get("monitor_at", 40))
        self.out_path = Path(c.get("out_path", "artifacts/agents/investment_analysis.json"))

        self.agg = aggregator or IntelligenceAggregator(self.config, self.logger, cf_pipeline=cf_pipeline)
        # the five specialists (share the warm aggregator) — used when reports aren't supplied.
        self.specialists = [MarketAgent(self.config, self.logger, aggregator=self.agg),
                            FounderAgent(self.config, self.logger, aggregator=self.agg),
                            FinancialAgent(self.config, self.logger, aggregator=self.agg),
                            CompetitionAgent(self.config, self.logger, aggregator=self.agg),
                            RiskAgent(self.config, self.logger, aggregator=self.agg)]

    # ------------------------------------------------------------------ public
    def analyze(self, raw: dict[str, Any], intelligence: Optional[dict[str, Any]] = None,
                agent_reports: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        """Investment-synthesis AgentReport (reuses specialist reports if the orchestrator supplies them)."""
        vi = intelligence if intelligence is not None else self.agg.aggregate(raw)
        reports = agent_reports if agent_reports is not None else \
            [a.analyze(raw, intelligence=vi) for a in self.specialists]

        num = sum(r["score"] * r["confidence"] for r in reports)
        den = sum(r["confidence"] for r in reports)
        inv_score = clip(num / den if den else float(np.mean([r["score"] for r in reports])), 0, 100)
        band = self._band(inv_score)

        stage = vi["evolution"]["stage"]
        n_pos = sum(1 for r in reports if r["stance"] == "positive")
        n_neg = sum(1 for r in reports if r["stance"] == "negative")
        n_cau = len(reports) - n_pos - n_neg
        best = max(reports, key=lambda r: r["score"]); worst = min(reports, key=lambda r: r["score"])
        findings = [
            f"{n_pos} positive / {n_cau} cautious / {n_neg} negative across {len(reports)} specialist agents.",
            f"Funding-readiness: {stage['stage']} ({stage['funding_probability']:.0%} -> reachable "
            f"{stage['optimized_probability']:.0%}).",
            f"Strongest view: {best['agent']} ({best['score']}); weakest: {worst['agent']} ({worst['score']}).",
        ]
        report = agent_report(
            "InvestmentAgent", "Committee synthesis over the five specialist agents", inv_score,
            self._confidence(reports), findings,
            {"name": "preliminary_recommendation", "value": band},
            _DEFERRED, _CAPTION, self.positive_at, self.negative_at)
        self.logger.info("InvestmentAgent: %s %d (conf %.2f) — '%s'.", report["stance"],
                         report["score"], report["confidence"], band)
        return report

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Investment syntheses for a spread of test campaigns -> deliverable JSON."""
        df = pd.read_csv(self.config.get("day6", {}).get("knowledge_base", {})
                         .get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(self.config.get("day6", {}).get("knowledge_base", {})
                          .get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.analyze(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved investment syntheses -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _band(self, score: float) -> str:
        if score >= self.strong_at:
            return "Strong Invest"
        if score >= self.proceed_at:
            return "Proceed to Due Diligence"
        if score >= self.monitor_at:
            return "Monitor Progress"
        return "Do Not Invest"

    def _confidence(self, reports: list[dict[str, Any]]) -> float:
        """Confident when the specialists AGREE (low stance dispersion) and are individually sure."""
        stances = [r["stance"] for r in reports]
        agreement = max(stances.count(s) for s in set(stances)) / len(stances) if stances else 0.5
        mean_conf = float(np.mean([r["confidence"] for r in reports])) if reports else 0.5
        return min(0.99, max(0.5, 0.5 * agreement + 0.5 * mean_conf))
