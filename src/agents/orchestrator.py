"""Phase H — Agent Orchestrator (Day-12, Layer 12).

Coordinate the committee: aggregate the campaign **once**, run all six agents on that one shared
``VentureIntelligence``, and collect their uniform reports for the consensus engine. The five
specialists run first; their reports are passed **into** the investment agent so it synthesizes
without re-running them (handle the one dependency in the graph).

    aggregate (once) -> [market, founder, financial, competition, risk] -> investment(reports) -> collect

One warm faiss-free stack, one model in memory, deterministic. The orchestrator owns sequencing and
sharing; it computes nothing itself (consensus is Phase I).

Example
-------
>>> orch = AgentOrchestrator()
>>> out = orch.run(raw_campaign)
>>> [r["agent"] for r in out["agent_reports"]]
['MarketAgent', 'FounderAgent', 'FinancialAgent', 'CompetitionAgent', 'RiskAgent', 'InvestmentAgent']
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.agents.competition_agent import CompetitionAgent
from src.agents.financial_agent import FinancialAgent
from src.agents.founder_agent import FounderAgent
from src.agents.investment_agent import InvestmentAgent
from src.agents.market_agent import MarketAgent
from src.agents.risk_agent import RiskAgent
from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.intelligence_aggregator import IntelligenceAggregator
from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("Six specialist agents run on ONE shared VentureIntelligence — correlated views of one "
            "signal set, not independent experts. Deterministic, no LLM, no faiss. Consensus is "
            "Phase I.")


class AgentOrchestrator:
    """Run all six agents on one aggregation and collect their reports.

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
        self.logger = logger or get_logger("avip.agents.orchestrator")

        c = self.config.get("day12", {}).get("orchestrator", {})
        self.out_path = Path(c.get("out_path", "artifacts/agents/agent_reports.json"))

        # ONE aggregator (=> one warm cf), shared into every agent.
        self.agg = IntelligenceAggregator(self.config, self.logger, cf_pipeline=cf_pipeline)
        self.specialists = [
            MarketAgent(self.config, self.logger, aggregator=self.agg),
            FounderAgent(self.config, self.logger, aggregator=self.agg),
            FinancialAgent(self.config, self.logger, aggregator=self.agg),
            CompetitionAgent(self.config, self.logger, aggregator=self.agg),
            RiskAgent(self.config, self.logger, aggregator=self.agg),
        ]
        self.investment = InvestmentAgent(self.config, self.logger, aggregator=self.agg)
        self.logger.info("AgentOrchestrator ready (6 agents, one warm faiss-free stack).")

    # ------------------------------------------------------------------ public
    def run(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Run all six agents on one aggregation; collect their reports."""
        vi = self.agg.aggregate(raw)                        # ONE aggregation, threaded into every agent
        specialist_reports = [a.analyze(raw, intelligence=vi) for a in self.specialists]
        investment_report = self.investment.analyze(
            raw, intelligence=vi, agent_reports=specialist_reports)   # reuse — no re-run
        reports = specialist_reports + [investment_report]

        n_pos = sum(1 for r in reports if r["stance"] == "positive")
        n_neg = sum(1 for r in reports if r["stance"] == "negative")
        self.logger.info("Orchestrated 6 agents: %d positive / %d negative / %d cautious.",
                         n_pos, n_neg, len(reports) - n_pos - n_neg)
        return {
            "campaign": vi["campaign"],
            "agent_reports": reports,
            "summary": {
                "n_positive": n_pos, "n_cautious": len(reports) - n_pos - n_neg, "n_negative": n_neg,
                "mean_confidence": round(float(np.mean([r["confidence"] for r in reports])), 3),
                "preliminary_recommendation": investment_report["key_metric"]["value"],
            },
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Orchestrated agent reports for a spread of test campaigns -> deliverable JSON."""
        df = pd.read_csv(self.config.get("day6", {}).get("knowledge_base", {})
                         .get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(self.config.get("day6", {}).get("knowledge_base", {})
                          .get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.run(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved orchestrated agent reports -> %s", self.out_path)
        return self.out_path
