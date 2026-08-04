"""Phase D — Financial Analyst Agent (Funding Viability) (Day-12, Layer 12).

The finance specialist. Kickstarter has no P&L — the only "financial" number is the **funding
goal**. So this agent scores **funding viability**: is the goal realistic for the category, and are
the funding odds sound? It blends two signals:

- **funding odds** — the calibrated probability vs the decision bar (Day 4).
- **goal realism** — the goal vs the category winners' median (Day 9); a goal many multiples above
  what funded campaigns asked is a viability red flag.

    viability = odds_weight * odds + realism_weight * realism

**Deferred:** burn rate / runway / revenue / revenue growth / CAC-LTV / unit economics /
profitability — none exist in the dataset. Deterministic, no faiss.

Example
-------
>>> fa = FinancialAgent()
>>> r = fa.analyze(raw_campaign)
>>> r["agent"], r["key_metric"]["name"]
('FinancialAgent', 'goal_to_median_ratio')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.agents.base import agent_report, clip, clip01
from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.intelligence_aggregator import IntelligenceAggregator
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED = {
    b: {"status": "deferred", "reason": "no financial statements in a Kickstarter snapshot"}
    for b in ["burn_rate", "runway", "revenue", "revenue_growth", "cac_ltv",
              "unit_economics", "profitability"]
}
_CAPTION = ("Funding viability = goal realism (vs category winners' median) + funding odds (vs the "
            "decision bar). The only 'financial' signal in the data is the goal. Model-simulated, "
            "not causal; burn / runway / revenue / CAC-LTV are deferred.")


class FinancialAgent:
    """Score funding viability from goal realism + funding odds.

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
        self.logger = logger or get_logger("avip.agents.financial")

        c = self.config.get("day12", {}).get("financial", {})
        self.positive_at = float(c.get("positive_at", 60))
        self.negative_at = float(c.get("negative_at", 40))
        self.odds_w = float(c.get("odds_weight", 0.65))
        self.realism_w = float(c.get("realism_weight", 0.35))
        self.penalty_per_x = float(c.get("realism_penalty_per_x", 20))
        self.out_path = Path(c.get("out_path", "artifacts/agents/financial_analysis.json"))

        self.agg = aggregator or IntelligenceAggregator(self.config, self.logger, cf_pipeline=cf_pipeline)

    # ------------------------------------------------------------------ public
    def analyze(self, raw: dict[str, Any],
                intelligence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Funding-viability AgentReport for one campaign."""
        vi = intelligence if intelligence is not None else self.agg.aggregate(raw)
        prob = float(vi["prediction"]["funding_probability"])
        threshold = float(vi["prediction"]["threshold"])
        goal = float(vi["campaign"].get("usd_goal_real", 0.0))
        pos = vi["copilot"]["competition_analysis"].get("your_position", {})
        median = float(pos.get("winners_median_goal_usd") or 0.0)
        ratio = goal / median if median else 1.0

        odds = clip(50 * prob / threshold, 0, 100) if threshold else clip(prob * 100, 0, 100)
        realism = clip(100 - (ratio - 1) * self.penalty_per_x, 0, 100)
        score = clip(self.odds_w * odds + self.realism_w * realism, 0, 100)

        findings = [
            f"Goal ${goal:,.0f} is {ratio:.1f}x the category winners' median (${median:,.0f}).",
            f"Calibrated funding probability {prob:.0%} vs the {threshold:.0%} decision bar.",
        ]
        gr = next((r for r in vi["copilot"]["risk_analysis"].get("risks", [])
                   if "goal" in str(r.get("category", "")).lower()), None)
        if gr:
            findings.append(f"Goal-size risk ({gr.get('severity')}): {gr.get('reason')}")

        report = agent_report(
            "FinancialAgent", "Funding viability (goal realism + funding odds)", score,
            self._confidence(prob, threshold, ratio), findings,
            {"name": "goal_to_median_ratio", "value": round(ratio, 2)},
            _DEFERRED, _CAPTION, self.positive_at, self.negative_at)
        self.logger.info("FinancialAgent: %s %d (conf %.2f) — goal %.1fx median.", report["stance"],
                         report["score"], report["confidence"], ratio)
        return report

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Funding-viability analyses for a spread of test campaigns -> deliverable JSON."""
        df = pd.read_csv(self.config.get("day6", {}).get("knowledge_base", {})
                         .get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(self.config.get("day6", {}).get("knowledge_base", {})
                          .get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.analyze(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved financial (funding-viability) analyses -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _confidence(self, prob: float, threshold: float, ratio: float) -> float:
        """Confident when odds are decisive (far from the bar) and the goal ratio is clearly hi/lo."""
        margin = clip01(abs(prob - threshold) / 0.30)
        ratio_clarity = clip01(abs(ratio - 1.0) / 3.0)
        return min(0.99, max(0.5, 0.5 + 0.49 * (0.6 * margin + 0.4 * ratio_clarity)))
