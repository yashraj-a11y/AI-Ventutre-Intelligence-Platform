"""Phase E — Competition Analyst Agent (Day-12, Layer 12).

The competition specialist. The honest "competition" for a Kickstarter campaign is the **other
campaigns in its category** (our own data), not named rivals. It reads Day-9 ``competition_analysis``
— competitive intensity / crowding, what winning campaigns do, and whether this campaign is
**positioned like winners or like strugglers** — and scores competitive standing.

    score = position_anchor (winners / strugglers) + crowding adjustment

**Overlap caveat (C-7):** the winners-vs-strugglers position is derived from goal-vs-median, which
the financial agent also reads — so this agent's stance is *correlated* with the financial agent's,
not independent. The distinct competitive signal it adds is the **crowding / intensity** lens.

**Deferred:** competitive moat / differentiation / named competitors / network effects — no product
or competitor data. Deterministic, no faiss (cohort = the category, not semantic retrieval).

Example
-------
>>> ca = CompetitionAgent()
>>> ca.analyze(raw_campaign)["key_metric"]
{'name': 'positioned_like', 'value': 'strugglers'}
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

_POSITION = {"winners": 78, "strugglers": 25}                 # position anchor score
_CROWD = {"Crowded": -8, "Moderate": 0, "Sparse": 6, "Niche": 6}
_DEFERRED = {
    b: {"status": "deferred", "reason": "no product / competitor data in this dataset"}
    for b in ["competitive_moat", "differentiation", "named_competitors", "network_effects"]
}
_CAPTION = ("Competition = other campaigns in your category (our own data), not named rivals. The "
            "winners-vs-strugglers position overlaps the financial agent (both read goal-vs-median) "
            "— not an independent signal. Moat / differentiation / named competitors are deferred.")


class CompetitionAgent:
    """Score competitive standing from cohort intensity + position-vs-winners.

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
        self.logger = logger or get_logger("avip.agents.competition")

        c = self.config.get("day12", {}).get("competition", {})
        self.positive_at = float(c.get("positive_at", 60))
        self.negative_at = float(c.get("negative_at", 40))
        self.n_full = float(c.get("n_full_confidence", 5000))
        self.out_path = Path(c.get("out_path", "artifacts/agents/competition_analysis.json"))

        self.agg = aggregator or IntelligenceAggregator(self.config, self.logger, cf_pipeline=cf_pipeline)

    # ------------------------------------------------------------------ public
    def analyze(self, raw: dict[str, Any],
                intelligence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Competition AgentReport for one campaign."""
        vi = intelligence if intelligence is not None else self.agg.aggregate(raw)
        comp = vi["copilot"]["competition_analysis"]
        intensity = comp.get("competitive_intensity", {})
        level = intensity.get("level")
        n = int(intensity.get("n_campaigns", 0))
        share = float(intensity.get("share_of_all", 0.0))
        pos = comp.get("your_position", {})
        positioned = pos.get("positioned_like")

        score = clip(_POSITION.get(positioned, 50) + _CROWD.get(level, 0), 0, 100)

        findings = []
        if comp.get("what_winners_do"):
            findings.append(comp["what_winners_do"][0])
        if pos.get("notes"):
            findings.append(pos["notes"][0])
        findings.append(f"Category is {str(level).lower()} — {n:,} campaigns ({share:.0%} of all).")

        report = agent_report(
            "CompetitionAgent", "Cohort competition & positioning vs winners", score,
            self._confidence(vi, n), findings,
            {"name": "positioned_like", "value": positioned},
            {**_DEFERRED, **comp.get("deferred", {})}, _CAPTION, self.positive_at, self.negative_at)
        self.logger.info("CompetitionAgent: %s %d (conf %.2f) — positioned like %s.", report["stance"],
                         report["score"], report["confidence"], positioned)
        return report

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Competition analyses for a spread of test campaigns -> deliverable JSON."""
        df = pd.read_csv(self.config.get("day6", {}).get("knowledge_base", {})
                         .get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(self.config.get("day6", {}).get("knowledge_base", {})
                          .get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.analyze(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved competition analyses -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _confidence(self, vi: dict[str, Any], n: int) -> float:
        """Confident with a big cohort and a clearly-positioned goal (far from the winners' median)."""
        n_comp = clip01(n / self.n_full)
        pos = vi["copilot"]["competition_analysis"].get("your_position", {})
        goal = float(pos.get("goal_usd") or 0.0); median = float(pos.get("winners_median_goal_usd") or 0.0)
        ratio = goal / median if median else 1.0
        clarity = clip01(abs(ratio - 1.0) / 3.0)
        return min(0.99, max(0.5, 0.5 + 0.49 * (0.6 * n_comp + 0.4 * clarity)))
