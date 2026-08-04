"""Phase C — Founder Analyst Agent (Creator Execution) (Day-12, Layer 12).

The most-reframed agent. Kickstarter is **anonymous pre-launch** — there is NO founder experience,
team, leadership, or advisory data. So this agent does **not** evaluate a person; it evaluates the
**creator's controllable campaign craft** — the goal, title, and structure choices the creator
actually made.

The honest execution metric: **how much of the reachable funding probability has the creator already
captured?** — ``execution_ratio = base_p / optimized_p``. A creator whose controllable choices are
near their modeled optimum scores high; one who left large, fixable upside on the table (an
unrealistic goal, a weak title) scores low. The specific unrealized levers (Day-8) are surfaced as
**craft gaps**.

**Deferred:** founder experience / team composition / leadership / advisory board / domain
expertise / track record — none exist in the dataset. Deterministic, no faiss.

Example
-------
>>> fa = FounderAgent()
>>> r = fa.analyze(raw_campaign)
>>> r["agent"], r["key_metric"]["name"]
('FounderAgent', 'execution_ratio')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.agents.base import agent_report, clip, clip01
from src.counterfactual.explanation_generator import action_phrase
from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.intelligence_aggregator import IntelligenceAggregator
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED = {
    b: {"status": "deferred", "reason": "no founder/team data in an anonymous pre-launch snapshot"}
    for b in ["founder_experience", "team_composition", "leadership", "advisory_board",
              "domain_expertise", "founder_track_record"]
}
_CAPTION = ("'Founder' analysis = the creator's controllable CAMPAIGN CRAFT (goal / title / "
            "structure), not the person — Kickstarter has no founder data. Execution = share of the "
            "modeled reachable optimum already captured. Model-simulated, not causal; founder "
            "experience / team / leadership are deferred.")


class FounderAgent:
    """Score the creator's controllable campaign craft (execution), not the founder.

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
        self.logger = logger or get_logger("avip.agents.founder")

        c = self.config.get("day12", {}).get("founder", {})
        self.positive_at = float(c.get("positive_at", 60))
        self.negative_at = float(c.get("negative_at", 40))
        self.top_gaps = int(c.get("top_gaps", 2))
        self.out_path = Path(c.get("out_path", "artifacts/agents/founder_analysis.json"))

        self.agg = aggregator or IntelligenceAggregator(self.config, self.logger, cf_pipeline=cf_pipeline)

    # ------------------------------------------------------------------ public
    def analyze(self, raw: dict[str, Any],
                intelligence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Creator-execution AgentReport for one campaign."""
        vi = intelligence if intelligence is not None else self.agg.aggregate(raw)
        stage = vi["evolution"]["stage"]
        base = float(stage["funding_probability"]); opt = float(stage["optimized_probability"])
        ratio = base / opt if opt > 0 else 1.0
        score = clip(100 * ratio, 0, 100)
        recs = vi["counterfactual"]["ranked_recommendations"][:self.top_gaps]

        if score >= self.positive_at:
            findings = ["The creator's controllable choices are already near their modeled optimum "
                        f"({base:.0%} of a reachable {opt:.0%})."]
        else:
            findings = [f"The creator left ~{(opt-base)*100:.0f} points of controllable upside "
                        f"unrealized ({base:.0%} of a reachable {opt:.0%}) — craft gaps below."]
        for r in recs:
            findings.append(f"Craft gap: {action_phrase(r['feature'], r.get('current'), r.get('proposed'))} "
                            f"(~{float(r.get('delta',0))*100:.0f} pts).")

        report = agent_report(
            "FounderAgent", "Creator's controllable campaign craft (goal / title / structure)",
            score, self._confidence(vi), findings,
            {"name": "execution_ratio", "value": round(ratio, 3)},
            _DEFERRED, _CAPTION, self.positive_at, self.negative_at)
        self.logger.info("FounderAgent: %s %d (conf %.2f) — execution %.2f.", report["stance"],
                         report["score"], report["confidence"], ratio)
        return report

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Creator-execution analyses for a spread of test campaigns -> deliverable JSON."""
        df = pd.read_csv(self.config.get("day6", {}).get("knowledge_base", {})
                         .get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(self.config.get("day6", {}).get("knowledge_base", {})
                          .get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.analyze(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved founder (creator-execution) analyses -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _confidence(self, vi: dict[str, Any]) -> float:
        """Confidence from how well-evidenced the identified craft gaps are (Day-7 strength)."""
        recs = vi["counterfactual"]["ranked_recommendations"][:3]
        strengths = []
        for r in recs:
            ev = r.get("evidence") or {}
            lift = float(ev.get("lift", 1.0)); n = int(ev.get("n", 0))
            s = clip01((lift - 1.0) / 0.6)
            strengths.append(s * 0.5 if n < 100 else s)
        conf = 0.5 + 0.49 * (float(np.mean(strengths)) if strengths else 0.2)
        return min(0.99, max(0.5, conf))
