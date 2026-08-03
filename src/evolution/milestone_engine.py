"""Phase D — Milestone Recommendation Engine (Day-10, Layer 9).

Reframe the Day-8 ranked improvements as **preparation milestones**, each with the ranking
dimensions the draft asks for made explicit: **predicted impact**, **historical evidence**,
**funding influence**, and **execution difficulty**. Ranked by the Day-8 score (gain × tier ×
evidence × feasibility) — no re-ranking, so a milestone and its Day-8 recommendation agree.

Real, controllable milestones only ("set a fundable goal", "choose an optimal campaign
length") — never "validate customer problem / improve retention / expand founding team", which
require data the dataset doesn't have. Those are **deferred**.

Reuses the Day-8 :class:`CounterfactualPipeline`; no faiss → no OpenMP.

Example
-------
>>> eng = MilestoneEngine()
>>> ms = eng.generate(raw_campaign)["milestones"]
>>> ms[0]["milestone"], ms[0]["predicted_impact_points"]
('Set a fundable funding goal', 19)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.explanation_generator import action_phrase
from src.counterfactual.pipeline import CounterfactualPipeline
from src.utils.config import load_config
from src.utils.logger import get_logger

_MILESTONE_NAMES = {
    "goal": "Set a fundable funding goal",
    "campaign_duration": "Choose an optimal campaign length",
    "launch_weekday": "Time your launch well",
    "title": "Strengthen your campaign title",
}
# Execution difficulty (heuristic — all are pre-launch settings, genuinely easy to change).
_DIFFICULTY = {"goal": "Low", "campaign_duration": "Low",
               "launch_weekday": "Low", "title": "Medium"}
_DEFERRED_MILESTONES = {
    b: {"status": "deferred", "reason": "no data in this dataset"}
    for b in ["validate_customer_problem", "improve_retention", "grow_revenue",
              "expand_founding_team", "build_partnerships", "improve_unit_economics"]
}
_CAPTION = ("Milestones over controllable pre-launch features (goal / duration / timing / title), "
            "ranked by model-estimated impact + historical evidence + feasibility. Difficulty labels "
            "are heuristic. Product / team / traction milestones are deferred (no data).")


class MilestoneEngine:
    """Rank the controllable improvements as preparation milestones.

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
        self.logger = logger or get_logger("avip.evolution.milestones")

        c = self.config.get("day10", {}).get("milestones", {})
        self.out_path = Path(c.get("out_path", "artifacts/evolution/milestones.json"))

        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any],
                 analysis: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Ranked preparation milestones for one campaign."""
        raw = dict(raw)
        analysis = analysis if analysis is not None else self.cf.analyze(raw)
        # One milestone per controllable feature — the best-ranked option (recs are score-sorted).
        seen: set[str] = set()
        best = [r for r in analysis["ranked_recommendations"]
                if not (r["feature"] in seen or seen.add(r["feature"]))]
        milestones = [dict(m, priority=i) for i, m in
                      enumerate((self._milestone(r) for r in best), start=1)]
        return {
            "campaign": {k: raw.get(k) for k in ("name", "main_category", "usd_goal_real")},
            "milestones": milestones,
            "deferred_milestones": _DEFERRED_MILESTONES,
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Milestones for a spread of improvable test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.generate(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved milestones -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    @staticmethod
    def _milestone(rec: dict[str, Any]) -> dict[str, Any]:
        feature = rec["feature"]
        ev = rec.get("evidence") or {}
        return {
            "priority": rec.get("priority"),
            "milestone": _MILESTONE_NAMES.get(feature, feature),
            "action": action_phrase(feature, rec.get("current"), rec.get("proposed")),
            # the four ranking dimensions the draft asks for:
            "predicted_impact_points": round(float(rec.get("delta", 0)) * 100),
            "funding_influence": round(float(rec.get("delta", 0)), 4),
            "historical_evidence": ev.get("statement"),
            "execution_difficulty": _DIFFICULTY.get(feature, "Medium"),
            "confidence": rec.get("confidence"),
            "score": rec.get("score"),
        }
