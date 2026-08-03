"""Phase C — Evolution Planner: Pre-Launch Preparation Roadmap (Day-10, Layer 9).

The honest "growth roadmap" — but for a pre-launch campaign, it's a **preparation roadmap**:
the controllable changes (Day-8 optimization plan) ordered from the current state to funding
readiness, each step annotated with the **readiness stage after it**, so a founder sees exactly
when the campaign crosses the funding bar.

    Current: Fixable (3%)
      → Lower your funding goal to $5,000        ⇒ 8%   (still Fixable)
      → Shorten your campaign to 30 days         ⇒ 38%  (now READY ✔)
    Target: Ready (38%)

This is NOT a 12-month growth journey (acquire users → improve retention → raise pre-seed) —
those require post-launch data we don't have. Real levers only; growth milestones deferred.

Reuses the Day-8 :class:`CounterfactualPipeline` plan + Day-10 :class:`StageDetector`. No faiss.

Example
-------
>>> gen = RoadmapGenerator()
>>> r = gen.generate(raw_campaign)
>>> r["current_stage"], r["target_stage"], [s["crosses_bar"] for s in r["steps"]]
('Fixable', 'Ready', [False, True])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.evolution.stage_detector import StageDetector
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED = {
    "growth_milestones": {"status": "deferred",
                          "reason": ("post-launch milestones (users / retention / revenue / "
                                     "funding rounds) require data the dataset doesn't have")}
}
_CAPTION = ("A pre-launch preparation roadmap over controllable features (goal / duration / "
            "launch timing / title) — model-simulated, not a causal growth plan. Post-launch "
            "growth milestones are deferred.")


class RoadmapGenerator:
    """Order the controllable changes into a current → readiness preparation roadmap.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    cf_pipeline, stage_detector:
        Optional pre-built collaborators (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None,
                 stage_detector: Optional[StageDetector] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.evolution.roadmap")

        c = self.config.get("day10", {}).get("roadmap", {})
        self.out_path = Path(c.get("out_path", "artifacts/evolution/roadmap.json"))

        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)
        self.stage = stage_detector or StageDetector(self.config, self.logger, cf_pipeline=self.cf)

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any],
                 analysis: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Build the preparation roadmap for one campaign."""
        raw = dict(raw)
        analysis = analysis if analysis is not None else self.cf.analyze(raw)
        stage_info = self.stage.detect(raw, analysis=analysis)
        plan = analysis["optimization_plan"]
        threshold = float(self.cf.simulator.engine.threshold)
        base_p = analysis["baseline_p"]
        final_p = plan.get("final_p", base_p)

        steps = []
        for i, st in enumerate(plan.get("steps", []), start=1):
            cum = st.get("cumulative_p", base_p)
            steps.append({
                "order": i,
                "action": st.get("action"),
                "cumulative_probability": round(cum, 4),
                "marginal_gain": st.get("marginal_gain"),
                "stage_after": self.stage.stage_of(cum, final_p),
                "crosses_bar": cum >= threshold,
                "confidence": st.get("confidence"),
                "evidence": st.get("evidence"),
            })

        target_stage = self.stage.stage_of(final_p, final_p)
        return {
            "campaign": {k: raw.get(k) for k in ("name", "main_category", "usd_goal_real")},
            "current_stage": stage_info["stage"],
            "current_probability": round(base_p, 4),
            "steps": steps,
            "target_stage": target_stage,
            "target_probability": round(final_p, 4),
            "summary": self._summary(stage_info["stage"], base_p, target_stage, final_p, steps, threshold),
            "deferred": _DEFERRED,
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Roadmaps for a spread of improvable test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.generate(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved roadmaps -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    @staticmethod
    def _summary(cur_stage, base_p, target_stage, final_p, steps, threshold) -> str:
        if not steps:
            return (f"Currently {cur_stage} ({base_p:.0%}). No controllable changes materially "
                    f"improve the odds — the roadmap is to launch as-is or reconsider scope.")
        cross = next((s["order"] for s in steps if s["crosses_bar"]), None)
        cross_txt = (f"You cross the funding bar at step {cross}." if cross
                     else "The plan does not clear the funding bar; consider rescoping.")
        return (f"Currently {cur_stage} ({base_p:.0%}). A {len(steps)}-step preparation roadmap — "
                f"starting with '{steps[0]['action'].lower()}' — reaches {target_stage} "
                f"(~{final_p:.0%}). {cross_txt}")
