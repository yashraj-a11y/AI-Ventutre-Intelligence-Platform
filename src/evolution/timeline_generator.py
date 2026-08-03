"""Phase H — Funding-Readiness Preparation Timeline (Day-10, Layer 9).

The honest "timeline". A pre-launch campaign has no post-launch history, so this is **not** a
12-month growth journey (acquire users → retention → raise a round). It's a **preparation
timeline**: the controllable levers placed on a countdown-to-launch schedule — a sequence of
decisions to lock in during the run-up — each checkpoint tagged with the readiness stage it
unlocks.

    Week -4  Lower your funding goal to $5,000     ⇒ Fixable (8%)
    Week -2  Shorten your campaign to 30 days      ⇒ READY   (38%)  ✔ funding bar
    Launch   Go live                               ⇒ Ready   (38%)

These are the **same levers** as the Phase-C roadmap and Phase-D milestones — three framings of
one controllable lever set; this one adds the *time axis*. It reuses the Phase-C
:class:`RoadmapGenerator` (which already orders the steps and annotates each with its readiness
stage), and simply distributes those steps across the lead window. Long-horizon post-launch
growth timelines are **deferred** (no data). No faiss → no OpenMP.

Example
-------
>>> gen = TimelineGenerator()
>>> t = gen.generate(raw_campaign)
>>> [(c["when"], c["readiness_after"]) for c in t["checkpoints"]]
[('Week -4', 'Fixable'), ('Week -2', 'Ready'), ('Launch day', 'Ready')]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.evolution.roadmap_generator import RoadmapGenerator
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED = {
    "growth_timeline": {"status": "deferred",
                        "reason": ("a multi-month post-launch growth timeline (users / retention / "
                                   "revenue / funding rounds over time) needs longitudinal data "
                                   "the dataset doesn't have")}
}
_CAPTION = ("The SAME controllable levers as the roadmap (Phase C) and milestones (Phase D), placed "
            "on a pre-launch preparation schedule — decisions to lock in before launch, "
            "model-simulated, not a causal multi-month growth plan. Post-launch growth deferred.")


class TimelineGenerator:
    """Place the preparation levers on a countdown-to-launch readiness schedule.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    cf_pipeline, roadmap_generator:
        Optional pre-built collaborators (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None,
                 roadmap_generator: Optional[RoadmapGenerator] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.evolution.timeline")

        c = self.config.get("day10", {}).get("timeline", {})
        self.lead_weeks = int(c.get("lead_weeks", 4))
        self.out_path = Path(c.get("out_path", "artifacts/evolution/funding_timeline.json"))

        self.roadmap = roadmap_generator or RoadmapGenerator(
            self.config, self.logger, cf_pipeline=cf_pipeline)

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any],
                 analysis: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Build the pre-launch preparation timeline for one campaign."""
        rm = self.roadmap.generate(raw, analysis=analysis)
        steps = rm["steps"]
        n = len(steps)

        checkpoints = []
        for i, st in enumerate(steps, start=1):
            w = self._week_of(i, n)
            checkpoints.append({
                "checkpoint": i,
                "when": f"Week -{w}",
                "weeks_before_launch": w,
                "task": st["action"],
                "cumulative_probability": st["cumulative_probability"],
                "marginal_gain": st["marginal_gain"],
                "readiness_after": st["stage_after"],
                "crosses_bar": st["crosses_bar"],
            })
        # terminal launch checkpoint
        checkpoints.append({
            "checkpoint": n + 1,
            "when": "Launch day",
            "weeks_before_launch": 0,
            "task": "Launch the campaign",
            "cumulative_probability": rm["target_probability"],
            "marginal_gain": None,
            "readiness_after": rm["target_stage"],
            "crosses_bar": rm["target_probability"] >= self.roadmap.stage.cf.simulator.engine.threshold,
        })

        return {
            "campaign": rm["campaign"],
            "current_stage": rm["current_stage"],
            "current_probability": rm["current_probability"],
            "lead_weeks": self.lead_weeks,
            "checkpoints": checkpoints,
            "target_stage": rm["target_stage"],
            "target_probability": rm["target_probability"],
            "summary": self._summary(rm, checkpoints),
            "deferred": _DEFERRED,
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Timelines for a spread of improvable test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.generate(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved funding timelines -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _week_of(self, i: int, n: int) -> int:
        """Weeks-before-launch for step *i* of *n* — foundational steps earliest, monotone down."""
        if n <= 0:
            return self.lead_weeks
        return max(1, round(self.lead_weeks * (n - i + 1) / n))

    @staticmethod
    def _summary(rm: dict[str, Any], checkpoints: list[dict[str, Any]]) -> str:
        prep = checkpoints[:-1]                               # exclude the launch checkpoint
        if not prep:
            return (f"Currently {rm['current_stage']} ({rm['current_probability']:.0%}). No "
                    f"preparation levers materially help — launch as-is or reconsider scope.")
        cross = next((c for c in prep if c["crosses_bar"]), None)
        cross_txt = (f"You clear the funding bar by {cross['when']}." if cross
                     else "The plan does not clear the funding bar; consider rescoping.")
        return (f"Currently {rm['current_stage']} ({rm['current_probability']:.0%}). A "
                f"{len(prep)}-checkpoint pre-launch timeline — starting with "
                f"'{prep[0]['task'].lower()}' by {prep[0]['when']} — reaches {rm['target_stage']} "
                f"(~{rm['target_probability']:.0%}) by launch. {cross_txt}")
