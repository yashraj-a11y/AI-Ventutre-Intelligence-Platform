"""Phase I — Evolution Planner Pipeline (Day-10, Layer 9).

One entry point for the whole evolution-planner layer: funding-readiness **stage** (Phase B),
pre-launch preparation **roadmap** (Phase C), ranked **milestones** (Phase D), and the
preparation **timeline** (Phase H) — assembled into a single report.

**The point of this pipeline is to pay for the model once.** Every evolution module accepts a
shared ``analysis=`` (the Day-8 counterfactual analysis) — the one model-heavy call. The pipeline
runs ``cf.analyze(raw)`` a single time and threads that analysis into all four modules, which then
do only cheap arithmetic on top. All four share one warm :class:`CounterfactualPipeline`, so the
stage a founder sees in the roadmap, the timeline and the milestones are the *same* numbers.

No faiss → no OpenMP.

Example
-------
>>> pipe = EvolutionPipeline()
>>> rep = pipe.run(raw_campaign)
>>> rep["stage"]["stage"], [s["when"] for s in rep["timeline"]["checkpoints"]]
('Fixable', ['Week -4', 'Week -2', 'Launch day'])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.evolution.milestone_engine import MilestoneEngine
from src.evolution.roadmap_generator import RoadmapGenerator
from src.evolution.stage_detector import StageDetector
from src.evolution.timeline_generator import TimelineGenerator
from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("A funding-readiness plan over controllable pre-launch features (goal / duration / "
            "launch timing / title) — model-simulated, not a causal growth plan. Product, team, "
            "traction and post-launch growth items are deferred (the dataset can't observe them).")


class EvolutionPipeline:
    """Assemble stage + roadmap + milestones + timeline on one warm stack, one ``analyze``.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    cf_pipeline:
        Optional pre-built Day-8 :class:`CounterfactualPipeline` (share the warm stack — e.g.
        with the scenario pipeline for a full Day-10 report).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.evolution.pipeline")

        c = self.config.get("day10", {}).get("evolution", {})
        self.out_path = Path(c.get("out_path", "artifacts/evolution/evolution_engine.json"))

        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)
        # share the ONE cf (and one StageDetector / RoadmapGenerator) across every module
        self.stage = StageDetector(self.config, self.logger, cf_pipeline=self.cf)
        self.roadmap = RoadmapGenerator(self.config, self.logger, cf_pipeline=self.cf,
                                        stage_detector=self.stage)
        self.milestones = MilestoneEngine(self.config, self.logger, cf_pipeline=self.cf)
        self.timeline = TimelineGenerator(self.config, self.logger, cf_pipeline=self.cf,
                                          roadmap_generator=self.roadmap)

    # ------------------------------------------------------------------ public
    def run(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Full evolution-planner report for one campaign (one model-heavy call)."""
        raw = dict(raw)
        analysis = self.cf.analyze(raw)                       # the ONE model-heavy call, shared
        stage = self.stage.detect(raw, analysis=analysis)
        roadmap = self.roadmap.generate(raw, analysis=analysis)
        milestones = self.milestones.generate(raw, analysis=analysis)
        timeline = self.timeline.generate(raw, analysis=analysis)

        self.logger.info("Evolution report: %s (%.0f%% -> %.0f%%).", stage["stage"],
                         stage["funding_probability"] * 100, roadmap["target_probability"] * 100)
        return {
            "campaign": stage["campaign"],
            "stage": stage,
            "roadmap": roadmap,
            "milestones": milestones,
            "timeline": timeline,
            "deferred": {**stage["deferred_stages"], **roadmap["deferred"],
                         **milestones["deferred_milestones"], **timeline["deferred"]},
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Evolution reports for a spread of improvable test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.run(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved evolution reports -> %s", self.out_path)
        return self.out_path
