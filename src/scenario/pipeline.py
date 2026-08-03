"""Phase I — Scenario Simulator Pipeline (Day-10, Layer 10).

One entry point for the whole scenario-simulator layer: **generate** the multi-variable
what-if scenarios (Phase E), **simulate** each through the real model (Phase F), and **compare**
them — rank, marginal value, best + most-efficient (Phase G).

The comparison already composes the simulator, which composes the generator, so this pipeline is
a thin single-entry wrapper that shares **one** :class:`ScenarioSimulator` (and hence one warm
:class:`CounterfactualPipeline`) across the layer — the model is loaded once, and each scenario
is still a fresh model call (never a sum). It can share the *same* ``cf_pipeline`` as the
:class:`~src.evolution.pipeline.EvolutionPipeline` so a full Day-10 report runs on one stack.

No faiss → no OpenMP.

Example
-------
>>> pipe = ScenarioPipeline()
>>> rep = pipe.run(raw_campaign)
>>> rep["best_scenario"]["name"], rep["most_efficient_scenario"]["n_changes"]
('Lower goal + shorten duration + Tuesday launch + add title number', 1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.scenario.comparison import ScenarioComparison
from src.scenario.simulator import ScenarioSimulator
from src.utils.config import load_config
from src.utils.logger import get_logger


class ScenarioPipeline:
    """Assemble generator + simulator + comparison on one warm stack.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    cf_pipeline:
        Optional pre-built Day-8 :class:`CounterfactualPipeline` (share the warm stack — e.g.
        with the evolution pipeline for a full Day-10 report).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.scenario.pipeline")

        c = self.config.get("day10", {}).get("scenario_report", {})
        self.out_path = Path(c.get("out_path", "artifacts/scenario/scenario_report.json"))

        # one simulator (=> one generator + one cf), shared into the comparison
        self.simulator = ScenarioSimulator(self.config, self.logger, cf_pipeline=cf_pipeline)
        self.comparison = ScenarioComparison(self.config, self.logger, simulator=self.simulator)

    # ------------------------------------------------------------------ public
    def run(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Full scenario report for one campaign (rank + marginal value + recommendation)."""
        report = self.comparison.compare(raw)                 # simulates once via the shared simulator
        self.logger.info("Scenario report: best %s (%.0f%%), efficient %s.",
                         report["best_scenario"]["name"], report["best_scenario"]["probability"] * 100,
                         (report["most_efficient_scenario"] or {}).get("name", "none"))
        return report

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Scenario reports for a spread of improvable test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.run(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved scenario reports -> %s", self.out_path)
        return self.out_path
