"""Phase F — Scenario Simulation Engine (Day-10, Layer 10).

Run each generated scenario through the **real Day-4 calibrated model** and attach a funding
probability, the gain over the current campaign, whether it crosses the funding bar, and a
readiness stage.

    Current (as-is)                          3%
    Lower goal                               8%   (+5)
    Lower goal + shorten duration            38%  (+34)  READY ✔
    Lower goal + shorten duration + Tuesday  38%  (+35)  READY ✔

Reuses the Day-8 :class:`~src.counterfactual.probability_simulator.ProbabilitySimulator` as the
scoring oracle (inherits its calibration + train/serve consistency) and the Phase-E
:class:`ScenarioGenerator` (canonical composition). Each scenario is a fresh model call — never
a sum of gains. No faiss → no OpenMP.

Example
-------
>>> sim = ScenarioSimulator()
>>> res = sim.simulate(raw_campaign)
>>> res["results"][2]["probability"], res["results"][2]["crosses_bar"]
(0.375, True)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.evolution.stage_detector import StageDetector
from src.scenario.scenario_generator import ScenarioGenerator
from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("Each scenario's probability is a fresh call to the real calibrated model on the "
            "canonically-composed campaign — model-simulated over controllable features, not a "
            "causal guarantee, and never a sum of individual gains.")


class ScenarioSimulator:
    """Score every scenario through the real Day-4 model + readiness verdict.

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
        self.logger = logger or get_logger("avip.scenario.simulator")

        c = self.config.get("day10", {}).get("simulate", {})
        self.out_path = Path(c.get("out_path", "artifacts/scenario/simulation_results.json"))

        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)
        self.sim = self.cf.simulator                       # Day-8 ProbabilitySimulator (scoring oracle)
        self.generator = ScenarioGenerator(self.config, self.logger,
                                           candidate_generator=self.cf.generator)
        self.stage = StageDetector(self.config, self.logger, cf_pipeline=self.cf)

    # ------------------------------------------------------------------ public
    def simulate(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Generate + score every scenario for one campaign."""
        raw = dict(raw)
        scen = self.generator.generate(raw)
        threshold = float(self.sim.engine.threshold)

        scored = []
        for s in scen["scenarios"]:
            p = self.sim.baseline(s["campaign"])            # real model, fresh call per scenario
            scored.append({"name": s["name"], "changes": s["changes"],
                           "description": s["description"], "probability": round(p, 4),
                           "crosses_bar": p >= threshold})

        current_p = scored[0]["probability"]                # 'Current (as-is)' is first
        best_p = max(r["probability"] for r in scored)
        for r in scored:
            r["gain"] = round(r["probability"] - current_p, 4)
            r["stage"] = self.stage.stage_of(r["probability"], best_p)

        self.logger.info("Simulated %d scenarios (current %.3f -> best %.3f).",
                         len(scored), current_p, best_p)
        return {
            "campaign": scen["campaign"],
            "current_probability": current_p,
            "best_probability": best_p,
            "threshold": round(threshold, 4),
            "results": scored,
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Scenario simulations for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.simulate(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved simulation results -> %s", self.out_path)
        return self.out_path
