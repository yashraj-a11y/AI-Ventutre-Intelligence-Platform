"""Phase G — Scenario Comparison Engine (Day-10, Layer 10).

Turn the Phase-F scored scenario table into a **decision**: rank the scenarios by probability,
compute the **marginal value of each added change** (does this lever actually pull its weight?),
flag which scenarios cross the funding bar, and recommend two things a founder cares about — the
*highest-probability* strategy and the *fewest-changes* path to Ready.

    Ranked (best first)                       Marginal value of each added change
      Lower goal + shorten duration  38% ✔      Lower goal        +5   worth it
      + Tuesday launch               38% ✔      shorten duration  +30  worth it   ← the mover
      Lower goal                      8%         Tuesday launch    +0   negligible
      Current (as-is)                 3%         add title number  -2   skip

    → Fastest path to Ready: 'Lower goal + shorten duration' (2 changes, ~38%).

This is the genuinely new value over Day-8 (which returns a single greedy optimum): several
discrete strategies weighed side-by-side, with the *marginal* contribution of each step exposed
so a founder can drop the changes that don't help.

Composes the Phase-F :class:`ScenarioSimulator`. No faiss → no OpenMP.

Example
-------
>>> cmp = ScenarioComparison()
>>> rep = cmp.compare(raw_campaign)
>>> rep["best_scenario"]["name"], rep["most_efficient_scenario"]["n_changes"]
('Lower goal + shorten duration', 2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.scenario.scenario_generator import _LABEL
from src.scenario.simulator import ScenarioSimulator
from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("Scenarios ranked by the real calibrated model. 'Marginal value' is the probability "
            "gain of adding one change on top of the ones before it (cumulative order), so it "
            "depends on what precedes it — model-simulated over controllable features, not causal.")


class ScenarioComparison:
    """Rank scenarios, expose marginal value, and recommend best + most-efficient.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    cf_pipeline:
        Optional pre-built Day-8 :class:`CounterfactualPipeline` (share the warm stack).
    simulator:
        Optional pre-built Phase-F :class:`ScenarioSimulator` (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None,
                 simulator: Optional[ScenarioSimulator] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.scenario.comparison")

        c = self.config.get("day10", {}).get("comparison", {})
        self.worth_it_min = float(c.get("worth_it_min", 0.005))
        self.out_path = Path(c.get("out_path", "artifacts/scenario/comparison_report.json"))

        self.simulator = simulator or ScenarioSimulator(self.config, self.logger,
                                                        cf_pipeline=cf_pipeline)

    # ------------------------------------------------------------------ public
    def compare(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Rank + weigh scenarios for one campaign, and recommend."""
        sim = self.simulator.simulate(raw)
        results = sim["results"]                              # cumulative order (Current first)
        threshold = sim["threshold"]

        ranked = [dict(r, rank=i) for i, r in
                  enumerate(sorted(results, key=lambda r: r["probability"], reverse=True), start=1)]
        marginals = self._marginals(results)

        best = ranked[0]                                     # highest probability
        # fewest changes to Ready = first bar-crosser in cumulative order (changes grow by one).
        efficient = next((r for r in results if r["crosses_bar"]), None)

        return {
            "campaign": sim["campaign"],
            "current_probability": sim["current_probability"],
            "threshold": threshold,
            "ranked_scenarios": [{k: r[k] for k in
                                  ("rank", "name", "probability", "gain", "crosses_bar", "stage")}
                                 for r in ranked],
            "marginal_values": marginals,
            "best_scenario": {"name": best["name"], "probability": best["probability"],
                              "stage": best["stage"]},
            "most_efficient_scenario": (
                {"name": efficient["name"], "probability": efficient["probability"],
                 "n_changes": len(efficient["changes"])} if efficient else None),
            "recommendation": self._recommend(best, efficient, marginals, threshold),
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Comparison reports for a spread of improvable test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.compare(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved comparison reports -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _marginals(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Probability gain of each added change over the scenario before it (cumulative order)."""
        out = []
        for prev, cur in zip(results, results[1:]):
            added = [k for k in cur["changes"] if k not in prev["changes"]]
            key = added[0] if added else "?"
            gain = round(cur["probability"] - prev["probability"], 4)
            out.append({
                "change": key,
                "change_label": _LABEL.get(key, key),
                "marginal_gain": gain,
                "worth_it": gain >= self.worth_it_min,
            })
        return out

    def _recommend(self, best, efficient, marginals, threshold) -> str:
        if efficient is None:
            rec = (f"No scenario reaches the {threshold:.0%} funding bar; the strongest is "
                   f"'{best['name']}' (~{best['probability']:.0%}, {best['stage']}). "
                   f"Consider rescoping the goal or the project.")
        else:
            rec = (f"Fastest path to Ready: '{efficient['name']}' — "
                   f"{len(efficient['changes'])} change(s), ~{efficient['probability']:.0%}.")
            if best["name"] != efficient["name"] and best["probability"] > efficient["probability"]:
                rec += f" Highest probability overall: '{best['name']}' (~{best['probability']:.0%})."
        skip = [m["change_label"] for m in marginals if not m["worth_it"]]
        if skip:
            rec += f" Negligible or harmful, safe to skip: {', '.join(skip)}."
        return rec
