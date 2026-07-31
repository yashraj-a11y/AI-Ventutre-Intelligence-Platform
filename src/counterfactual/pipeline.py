"""Phase G — Counterfactual Pipeline (Day-8, Layer 7 entry point).

One object that turns a **raw campaign** into a ranked, evidence-backed improvement plan:

    raw campaign
      → baseline prediction (Day-4)
      → generate candidates (mutable features, Day-7-seeded)
      → simulate Δp (re-run the real model)
      → validate with real historical evidence (Day-7 substrate)
      → rank (gain × evidence × feasibility)
      → explain (founder-readable, honesty-captioned)
      → greedy optimization plan (cumulative, re-simulated jointly)

Two outputs per campaign: the **ranked recommendations** (every option, analysed
individually) and the **optimization plan** (a non-conflicting, one-change-per-feature
sequence whose cumulative probability is re-simulated, not summed).

One warm ``PredictionEngine`` and one shared Day-7 item frame are built once and reused.
No faiss/torch (evidence is pandas over the item frame) → no OpenMP concern.

Example
-------
>>> pipe = CounterfactualPipeline()
>>> report = pipe.analyze(raw_campaign)
>>> report["baseline_p"], report["optimization_plan"]["final_p"]
(0.035, 0.375)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.candidate_generator import CandidateGenerator
from src.counterfactual.evidence_validator import EvidenceValidator
from src.counterfactual.explanation_generator import (
    CounterfactualExplanationGenerator, action_phrase)
from src.counterfactual.probability_simulator import ProbabilitySimulator
from src.counterfactual.ranking_engine import RankingEngine
from src.models.prediction_engine import PredictionEngine
from src.utils.config import load_config
from src.utils.logger import get_logger


class CounterfactualPipeline:
    """Raw campaign → ranked recommendations + optimization plan (one warm stack).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    engine:
        Optional pre-built :class:`PredictionEngine` (share one warm calibrated model).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 engine: Optional[PredictionEngine] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.cf.pipeline")

        c = self.config.get("day8", {}).get("pipeline", {})
        self.n_examples = int(c.get("n_examples", 3))
        self.recommendations_path = Path(c.get("recommendations_path",
                                               "artifacts/counterfactual/recommendations.json"))
        self.scenario_results_path = Path(c.get("scenario_results_path",
                                                "artifacts/counterfactual/scenario_results.json"))
        self.ranked_actions_path = Path(c.get("ranked_actions_path",
                                              "artifacts/counterfactual/ranked_actions.csv"))

        # One warm calibrated model + one shared Day-7 item frame across all phases.
        self.generator = CandidateGenerator(self.config, self.logger)
        self.simulator = ProbabilitySimulator(self.config, self.logger, engine=engine)
        self.evidence = EvidenceValidator(self.config, self.logger)   # builds item frame once
        self.ranking = RankingEngine(self.config, self.logger)
        self.explainer = CounterfactualExplanationGenerator(self.config, self.logger)
        self.logger.info("CounterfactualPipeline ready (warm model + item frame).")

    # ------------------------------------------------------------------ public
    def analyze(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Full counterfactual analysis for one raw campaign."""
        raw = dict(raw)
        baseline_p = self.simulator.baseline(raw)
        candidates = self.generator.generate(raw)
        if not candidates:
            return {"baseline_p": round(baseline_p, 4), "ranked_recommendations": [],
                    "optimization_plan": {"baseline_p": round(baseline_p, 4),
                                          "final_p": round(baseline_p, 4), "steps": []},
                    "note": "no feasible mutable-feature candidates."}

        sims = self.simulator.simulate_all(raw, candidates)        # individual Δp
        validated = self.evidence.validate_all(sims, raw)          # + real evidence
        ranked = self.ranking.rank(validated)                     # scored + prioritised
        explained = self.explainer.explain_all(ranked)            # + readable sentence
        plan = self.simulator.greedy_plan(raw, candidates)        # honest cumulative plan
        plan["steps"] = self._enrich_plan(plan["steps"], explained)

        return {
            "query": {k: raw.get(k) for k in ("name", "main_category", "category",
                                              "country", "usd_goal_real")},
            "baseline_p": round(baseline_p, 4),
            "ranked_recommendations": explained,
            "optimization_plan": plan,
            "meta": {"n_candidates": len(candidates), "n_recommendations": len(explained),
                     "note": ("Model-simulated what-ifs over mutable features; cumulative "
                              "re-simulated jointly. Historical association, not a guarantee.")},
        }

    def generate_examples(self, n: Optional[int] = None, random_state: int = 4) -> dict[str, Path]:
        """Analyse a spread of improvable TEST campaigns; save the Day-8 artifacts."""
        import numpy as np

        n = n or self.n_examples
        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        pool = df.iloc[ti]
        pool = pool[pool["usd_goal_real"] > 12000]                 # campaigns with room to improve
        sample = pool.sample(n, random_state=random_state).reset_index(drop=True)
        reports = [self.analyze(r.to_dict()) for _, r in sample.iterrows()]

        self.recommendations_path.parent.mkdir(parents=True, exist_ok=True)
        self.recommendations_path.write_text(json.dumps(
            [{"query": r["query"], "baseline_p": r["baseline_p"],
              "ranked_recommendations": r["ranked_recommendations"]} for r in reports], indent=2))
        self.scenario_results_path.write_text(json.dumps(
            [{"query": r["query"], "optimization_plan": r["optimization_plan"]} for r in reports],
            indent=2))
        # flat ranked-actions CSV (gitignored via *.csv; regenerable)
        rows = [{"campaign": r["query"]["name"], "priority": a["priority"], "feature": a["feature"],
                 "current": a["current"], "proposed": a["proposed"], "delta": a["delta"],
                 "confidence": a["confidence"], "score": a["score"]}
                for r in reports for a in r["ranked_recommendations"]]
        pd.DataFrame(rows).to_csv(self.ranked_actions_path, index=False)

        self.logger.info("Saved %d counterfactual reports -> %s / %s / %s", len(reports),
                         self.recommendations_path, self.scenario_results_path,
                         self.ranked_actions_path)
        return {"recommendations": self.recommendations_path,
                "scenario_results": self.scenario_results_path,
                "ranked_actions": self.ranked_actions_path}

    # --------------------------------------------------------------- internals
    def _enrich_plan(self, steps: list[dict[str, Any]],
                     explained: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach a readable action + evidence to each greedy-plan step."""
        by_key = {(e["feature"], str(e["proposed"])): e for e in explained}
        out = []
        for st in steps:
            e = by_key.get((st["feature"], str(st["proposed"])), {})
            out.append({**st,
                        "action": action_phrase(st["feature"], st["current"], st["proposed"]),
                        "confidence": e.get("confidence"),
                        "evidence": (e.get("evidence") or {}).get("statement")})
        return out
