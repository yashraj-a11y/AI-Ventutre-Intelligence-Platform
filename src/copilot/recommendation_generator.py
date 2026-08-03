"""Phase G — Strategic Recommendation Engine (Day-9, Layer 8).

Turn the analysis into a prioritized action plan a founder can execute — **reusing the Day-8
counterfactual levers** as the tactical core (no duplication) and adding **strategic framing**:

- **Tactical actions** — the Day-8 optimization plan steps (lower goal, shorten campaign …),
  each with its estimated impact, evidence, and confidence.
- **Strategic guidance** — honest, higher-level advice:
  - **delay / rescope** if even the optimized probability stays below the funding bar (a weak
    campaign can't be tweaked into a strong one — say so);
  - **timing headwind** if the category's funding rates have declined (Day-9 Phase E);
  - **prioritize** if the plan can lift the campaign above the threshold.

**Real controllable actions only** — never "acquire an advisor" / "expand the team" (no data).
Composes the Day-8 pipeline + Day-9 category engine (shared item frame). No faiss → no OpenMP.

Example
-------
>>> eng = RecommendationEngine()
>>> rec = eng.generate(raw_campaign)
>>> rec["recommendations"][0]["type"], rec["overall_guidance"][:20]
('action', 'Prioritize: shorten ')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.copilot.market_analysis import CategoryAnalysisEngine
from src.counterfactual.pipeline import CounterfactualPipeline
from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("Strategic guidance from model simulations + historical patterns over controllable "
            "features — an estimate, not a guarantee. Team/advisor/product actions are out of scope "
            "(no data).")


class RecommendationEngine:
    """Prioritized tactical actions (Day-8 levers) + honest strategic guidance.

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
        self.logger = logger or get_logger("avip.copilot.recommend")

        c = self.config.get("day9", {}).get("recommendations", {})
        self.out_path = Path(c.get("out_path", "artifacts/copilot/recommendations.json"))

        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)
        self.category = CategoryAnalysisEngine(self.config, self.logger,
                                               frame=self.cf.evidence.frame, base=self.cf.evidence.base)

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any],
                 analysis: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Prioritized recommendations for one campaign."""
        raw = dict(raw)
        analysis = analysis if analysis is not None else self.cf.analyze(raw)
        threshold = float(self.cf.simulator.engine.threshold)
        base_p = analysis["baseline_p"]
        plan = analysis["optimization_plan"]
        final_p = plan.get("final_p", base_p)
        cat = self.category.analyze(raw)

        recs: list[dict[str, Any]] = []
        for step in plan.get("steps", []):                       # tactical (Day-8 levers)
            recs.append({
                "type": "action",
                "recommendation": step.get("action"),
                "rationale": step.get("evidence"),
                "estimated_impact": f"{round(step.get('marginal_gain', 0)*100):+d} points",
                "confidence": step.get("confidence"),
            })
        recs += self._strategic(base_p, final_p, threshold, cat)  # strategic framing
        for i, r in enumerate(recs, start=1):
            r["priority"] = i

        return {
            "campaign": {k: raw.get(k) for k in ("name", "main_category", "usd_goal_real")},
            "current_probability": round(base_p, 4),
            "optimized_probability": round(final_p, 4),
            "threshold": round(threshold, 4),
            "recommendations": recs,
            "overall_guidance": self._headline(base_p, final_p, threshold, plan, cat),
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 3) -> Path:
        """Recommendations for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.generate(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved recommendations -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _strategic(self, base_p, final_p, threshold, cat) -> list[dict[str, Any]]:
        out = []
        if final_p < threshold:
            out.append({"type": "strategic",
                "recommendation": (f"Even with the changes above, the model estimates only "
                                   f"~{final_p:.0%} — below the {threshold:.0%} funding bar. "
                                   f"Consider rescoping (a smaller ask), strengthening category "
                                   f"fit, or delaying until the fundamentals improve."),
                "rationale": f"Optimized probability {final_p:.0%} < {threshold:.0%} threshold.",
                "estimated_impact": None, "confidence": "high"})
        else:
            out.append({"type": "strategic",
                "recommendation": (f"The recommended changes could lift you to ~{final_p:.0%}, "
                                   f"above the {threshold:.0%} bar — prioritize them before launch."),
                "rationale": f"Optimized probability {final_p:.0%} ≥ {threshold:.0%} threshold.",
                "estimated_impact": None, "confidence": "medium"})
        if cat.get("year_trend", {}).get("direction") == "declining":
            out.append({"type": "strategic",
                "recommendation": (f"Timing headwind: funding rates in {cat['category']} have "
                                   f"declined over time — factor this into your launch decision."),
                "rationale": f"{cat['category']} year-trend: declining (market timing "
                             f"{cat.get('market_timing', 'unfavorable').lower()}).",
                "estimated_impact": None, "confidence": "medium"})
        return out

    @staticmethod
    def _headline(base_p, final_p, threshold, plan, cat) -> str:
        steps = plan.get("steps", [])
        if not steps:
            lead = "No high-impact controllable changes were found"
        else:
            lead = f"Prioritize: {steps[0].get('action', 'the top change').lower()}"
        verdict = (f"the plan reaches ~{final_p:.0%}, {'above' if final_p >= threshold else 'still below'} "
                   f"the {threshold:.0%} bar")
        return f"{lead}. Applying all recommended changes, {verdict} (from {base_p:.0%})."
