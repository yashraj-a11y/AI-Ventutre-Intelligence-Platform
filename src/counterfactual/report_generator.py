"""Phase H — Startup Optimization Report (Day-8, Layer 7).

Assemble a counterfactual analysis into a founder-facing **optimization report**: the current
funding probability, the ranked improvements (each with its evidence and confidence), the
honest cumulative optimized probability, the resulting **optimized profile** (real mutable
features only), and an overall recommendation.

The optimized profile applies the greedy plan's changes to the raw campaign and reports the
final *controllable* values (goal, duration, launch weekday) + the model-estimated
probability — never a fabricated founder/market/revenue profile.

Example
-------
>>> rg = ReportGenerator()
>>> report = rg.generate(raw_campaign)
>>> print(rg.render_text(report))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.utils.config import load_config
from src.utils.logger import get_logger

_HONESTY = ("Model-simulated what-ifs over features you control (goal, duration, launch "
            "timing, title), backed by historical patterns. Cumulative probability is "
            "re-simulated jointly. An estimate based on historical association, not a guarantee.")


class ReportGenerator:
    """Turn a counterfactual analysis into a structured + readable optimization report.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    pipeline:
        Optional pre-built :class:`CounterfactualPipeline` (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 pipeline: Optional[CounterfactualPipeline] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.cf.report")

        c = self.config.get("day8", {}).get("report", {})
        self.top_n = int(c.get("top_n", 5))
        self.n_examples = int(c.get("n_examples", 3))
        self.out_path = Path(c.get("out_path", "artifacts/counterfactual/optimization_report.json"))

        self.pipeline = pipeline or CounterfactualPipeline(self.config, self.logger)

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Analyse a campaign and build its optimization report."""
        return self.build_report(self.pipeline.analyze(raw), dict(raw))

    def build_report(self, analysis: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
        """Format an existing analysis into the optimization report structure."""
        plan = analysis.get("optimization_plan", {})
        base_p = analysis.get("baseline_p", plan.get("baseline_p"))
        final_p = plan.get("final_p", base_p)

        improvements = [{
            "priority": r["priority"], "action": r.get("explanation", "").split(".")[0],
            "feature": r["feature"], "current": r["current"], "proposed": r["proposed"],
            "estimated_new_probability": r.get("new_p"), "delta": r.get("delta"),
            "tier": r.get("tier"), "confidence": r.get("confidence"),
            "evidence": (r.get("evidence") or {}).get("statement"),
        } for r in analysis.get("ranked_recommendations", [])[: self.top_n]]

        report = {
            "campaign": analysis.get("query", {}),
            "current_funding_probability": base_p,
            "recommended_improvements": improvements,
            "optimization_plan": {
                "baseline_p": base_p, "final_p": final_p,
                "total_gain": plan.get("total_gain"),
                "steps": [{"action": s.get("action"), "cumulative_p": s.get("cumulative_p"),
                           "marginal_gain": s.get("marginal_gain"),
                           "confidence": s.get("confidence"), "evidence": s.get("evidence")}
                          for s in plan.get("steps", [])],
            },
            "optimized_profile": self._optimized_profile(raw, plan, final_p),
            "overall_recommendation": self._headline(base_p, final_p, plan),
            "honesty_note": _HONESTY,
        }
        return report

    def render_text(self, report: dict[str, Any]) -> str:
        """A readable, terminal-friendly rendering of the report."""
        c = report["campaign"]
        lines = [
            f"STARTUP OPTIMIZATION REPORT — {c.get('name')}",
            f"  {c.get('main_category')} | goal ${float(c.get('usd_goal_real', 0)):,.0f}",
            f"  Current funding probability: {report['current_funding_probability']:.0%}",
            "  " + "-" * 60,
            "  RECOMMENDED IMPROVEMENTS (ranked):",
        ]
        for imp in report["recommended_improvements"]:
            lines.append(f"   {imp['priority']}. [{imp['confidence']}] {imp['action']} "
                         f"(→ {imp['estimated_new_probability']:.0%}, {imp['delta']:+.0%})")
        plan = report["optimization_plan"]
        lines.append("  " + "-" * 60)
        lines.append(f"  OPTIMIZATION PATH (cumulative): {plan['baseline_p']:.0%}")
        for s in plan["steps"]:
            lines.append(f"   → {s['action']} ⇒ {s['cumulative_p']:.0%} ({s['marginal_gain']:+.0%})")
        lines.append(f"   FINAL: {plan['final_p']:.0%} (total {plan['total_gain']:+.0%})")
        lines.append("  " + "-" * 60)
        lines.append(f"  OVERALL: {report['overall_recommendation']}")
        lines.append(f"  Note: {report['honesty_note']}")
        return "\n".join(lines)

    def generate_examples(self, n: Optional[int] = None, random_state: int = 4) -> Path:
        """Build + save optimization reports for a spread of improvable test campaigns."""
        import numpy as np

        n = n or self.n_examples
        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        pool = df.iloc[ti]
        pool = pool[pool["usd_goal_real"] > 12000]
        sample = pool.sample(n, random_state=random_state).reset_index(drop=True)
        reports = [self.generate(r.to_dict()) for _, r in sample.iterrows()]

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(reports, indent=2))
        self.logger.info("Saved %d optimization reports -> %s", len(reports), self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _optimized_profile(self, raw: dict[str, Any], plan: dict[str, Any],
                           final_p: float) -> dict[str, Any]:
        """Apply the plan to the raw campaign; report final CONTROLLABLE values."""
        opt = dict(raw)
        for s in plan.get("steps", []):
            opt.update(s.get("raw_changes", {}))
        launched = pd.to_datetime(opt.get("launched"), errors="coerce")
        deadline = pd.to_datetime(opt.get("deadline"), errors="coerce")
        duration = int((deadline - launched).days) if not (pd.isna(launched) or pd.isna(deadline)) else None
        return {
            "goal_usd": round(float(opt.get("usd_goal_real", 0)), 2),
            "campaign_duration_days": duration,
            "launch_weekday": launched.day_name() if not pd.isna(launched) else None,
            "title": opt.get("name"),
            "estimated_funding_probability": final_p,
            "features_are_mutable_only": True,
        }

    @staticmethod
    def _headline(base_p: float, final_p: float, plan: dict[str, Any]) -> str:
        steps = plan.get("steps", [])
        if not steps:
            return ("No high-impact controllable changes were found — the campaign is already "
                    "reasonably optimized on the features you control.")
        first = steps[0].get("action", "the top change")
        gain_pts = round((final_p - base_p) * 100)
        return (f"Prioritize: {first.lower()}. Applying all {len(steps)} recommended change(s), "
                f"the model estimates your funding probability could rise from {base_p:.0%} to "
                f"{final_p:.0%} (+{gain_pts} points).")
