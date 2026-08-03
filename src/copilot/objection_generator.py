"""Phase B — Investor Objection Generator (Day-9, Layer 8).

Generate the concerns a data-driven VC would raise — from **structured, assessable
dimensions**, not noisy raw SHAP tokens. Each objection is fired only when a real signal
supports it, with **severity grounded in the magnitude** and a **suggested response drawn
from a Day-8 counterfactual lever**.

Dimensions (real signals only):
- **funding_odds** — calibrated probability below the decision threshold (Day-4).
- **goal_size** — goal high vs the category's funded-median (Day-7 substrate).
- **category_fit** — the campaign's category funds below baseline (lift < 1).
- **timing** — the campaign's duration band underperforms.
- **cohort** — campaigns most like it (category + goal band) mostly failed.

No named-competitor / market-size objections — those aren't in the data (deferred). Every
objection carries its source layer + evidence, so it's auditable, not asserted.

Reuses the Day-8 :class:`CounterfactualPipeline` (prediction + evidence item frame +
counterfactual recommendations). No faiss → no OpenMP concern.

Example
-------
>>> gen = ObjectionGenerator()
>>> objs = gen.generate(raw_campaign)
>>> objs[0]["dimension"], objs[0]["severity"]
('goal_size', 'High')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.evidence_validator import duration_band
from src.counterfactual.explanation_generator import action_phrase
from src.counterfactual.pipeline import CounterfactualPipeline
from src.knowledge_base.venture_database import goal_band
from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("Model predictions + historical associations reframed as investor-style concerns "
            "— not a real investor's judgement.")


def campaign_signals(frame: pd.DataFrame, base: float, raw: dict[str, Any]) -> dict[str, Any]:
    """Category / cohort / timing / goal statistics for a campaign (shared by objections + risk).

    Computed over the Day-7 item frame; all real, all train-derived.
    """
    main = str(raw.get("main_category"))
    goal = float(raw.get("usd_goal_real", 0.0))
    launched = pd.to_datetime(raw.get("launched"), errors="coerce")
    deadline = pd.to_datetime(raw.get("deadline"), errors="coerce")
    dur = int((deadline - launched).days) if not (pd.isna(launched) or pd.isna(deadline)) else None

    cat = frame[frame["main_category"] == main]
    gb = goal_band(goal)
    cohort = cat[cat["goal_band"] == gb]
    db = duration_band(dur) if dur is not None else None
    dursub = cat[cat["duration_band"] == db] if db else cat.iloc[:0]
    if db and len(dursub) < 100:
        dursub = frame[frame["duration_band"] == db]

    def rate(sub):
        return float(sub["target"].mean()) if len(sub) else base
    return {
        "main": main, "goal": goal, "dur": dur, "goal_band": gb, "dur_band": db,
        "cat_rate": rate(cat), "cat_lift": rate(cat) / base, "cat_n": len(cat),
        "funded_med": float(cat.loc[cat.target == 1, "usd_goal_real"].median()) if len(cat) else None,
        "cohort_rate": rate(cohort), "cohort_lift": rate(cohort) / base, "cohort_n": len(cohort),
        "dur_rate": rate(dursub), "dur_lift": rate(dursub) / base, "dur_n": len(dursub),
    }


class ObjectionGenerator:
    """Generate ranked, evidence-backed investor objections from real signals.

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
        self.logger = logger or get_logger("avip.copilot.objections")

        c = self.config.get("day9", {}).get("objections", {})
        self.sev_high = float(c.get("severity_high", 0.40))
        self.sev_med = float(c.get("severity_medium", 0.20))
        self.goal_high_ratio = float(c.get("goal_high_ratio", 1.3))
        self.lift_concern = float(c.get("lift_concern", 0.9))
        self.min_cohort_n = int(c.get("min_cohort_n", 100))
        self.out_path = Path(c.get("out_path", "artifacts/copilot/investor_questions.json"))

        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)
        self.frame = self.cf.evidence.frame          # Day-7 item frame (shared)
        self.base = self.cf.evidence.base

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any],
                 analysis: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Ranked investor objections for one campaign (highest severity first)."""
        raw = dict(raw)
        analysis = analysis if analysis is not None else self.cf.analyze(raw)
        base_p = analysis["baseline_p"]
        threshold = float(self.cf.simulator.engine.threshold)
        recs = analysis["ranked_recommendations"]
        s = self._signals(raw)

        objs = []
        objs += self._funding_odds(base_p, threshold, recs)
        objs += self._goal_size(raw, s, recs)
        objs += self._category_fit(s, recs)
        objs += self._timing(s, recs)
        objs += self._cohort(s, recs)

        objs.sort(key=lambda o: o["concern_score"], reverse=True)
        self.logger.info("Generated %d investor objections (top: %s).",
                         len(objs), objs[0]["dimension"] if objs else "none")
        return objs

    def generate_examples(self, n: int = 3, random_state: int = 3) -> Path:
        """Objections for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [{"campaign": {k: r[k] for k in ("name", "main_category", "usd_goal_real")},
                "objections": self.generate(r.to_dict())} for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved investor objections -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _signals(self, raw) -> dict[str, Any]:
        return campaign_signals(self.frame, self.base, raw)

    def _funding_odds(self, base_p, threshold, recs):
        if base_p >= threshold:
            return []
        score = min(1.0, (threshold - base_p) / max(threshold, 1e-9))
        return [self._obj("funding_odds",
            f"The model estimates only a {base_p:.0%} funding probability — below the "
            f"~{threshold:.0%} decision threshold. How will you beat these odds?",
            score, f"Calibrated funding probability {base_p:.0%} vs {threshold:.0%} bar.",
            self._response(recs, None), "Day-4 prediction",
            {"probability": round(base_p, 4), "threshold": round(threshold, 4)})]

    def _goal_size(self, raw, s, recs):
        goal, fm = s["goal"], s["funded_med"]
        if not fm or goal <= fm * self.goal_high_ratio:
            return []
        score = min(1.0, (goal / fm - 1) / 2)
        return [self._obj("goal_size",
            f"Your ${goal:,.0f} goal is high for {s['main']}, where funded campaigns typically "
            f"ask a median of ${fm:,.0f}. Why this ask — and can you hit it?",
            score, f"Goal ${goal:,.0f} vs {s['main']} funded-median ${fm:,.0f} "
                   f"({goal/fm:.1f}x).",
            self._response(recs, "goal"), "Day-5 SHAP + Day-7 patterns",
            {"goal": goal, "funded_median": fm})]

    def _category_fit(self, s, recs):
        if s["cat_lift"] >= self.lift_concern:
            return []
        score = min(1.0, 1 - s["cat_lift"])
        return [self._obj("category_fit",
            f"{s['main']} campaigns historically fund at {s['cat_rate']:.0%} (lift "
            f"{s['cat_lift']:.2f}) — below the {self.base:.0%} baseline. How do you overcome "
            f"the category headwind?",
            score, f"{s['main']} funding rate {s['cat_rate']:.0%} vs {self.base:.0%} baseline "
                   f"(n={s['cat_n']:,}).",
            self._response(recs, None), "Day-7 patterns",
            {"category_rate": round(s["cat_rate"], 4), "lift": round(s["cat_lift"], 3), "n": s["cat_n"]})]

    def _timing(self, s, recs):
        if s["dur"] is None or s["dur_lift"] >= self.lift_concern:
            return []
        score = min(1.0, 1 - s["dur_lift"])
        return [self._obj("timing",
            f"Your {s['dur']}-day campaign sits in a band that funds at {s['dur_rate']:.0%} "
            f"(lift {s['dur_lift']:.2f}). Would a shorter campaign convert better?",
            score, f"Duration '{s['dur_band']}' funds at {s['dur_rate']:.0%} vs {self.base:.0%} "
                   f"baseline (n={s['dur_n']:,}).",
            self._response(recs, "campaign_duration"), "Day-7 patterns",
            {"duration_days": s["dur"], "lift": round(s["dur_lift"], 3)})]

    def _cohort(self, s, recs):
        if s["cohort_n"] < self.min_cohort_n or s["cohort_lift"] >= self.lift_concern:
            return []
        score = min(1.0, 1 - s["cohort_lift"])
        return [self._obj("cohort",
            f"Among campaigns most like yours ({s['main']}, {s['goal_band']}), only "
            f"{s['cohort_rate']:.0%} funded. What sets you apart from the {s['cohort_n']:,} "
            f"that came before?",
            score, f"Cohort ({s['main']} + {s['goal_band']}) funding rate {s['cohort_rate']:.0%} "
                   f"vs {self.base:.0%} (n={s['cohort_n']:,}).",
            self._response(recs, "goal"), "Day-6/7 cohort",
            {"cohort_rate": round(s["cohort_rate"], 4), "lift": round(s["cohort_lift"], 3), "n": s["cohort_n"]})]

    def _obj(self, dimension, concern, score, reason, response, source, evidence):
        return {"dimension": dimension, "concern": concern,
                "severity": self._severity(score), "concern_score": round(score, 3),
                "reason": reason, "suggested_response": response,
                "source_layer": source, "evidence": evidence, "honesty_note": _CAPTION}

    def _severity(self, score: float) -> str:
        return ("High" if score >= self.sev_high else
                "Medium" if score >= self.sev_med else "Low")

    @staticmethod
    def _response(recs: list[dict[str, Any]], feature: Optional[str]) -> Optional[str]:
        """A suggested response drawn from a matching Day-8 counterfactual lever."""
        if not recs:
            return None
        rec = next((r for r in recs if r["feature"] == feature), recs[0]) if feature else recs[0]
        act = action_phrase(rec["feature"], rec.get("current"), rec.get("proposed"))
        return (f"{act} — the model estimates about {round(rec.get('delta', 0)*100):+d} points "
                f"({rec.get('new_p', 0):.0%}).")
