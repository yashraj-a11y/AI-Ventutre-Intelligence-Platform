"""Phase D — SWOT Analysis Engine (Day-9, Layer 8).

A real-signal SWOT (supersedes the Day-5 SWOT-lite prototype). Every quadrant is a reframing
of an existing layer's output:

- **Strengths** — the campaign's top **positive SHAP** contributions (real features helping the
  probability), humanised via the Day-5 label map.
- **Weaknesses** — the top **negative SHAP** contributions (real features dragging it down).
- **Opportunities** — the **Day-8 counterfactual levers** (controllable changes that would raise
  the odds).
- **Threats** — **category / timing / cohort headwinds** + below-threshold odds (forces working
  against it), with real evidence.

**Deferred** (no data): market / competitor threats — explicit placeholders, never fabricated
("Microsoft" / "open-source competitors" don't exist here).

Composes the Day-5 :class:`LocalExplainer` (SHAP) + Day-8 :class:`CounterfactualPipeline`,
**sharing one warm prediction engine**. No faiss → no OpenMP.

Example
-------
>>> eng = SWOTAnalysisEngine()
>>> swot = eng.analyze(raw_campaign)
>>> swot["strengths"][0]["point"], swot["threats"][0]["point"]
('A modest funding goal', 'Category headwind: Technology funds below baseline')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.copilot.objection_generator import campaign_signals
from src.counterfactual.explanation_generator import action_phrase
from src.counterfactual.pipeline import CounterfactualPipeline
from src.explainability.explanation_generator import humanize
from src.explainability.local_explainer import LocalExplainer
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED_THREATS = {
    "market_threats": {"status": "deferred", "reason": "no market data in this dataset"},
    "competitor_threats": {"status": "deferred", "reason": "no named-competitor data"},
}
_CAPTION = ("SWOT reframed from model SHAP contributions, historical patterns, and "
            "counterfactuals — not a real investor's judgement. Market/competitor threats deferred.")


class SWOTAnalysisEngine:
    """Real-signal SWOT: SHAP strengths/weaknesses + counterfactual opportunities + headwind threats.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    cf_pipeline, local_explainer:
        Optional pre-built collaborators (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None,
                 local_explainer: Optional[LocalExplainer] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.copilot.swot")

        c = self.config.get("day9", {}).get("swot", {})
        self.top_k = int(c.get("top_k", 4))
        self.lift_concern = float(c.get("lift_concern", 0.9))
        self.out_path = Path(c.get("out_path", "artifacts/copilot/swot_report.json"))

        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)
        # Share the warm prediction engine between counterfactual + SHAP explainer.
        self.lx = local_explainer or LocalExplainer(self.config, self.logger,
                                                    engine=self.cf.simulator.engine)
        self.frame = self.cf.evidence.frame
        self.base = self.cf.evidence.base

    # ------------------------------------------------------------------ public
    def analyze(self, raw: dict[str, Any],
                analysis: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Full SWOT for one campaign."""
        raw = dict(raw)
        exp = self.lx.explain_raw(pd.DataFrame([raw]))[0]     # Day-5 SHAP
        analysis = analysis if analysis is not None else self.cf.analyze(raw)  # Day-8 levers
        s = campaign_signals(self.frame, self.base, raw)

        return {
            "campaign": {k: raw.get(k) for k in ("name", "main_category", "usd_goal_real")},
            "funding_probability": exp.funding_probability,
            "strengths": [self._contrib(c) for c in exp.top_positive[: self.top_k]],
            "weaknesses": [self._contrib(c) for c in exp.top_negative[: self.top_k]],
            "opportunities": [self._lever(r) for r in
                              analysis["ranked_recommendations"][: self.top_k]],
            "threats": self._threats(s, analysis["baseline_p"]),
            "deferred_threats": _DEFERRED_THREATS,
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 3) -> Path:
        """SWOT reports for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.analyze(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved SWOT reports -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _contrib(self, c) -> dict[str, Any]:
        phrase = humanize(c)
        return {"point": phrase[0].upper() + phrase[1:],
                "shap_contribution": round(float(c.contribution), 3), "source": "Day-5 SHAP"}

    def _lever(self, rec) -> dict[str, Any]:
        return {"point": action_phrase(rec["feature"], rec.get("current"), rec.get("proposed")),
                "estimated_gain_points": round(float(rec.get("delta", 0)) * 100),
                "source": "Day-8 counterfactual"}

    def _threats(self, s, base_p) -> list[dict[str, Any]]:
        threats = []
        if s["cat_lift"] < self.lift_concern:
            threats.append({"point": f"Category headwind: {s['main']} funds below baseline",
                            "evidence": f"{s['cat_rate']:.0%} funded, lift {s['cat_lift']:.2f} "
                                        f"(n={s['cat_n']:,})", "source": "Day-7 patterns"})
        if s["cohort_n"] >= 100 and s["cohort_lift"] < self.lift_concern:
            threats.append({"point": "Weak cohort: similar campaigns mostly fail",
                            "evidence": f"{s['cohort_rate']:.0%} of {s['main']} + {s['goal_band']} "
                                        f"campaigns funded (n={s['cohort_n']:,})", "source": "Day-6/7 cohort"})
        if s["dur"] is not None and s["dur_lift"] < self.lift_concern:
            threats.append({"point": "Timing headwind: this campaign length underperforms",
                            "evidence": f"{s['dur_band']} funds at {s['dur_rate']:.0%} "
                                        f"(lift {s['dur_lift']:.2f})", "source": "Day-7 patterns"})
        threats.sort(key=lambda t: 0)   # keep insertion order (category first)
        return threats
