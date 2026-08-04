"""Phase F — Investor Strategy Engine (Day-11, Layer 11).

The **investor-facing view** of the same intelligence (the founder view is Phase E). It emits one
of four bands and a grounded reason:

    Strong Investment   — comfortably above the funding bar (Strong stage).
    Proceed with Caution — fundable as-is (Ready), or below-bar with strong reachable upside.
    Monitor Progress    — below the bar with a borderline path.
    Do Not Invest Yet   — below the bar even after the best controllable changes.

The band is a deterministic mapping from the **funding-readiness stage** (Day-10) + the reachable
headroom, optionally downgraded one notch when Phase-G confidence is low. The reason is grounded in
the calibrated probability, the reachable optimum, and the **top risk** (Day-9). Investor-side
diligence items (financials, cap table, team, market sizing, traction) are **deferred** — the
dataset can't support them.

Example
-------
>>> inv = InvestorStrategyEngine()
>>> inv.generate(raw_campaign)["recommendation"]
'Proceed with Caution'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.intelligence_aggregator import IntelligenceAggregator
from src.utils.config import load_config
from src.utils.logger import get_logger

# Bands in ascending order of conviction (index arithmetic = one-notch downgrade).
_BANDS = ["Do Not Invest Yet", "Monitor Progress", "Proceed with Caution", "Strong Investment"]
_DEFERRED_DILIGENCE = {
    b: {"status": "deferred", "reason": "not in a pre-launch Kickstarter snapshot"}
    for b in ["financial_diligence", "cap_table", "team_diligence", "market_sizing",
              "traction_metrics", "unit_economics"]
}
_CAPTION = ("Investor recommendation band mapped from the funding-readiness stage + reachable "
            "headroom (model-simulated, not causal), with the top risk surfaced. Standard diligence "
            "(financials, cap table, team, market, traction) is deferred — no data.")


class InvestorStrategyEngine:
    """Project the intelligence into an investor recommendation band + reason.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    cf_pipeline:
        Optional pre-built Day-8 pipeline (share the warm stack).
    aggregator:
        Optional pre-built Phase-B :class:`IntelligenceAggregator` (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None,
                 aggregator: Optional[IntelligenceAggregator] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reco.investor")

        c = self.config.get("day11", {}).get("investor", {})
        self.low_conf = float(c.get("low_confidence_downgrade", 0.6))
        self.out_path = Path(c.get("out_path", "artifacts/recommendation/investor_strategy.json"))
        self.strong_margin = float(self.config.get("day10", {}).get("stage", {})
                                   .get("strong_margin", 0.15))

        self.agg = aggregator or IntelligenceAggregator(self.config, self.logger, cf_pipeline=cf_pipeline)

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any], intelligence: Optional[dict[str, Any]] = None,
                 confidence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Investor recommendation for one campaign.

        ``confidence`` (Phase G) is optional — the Phase-H pipeline threads it in to allow a
        one-notch downgrade of a positive band when confidence is low.
        """
        vi = intelligence if intelligence is not None else self.agg.aggregate(raw)
        stage = vi["evolution"]["stage"]
        ra = vi["copilot"]["risk_analysis"]
        top_risk = (ra.get("risks") or [{}])[0]
        objections = vi["copilot"]["investor_objections"]
        top_obj = objections[0] if objections else {}
        conf_score = (confidence or {}).get("score")

        band = self._band(stage, conf_score)
        p, opt, t = (stage["funding_probability"], stage["optimized_probability"], stage["threshold"])
        out = {
            "campaign": vi["campaign"],
            "recommendation": band,
            "funding_probability": p,
            "optimized_probability": opt,
            "threshold": t,
            "reachable_headroom": round(opt - p, 4),
            "overall_risk_level": ra.get("overall_risk_level"),
            "reason": self._reason(band, p, opt, t, top_risk),
            "key_risk": {"category": top_risk.get("category"), "severity": top_risk.get("severity"),
                         "reason": top_risk.get("reason")},
            "key_objection": {"dimension": top_obj.get("dimension"), "concern": top_obj.get("concern")},
            "conditions_to_reconsider": self._conditions(band, opt, t, top_obj),
            "deferred_diligence": _DEFERRED_DILIGENCE,
            "honesty_note": _CAPTION,
        }
        if confidence is not None:
            out["confidence"] = confidence
        self.logger.info("Investor strategy: %s (%s, %.0f%% -> %.0f%%).", band, stage["stage"],
                         p * 100, opt * 100)
        return out

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Investor strategies for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.generate(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved investor strategies -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _band(self, stage: dict[str, Any], conf_score: Optional[float]) -> str:
        s = stage["stage"]
        opt, t = stage["optimized_probability"], stage["threshold"]
        if s == "Strong":
            idx = 3
        elif s == "Ready":
            idx = 2
        elif s == "Fixable":
            idx = 2 if opt >= t + self.strong_margin else 1   # strong upside -> Caution, else Monitor
        else:                                                 # Not ready
            idx = 0
        if conf_score is not None and conf_score < self.low_conf and idx >= 2:
            idx -= 1                                          # low confidence downgrades a positive band
        return _BANDS[idx]

    @staticmethod
    def _reason(band: str, p: float, opt: float, t: float, top_risk: dict[str, Any]) -> str:
        risk = top_risk.get("reason", "")
        if band == "Strong Investment":
            return (f"The model estimates {p:.0%} funding probability — comfortably above the "
                    f"{t:.0%} bar. Residual risk: {risk}")
        if band == "Proceed with Caution" and p >= t:
            return (f"Fundable as-is at {p:.0%} (just above the {t:.0%} bar) but on a thin margin. "
                    f"Main risk: {risk}")
        if band == "Proceed with Caution":
            return (f"Below the bar as-is ({p:.0%}) but with strong reachable upside to ~{opt:.0%} "
                    f"if the team executes the priority changes. Main risk: {risk}")
        if band == "Monitor Progress":
            return (f"Below the {t:.0%} bar ({p:.0%}); a path to ~{opt:.0%} exists but is borderline. "
                    f"Watch whether the team addresses: {risk}")
        return (f"Even optimized the model estimates ~{opt:.0%}, below the {t:.0%} bar. {risk} "
                f"The fundamentals need to change first.")

    @staticmethod
    def _conditions(band: str, opt: float, t: float, top_obj: dict[str, Any]) -> list[str]:
        if band == "Strong Investment":
            return ["Standard diligence (deferred here) before committing."]
        conds = []
        dim = top_obj.get("dimension")
        if dim:
            conds.append(f"Team addresses the top concern ({dim}).")
        if opt >= t:
            conds.append(f"Campaign reaches its reachable readiness (~{opt:.0%}) before launch.")
        else:
            conds.append("Fundamentals (goal / scope / category) are reworked to clear the bar.")
        return conds
