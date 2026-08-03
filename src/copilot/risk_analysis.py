"""Phase C — Risk Analysis Engine (Day-9, Layer 8).

The same real signals as the objection generator, quantified as a **risk register**: each
assessable risk category with a severity (from magnitude), a reason, and real historical
evidence. Risk categories the data can't support are **deferred** as explicit placeholders —
never fabricated.

Assessable risks (real signals):
- **funding_risk** — the headline: failure risk = 1 − calibrated probability, vs the threshold.
- **financial_goal_risk** — goal high relative to the category's funded-median.
- **category_fit_risk** — the campaign's category funds below baseline (lift < 1).
- **timing_risk** — the campaign's duration band underperforms.
- **cohort_risk** — campaigns most like it (category + goal band) mostly failed.

Deferred (no data): technology, founder/team, unit-economics/burn, regulatory, market risk.

Shares :func:`~src.copilot.objection_generator.campaign_signals` with Phase B, so a risk and
its matching objection rest on the *same* number. Reuses the Day-8 pipeline; no faiss → no
OpenMP.

Example
-------
>>> eng = RiskAnalysisEngine()
>>> report = eng.analyze(raw_campaign)
>>> report["risks"][0]["category"], report["risks"][0]["severity"]
('financial_goal_risk', 'High')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.copilot.objection_generator import campaign_signals
from src.counterfactual.pipeline import CounterfactualPipeline
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED_RISKS = {
    b: {"status": "deferred", "reason": "no data in this dataset"}
    for b in ["technology_risk", "founder_team_risk", "unit_economics_risk",
              "regulatory_risk", "market_size_risk"]
}
_CAPTION = ("Risks derived from model predictions + historical funding rates (metadata + titles). "
            "Founder / market / financial / regulatory risks are deferred — no data.")


class RiskAnalysisEngine:
    """Quantify a campaign's assessable risks; defer the rest.

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
        self.logger = logger or get_logger("avip.copilot.risk")

        c = self.config.get("day9", {}).get("risk", {})
        self.sev_high = float(c.get("severity_high", 0.40))
        self.sev_med = float(c.get("severity_medium", 0.20))
        self.goal_high_ratio = float(c.get("goal_high_ratio", 1.3))
        self.lift_concern = float(c.get("lift_concern", 0.9))
        self.out_path = Path(c.get("out_path", "artifacts/copilot/risk_report.json"))

        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)
        self.frame = self.cf.evidence.frame
        self.base = self.cf.evidence.base

    # ------------------------------------------------------------------ public
    def analyze(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Full risk report for one campaign."""
        raw = dict(raw)
        base_p = self.cf.simulator.baseline(raw)
        threshold = float(self.cf.simulator.engine.threshold)
        s = campaign_signals(self.frame, self.base, raw)

        risks = []
        risks += self._funding_risk(base_p, threshold)
        risks += self._goal_risk(s)
        risks += self._category_risk(s)
        risks += self._timing_risk(s)
        risks += self._cohort_risk(s)
        risks.sort(key=lambda r: r["score"], reverse=True)

        return {
            "campaign": {k: raw.get(k) for k in ("name", "main_category", "usd_goal_real")},
            "funding_probability": round(base_p, 4),
            "failure_risk": round(1 - base_p, 4),
            "overall_risk_level": self._severity(min(1.0, 1 - base_p)),
            "risks": risks,
            "deferred_risks": _DEFERRED_RISKS,
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 3) -> Path:
        """Risk reports for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.analyze(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved risk reports -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _funding_risk(self, base_p, threshold):
        fr = 1 - base_p
        score = min(1.0, fr) if base_p < threshold else min(1.0, fr) * 0.5
        return [self._risk("funding_risk", score,
            f"Modeled failure risk {fr:.0%} (funding probability {base_p:.0%}"
            + (f", below the {threshold:.0%} threshold)." if base_p < threshold else ").."),
            f"Calibrated model: {base_p:.0%} funding probability.", "Day-4 prediction",
            {"failure_risk": round(fr, 4), "threshold": round(threshold, 4)})]

    def _goal_risk(self, s):
        goal, fm = s["goal"], s["funded_med"]
        if not fm or goal <= fm * self.goal_high_ratio:
            return []
        score = min(1.0, (goal / fm - 1) / 2)
        return [self._risk("financial_goal_risk", score,
            f"Goal ${goal:,.0f} is {goal/fm:.1f}x the {s['main']} funded-median (${fm:,.0f}); "
            f"large goals are harder to reach.",
            f"Funded {s['main']} campaigns asked a median ${fm:,.0f}.", "Day-7 patterns",
            {"goal": goal, "funded_median": fm})]

    def _category_risk(self, s):
        if s["cat_lift"] >= self.lift_concern:
            return []
        score = min(1.0, 1 - s["cat_lift"])
        return [self._risk("category_fit_risk", score,
            f"{s['main']} is a low-success category ({s['cat_rate']:.0%} funded, lift "
            f"{s['cat_lift']:.2f}).",
            f"{s['main']}: {s['cat_rate']:.0%} vs {self.base:.0%} baseline (n={s['cat_n']:,}).",
            "Day-7 patterns", {"lift": round(s["cat_lift"], 3), "n": s["cat_n"]})]

    def _timing_risk(self, s):
        if s["dur"] is None or s["dur_lift"] >= self.lift_concern:
            return []
        score = min(1.0, 1 - s["dur_lift"])
        return [self._risk("timing_risk", score,
            f"The {s['dur']}-day duration band underperforms ({s['dur_rate']:.0%} funded, lift "
            f"{s['dur_lift']:.2f}).",
            f"Duration '{s['dur_band']}': {s['dur_rate']:.0%} vs {self.base:.0%} (n={s['dur_n']:,}).",
            "Day-7 patterns", {"duration_days": s["dur"], "lift": round(s["dur_lift"], 3)})]

    def _cohort_risk(self, s):
        if s["cohort_n"] < 100 or s["cohort_lift"] >= self.lift_concern:
            return []
        score = min(1.0, 1 - s["cohort_lift"])
        return [self._risk("cohort_risk", score,
            f"Similar campaigns ({s['main']}, {s['goal_band']}) funded at only "
            f"{s['cohort_rate']:.0%} (lift {s['cohort_lift']:.2f}).",
            f"Cohort funding rate {s['cohort_rate']:.0%} vs {self.base:.0%} (n={s['cohort_n']:,}).",
            "Day-6/7 cohort", {"lift": round(s["cohort_lift"], 3), "n": s["cohort_n"]})]

    def _risk(self, category, score, reason, evidence, source, ev):
        return {"category": category, "severity": self._severity(score), "score": round(score, 3),
                "reason": reason, "historical_evidence": evidence,
                "source_layer": source, "evidence": ev}

    def _severity(self, score: float) -> str:
        return ("High" if score >= self.sev_high else
                "Medium" if score >= self.sev_med else "Low")
