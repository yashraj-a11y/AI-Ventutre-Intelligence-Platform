"""Phase F — Legal & Risk Analyst Agent (Day-12, Layer 12).

The risk specialist — and the one agent that carries the consensus **risk veto**. It reads Day-9
``risk_analysis`` (modeled failure risk, ``overall_risk_level``, the ranked risk list) and the
investor objections, and scores **safety** (``100 - risk``) so the uniform contract stays
consistent: high safety -> positive (low risk), low safety -> negative (risky).

    risk_score = failure_risk * failure_weight + high-severity bonus
    safety     = 100 - risk_score        # the reported score

``overall_risk_level`` is exposed in ``key_metric`` so the Phase-I consensus can apply the veto (a
High-risk negative stance caps an otherwise-positive committee).

**Deferred:** regulatory / legal / IP / operational risk — none exist in the dataset. Deterministic,
no faiss.

Example
-------
>>> ra = RiskAgent()
>>> ra.analyze(raw_campaign)["key_metric"]
{'name': 'overall_risk_level', 'value': 'High'}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.agents.base import agent_report, clip, clip01
from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.intelligence_aggregator import IntelligenceAggregator
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED = {
    b: {"status": "deferred", "reason": "no legal / regulatory / operational data in the dataset"}
    for b in ["regulatory_risk", "legal_risk", "ip_risk", "operational_risk"]
}
_CAPTION = ("Risk = modeled failure risk + goal / funding / cohort risks (Day-9) + investor "
            "objections. Score is SAFETY (100 - risk). Model-simulated, not causal; regulatory / "
            "legal / IP / operational risks are deferred.")


class RiskAgent:
    """Score campaign safety (100 - risk) from Day-9 risks + objections; carries the veto signal.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    aggregator:
        Optional pre-built Day-11 :class:`IntelligenceAggregator` (share the warm stack).
    cf_pipeline:
        Optional pre-built Day-8 pipeline (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 aggregator: Optional[IntelligenceAggregator] = None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.agents.risk")

        c = self.config.get("day12", {}).get("risk", {})
        self.positive_at = float(c.get("positive_at", 60))
        self.negative_at = float(c.get("negative_at", 40))
        self.failure_weight = float(c.get("failure_weight", 70))
        self.high_bonus = float(c.get("high_severity_bonus", 10))
        self.high_cap = float(c.get("high_severity_cap", 30))
        self.top_risks = int(c.get("top_risks", 2))
        self.out_path = Path(c.get("out_path", "artifacts/agents/risk_analysis.json"))

        self.agg = aggregator or IntelligenceAggregator(self.config, self.logger, cf_pipeline=cf_pipeline)

    # ------------------------------------------------------------------ public
    def analyze(self, raw: dict[str, Any],
                intelligence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Risk (safety) AgentReport for one campaign."""
        vi = intelligence if intelligence is not None else self.agg.aggregate(raw)
        ra = vi["copilot"]["risk_analysis"]
        risks = ra.get("risks", [])
        level = ra.get("overall_risk_level", "Medium")
        failure = float(ra.get("failure_risk", 1 - vi["prediction"]["funding_probability"]))
        n_high = sum(1 for r in risks if str(r.get("severity")) == "High")
        objections = vi["copilot"].get("investor_objections", [])

        risk_score = clip(failure * self.failure_weight + min(n_high * self.high_bonus, self.high_cap), 0, 100)
        score = 100 - risk_score                             # report SAFETY (uniform: high = favorable)

        findings = [f"Overall risk: {level}. Modeled failure risk {failure:.0%}."]
        for r in risks[:self.top_risks]:
            findings.append(f"{r.get('severity')} — {r.get('reason')}")
        if objections:
            findings.append(f"{len(objections)} investor objection(s) raised; top: "
                            f"{objections[0].get('dimension')}.")

        # key_metric = modeled failure risk (numeric, consistent with the safety score). The Phase-I
        # veto keys off this agent's OWN negative stance + confidence, not Day-9's absolute risk band
        # (which is High for most campaigns and would contradict a positive relative-safety stance).
        report = agent_report(
            "RiskAgent", "Risk & investor objections (score = safety, higher = safer)", score,
            self._confidence(failure, risks), findings,
            {"name": "modeled_failure_risk", "value": round(failure, 3)},
            {**_DEFERRED, **ra.get("deferred_risks", {})}, _CAPTION, self.positive_at, self.negative_at)
        self.logger.info("RiskAgent: %s safety %d (conf %.2f) — %s risk, %d high.", report["stance"],
                         report["score"], report["confidence"], level, n_high)
        return report

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Risk analyses for a spread of test campaigns -> deliverable JSON."""
        df = pd.read_csv(self.config.get("day6", {}).get("knowledge_base", {})
                         .get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(self.config.get("day6", {}).get("knowledge_base", {})
                          .get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.analyze(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved risk analyses -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _confidence(self, failure: float, risks: list[dict[str, Any]]) -> float:
        """Confident when the failure signal is decisive and the risks agree in severity."""
        decisiveness = clip01(abs(failure - 0.5) * 2)        # far from 50/50
        sev = [str(r.get("severity")) for r in risks]
        consistency = (max(sev.count(s) for s in set(sev)) / len(sev)) if sev else 0.5
        return min(0.99, max(0.5, 0.5 + 0.49 * (0.6 * decisiveness + 0.4 * consistency)))
