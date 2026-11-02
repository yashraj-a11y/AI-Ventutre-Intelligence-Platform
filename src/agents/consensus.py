"""Phase I — Consensus & Conflict Resolution (Day-12, Layer 12).

Turn the six agent reports into one **committee decision** via a **confidence-weighted stance
vote** (deliberately a different mechanism from the InvestmentAgent's score-mean, so comparing them
is informative):

    committee_score = Σ wᵢ·confᵢ·stanceᵢ / Σ wᵢ·confᵢ  ∈ [-1, 1]   (positive +1 / cautious 0 / negative -1)

mapped to Strong Invest / Proceed to Due Diligence / Monitor Progress / Do Not Invest. Two honesty
features: a **risk veto** (the RiskAgent's decisive-negative stance caps an otherwise-positive
committee) and **conflict surfacing** (agents dissenting from the majority are named). Agent weights
account for correlation (the CompetitionAgent is down-weighted for its disclosed overlap with the
FinancialAgent — C-7).

**Honesty (C-7):** the agents are correlated views of one shared signal set, not independent
experts, so confidence is **not** inflated by agent count. Deterministic, no faiss.

Example
-------
>>> ce = ConsensusEngine()
>>> ce.run(raw_campaign)["committee_recommendation"]
'Do Not Invest'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.agents.base import clip01
from src.agents.orchestrator import AgentOrchestrator
from src.counterfactual.pipeline import CounterfactualPipeline
from src.utils.config import load_config
from src.utils.logger import get_logger

_STANCE = {"positive": 1, "cautious": 0, "negative": -1}
_CAPTION = ("Committee = confidence-weighted stance vote over correlated specialist views of ONE "
            "shared signal set (not independent experts) — so confidence is NOT inflated by agent "
            "count, and the CompetitionAgent is down-weighted for overlap with the FinancialAgent. "
            "A decisive RiskAgent negative vetoes a positive committee. Model-simulated, not causal.")


class ConsensusEngine:
    """Resolve agent stances into a committee decision (weighted vote + risk veto + conflicts).

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    orchestrator:
        Optional pre-built Phase-H :class:`AgentOrchestrator` (for ``run`` / examples).
    cf_pipeline:
        Optional pre-built Day-8 pipeline (share the warm stack when building an orchestrator).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 orchestrator: Optional[AgentOrchestrator] = None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None,
                 aggregator=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.agents.consensus")

        c = self.config.get("day12", {}).get("consensus", {})
        self.weights = dict(c.get("weights", {}))
        self.veto_conf = float(c.get("veto_confidence", 0.6))
        self.strong_at = float(c.get("strong_at", 0.40))
        self.proceed_at = float(c.get("proceed_at", 0.10))
        self.monitor_at = float(c.get("monitor_at", -0.30))
        self.out_path = Path(c.get("out_path", "artifacts/agents/committee_decision.json"))

        self.orchestrator = orchestrator
        self._cf = cf_pipeline
        self._agg = aggregator                             # optional shared IntelligenceAggregator

    # ------------------------------------------------------------------ public
    def decide(self, agent_reports: list[dict[str, Any]]) -> dict[str, Any]:
        """Committee decision from a list of AgentReports (pure — no aggregation)."""
        w = lambda r: self.weights.get(r["agent"], 1.0)
        den = sum(w(r) * r["confidence"] for r in agent_reports)
        num = sum(w(r) * r["confidence"] * _STANCE[r["stance"]] for r in agent_reports)
        score = num / den if den else 0.0
        band = self._band(score)

        # risk veto: a decisive RiskAgent negative caps an otherwise-positive committee.
        risk = next((r for r in agent_reports if r["agent"] == "RiskAgent"), None)
        veto_eligible = bool(risk and risk["stance"] == "negative" and risk["confidence"] >= self.veto_conf)
        veto_applied = veto_eligible and band in ("Strong Invest", "Proceed to Due Diligence")
        if veto_applied:
            band = "Monitor Progress"

        stances = [r["stance"] for r in agent_reports]
        majority = max(set(stances), key=stances.count)
        agreement = clip01(stances.count(majority) / len(stances)) if stances else 0.5
        conflicts = [{"agent": r["agent"], "stance": r["stance"], "vs_majority": majority,
                      "score": r["score"]}
                     for r in agent_reports if _STANCE[r["stance"]] * _STANCE[majority] < 0]

        wmean_conf = (sum(w(r) * r["confidence"] for r in agent_reports) /
                      sum(w(r) for r in agent_reports)) if agent_reports else 0.5
        overall_conf = min(0.99, max(0.5, wmean_conf * (0.6 + 0.4 * agreement)))

        return {
            "committee_recommendation": band,
            "committee_score": round(score, 3),
            "overall_confidence": round(overall_conf, 3),
            "agreement_level": round(agreement, 3),
            "agent_votes": [{"agent": r["agent"], "stance": r["stance"],
                             "confidence": r["confidence"], "weight": w(r)} for r in agent_reports],
            "conflicts": conflicts,
            "risk_veto": {"eligible": veto_eligible, "applied": veto_applied},
            "rationale": self._rationale(band, score, agent_reports, veto_applied, conflicts),
            "deferred": {k: v for r in agent_reports for k, v in r.get("deferred", {}).items()},
            "honesty_note": _CAPTION,
        }

    def run(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Orchestrate the six agents, then decide — the full committee pass for one campaign."""
        if self.orchestrator is None:
            self.orchestrator = AgentOrchestrator(self.config, self.logger, cf_pipeline=self._cf,
                                                  aggregator=self._agg)
        collected = self.orchestrator.run(raw)
        decision = self.decide(collected["agent_reports"])
        decision["campaign"] = collected["campaign"]
        decision["agent_reports"] = collected["agent_reports"]
        self.logger.info("Committee: %s (score %+.2f, conf %.2f, agreement %.2f%s).",
                         decision["committee_recommendation"], decision["committee_score"],
                         decision["overall_confidence"], decision["agreement_level"],
                         ", VETO" if decision["risk_veto"]["applied"] else "")
        return decision

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Committee decisions for a spread of test campaigns -> deliverable JSON."""
        df = pd.read_csv(self.config.get("day6", {}).get("knowledge_base", {})
                         .get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(self.config.get("day6", {}).get("knowledge_base", {})
                          .get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.run(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved committee decisions -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _band(self, score: float) -> str:
        if score >= self.strong_at:
            return "Strong Invest"
        if score >= self.proceed_at:
            return "Proceed to Due Diligence"
        if score >= self.monitor_at:
            return "Monitor Progress"
        return "Do Not Invest"

    @staticmethod
    def _rationale(band, score, reports, veto_applied, conflicts) -> str:
        neg = sum(1 for r in reports if r["stance"] == "negative")
        pos = sum(1 for r in reports if r["stance"] == "positive")
        txt = (f"Committee recommendation: {band} (score {score:+.2f}; {pos} positive / {neg} "
               f"negative of {len(reports)} agents).")
        if veto_applied:
            txt += " Risk veto applied — capped at Monitor Progress despite a positive tally."
        if conflicts:
            txt += f" Dissent: {', '.join(c['agent'] for c in conflicts)} ({conflicts[0]['stance']})."
        return txt
