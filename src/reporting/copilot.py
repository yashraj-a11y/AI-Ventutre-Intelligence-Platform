"""Phase F — Venture Copilot Q&A (Day-13, Layer 13).

A **deterministic intent-router** conversational layer — NOT an LLM. It matches a question to a fixed
set of intents by keyword and returns a templated answer assembled from the ReportBundle (the real
Day 4-12 numbers). Out-of-scope questions (MRR, revenue, moat, founder background, …) hit an honest
**"not available in this dataset" fallback** — the copilot never invents an answer.

This is a NEW module — distinct from the pre-existing Day-5 ``src/copilot/venture_copilot.py``
(which produces SHAP briefs). No LLM, no faiss, deterministic.

Example
-------
>>> qa = VentureCopilotQA()
>>> qa.ask("What should I improve first?", raw=raw_campaign)["intent"]
'improve'
>>> qa.ask("What is my MRR?", raw=raw_campaign)["deferred"]
True
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.agents.consensus import ConsensusEngine
from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.pipeline import RecommendationPipeline
from src.reporting.base import feature_prose
from src.utils.config import load_config
from src.utils.logger import get_logger

# Out-of-scope concepts the copilot refuses to answer (honest fallback, never invented).
_DEFERRED_TERMS = [
    "mrr", "revenue", "retention", "churn", "burn", "runway", "moat", "valuation",
    "expected return", "irr", "cac", "ltv", "tam", "market size", "unit economics",
    "co-founder", "cofounder", "team size", "founder experience", "profit", "cap table",
]
_DEMO_QUESTIONS = [
    "Why is my funding probability low?",
    "What should I improve first?",
    "Which campaigns are most similar to mine?",
    "Why do similar campaigns fail?",
    "What is my biggest risk?",
    "What should I do before I launch?",
    "Should an investor invest?",
    "What is my MRR?",
]
_CAPTION = ("Deterministic intent-router over the real Day 4-12 outputs — not an LLM, never invents "
            "an answer. Out-of-scope questions (revenue / moat / founder background / …) return an "
            "honest 'not available' fallback. Model-simulated, not causal.")


# --------------------------------------------------------------- answer templates
def _ans_probability(b) -> tuple[str, str]:
    ir = b["decision"]["investment_readiness"]; conf = b["decision"]["confidence"]["score"]
    return (f"Your funding probability is {ir['funding_probability']:.0%} ({ir['stage']}). With the "
            f"priority pre-launch changes it's reachable to ~{ir['optimized_probability']:.0%}. "
            f"Confidence {conf:.0%}.", "Days 4 / 10 / 11")


def _ans_improve(b) -> tuple[str, str]:
    top = b["decision"]["priority_improvements"][0]
    return (f"First: {top['action']} — the model estimates about +{top['expected_impact_points']} "
            f"points. {top['evidence'] or ''}".strip(), "Day-11 prioritization (Day-7 evidence)")


def _ans_similar(b) -> tuple[str, str]:
    c = b["vi"]["cohort"]; pos = b["vi"]["copilot"]["competition_analysis"].get("your_position", {})
    return (f"The closest comparison is the {c['n_similar']:,} other {c['main_category']} campaigns in "
            f"our data, which fund at {c['cohort_funding_rate']:.0%} (vs a {c['platform_base_rate']:.0%} "
            f"baseline). You're positioned like the {pos.get('positioned_like', 'cohort')}.",
            "Day-9 category cohort")


def _ans_why_fail(b) -> tuple[str, str]:
    comp = b["vi"]["copilot"]["competition_analysis"]
    winners = (comp.get("what_winners_do") or ["Funded campaigns tend to ask for less."])[0]
    note = (comp.get("your_position", {}).get("notes") or [""])[0]
    return (f"{winners} {note}".strip(), "Day-9 competition (what winners do)")


def _ans_risk(b) -> tuple[str, str]:
    risks = b["decision"]["risks"]
    top = (risks.get("top_risks") or [{}])[0]
    return (f"Your biggest risk: {top.get('reason', 'n/a')} (Overall risk: "
            f"{risks.get('overall_risk_level')}).", "Day-9 risk analysis")


def _ans_before_launch(b) -> tuple[str, str]:
    focus = b["decision"]["founder_strategy"]["focus_first"]
    return (f"Before you launch, focus on: {'; '.join(focus).lower()}. These are the highest-impact "
            f"controllable changes.", "Day-11 founder strategy")


def _ans_recommendation(b) -> tuple[str, str]:
    cm = b["committee"]
    return (f"The committee's recommendation is {cm['committee_recommendation']} "
            f"({cm['overall_confidence']:.0%} confidence). {cm['rationale']}", "Day-12 committee")


def _ans_strengths(b) -> tuple[str, str]:
    pos = b["vi"]["drivers"]["top_positive"][:3]
    items = ", ".join(feature_prose(d["feature"], d.get("descriptor")) for d in pos)
    return (f"Working in your favor: {items}.", "Day-5 SHAP (positive drivers)")


_INTENTS = [
    ("probability", ["probability", "odds", "chance", "likely", "readiness", "how good"], _ans_probability),
    ("improve", ["improve", "better", "first", "fix", "increase", "boost", "should i do"], _ans_improve),
    ("similar", ["similar", "competitor", "competition", "comparable", "like mine", "others"], _ans_similar),
    ("why_fail", ["fail", "failed", "failure", "why do", "went wrong"], _ans_why_fail),
    ("risk", ["risk", "danger", "red flag", "concern", "worry"], _ans_risk),
    ("before_launch", ["before", "fundrais", "launch", "prepare", "milestone", "next step", "roadmap"], _ans_before_launch),
    ("recommendation", ["invest", "recommend", "should i invest", "decision", "committee", "verdict"], _ans_recommendation),
    ("strengths", ["strength", "working", "favor", "good about", "positive"], _ans_strengths),
]


class VentureCopilotQA:
    """Deterministic Q&A over the ReportBundle (intent-router, no LLM).

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    cf_pipeline, rec_pipeline, consensus:
        Optional pre-built collaborators (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None,
                 rec_pipeline: Optional[RecommendationPipeline] = None,
                 consensus: Optional[ConsensusEngine] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reporting.copilot")

        c = self.config.get("day13", {}).get("copilot", {})
        self.out_path = Path(c.get("out_path", "artifacts/reports/copilot_log.json"))

        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)
        self.rec = rec_pipeline or RecommendationPipeline(self.config, self.logger, cf_pipeline=self.cf)
        self.consensus = consensus or ConsensusEngine(self.config, self.logger, cf_pipeline=self.cf)
        self.agg = self.rec.agg

    # ------------------------------------------------------------------ public
    def ask(self, question: str, bundle: Optional[dict[str, Any]] = None,
            raw: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Answer one question deterministically from the bundle (or from a raw campaign)."""
        b = bundle if bundle is not None else self._bundle(raw)
        q = question.lower()

        deferred = next((t for t in _DEFERRED_TERMS if t in q), None)
        if deferred:
            return {"question": question, "intent": "deferred", "deferred": True, "source": "deferred",
                    "answer": (f"That isn't available — this platform has no {deferred} data for "
                               f"Kickstarter campaigns (they're anonymous, pre-launch). I can answer "
                               f"questions about funding probability, what to improve, similar "
                               f"campaigns, risks, and pre-launch steps.")}

        best, score = None, 0
        for intent_id, keywords, fn in _INTENTS:
            s = sum(1 for k in keywords if k in q)
            if s > score:
                best, score = (intent_id, fn), s
        if not best:
            return {"question": question, "intent": "unknown", "deferred": False, "source": "-",
                    "answer": ("I can answer questions about your funding probability, what to improve "
                               "first, similar campaigns, why similar campaigns fail, your biggest "
                               "risk, and what to do before launch.")}

        intent_id, fn = best
        answer, source = fn(b)
        return {"question": question, "intent": intent_id, "deferred": False,
                "source": source, "answer": answer}

    def converse(self, raw: Optional[dict[str, Any]] = None,
                 bundle: Optional[dict[str, Any]] = None,
                 questions: Optional[list[str]] = None) -> dict[str, Any]:
        """Answer a set of questions for one campaign (builds the bundle once, or reuses one given)."""
        b = bundle if bundle is not None else self._bundle(raw)
        qs = questions if questions is not None else _DEMO_QUESTIONS
        turns = [self.ask(q, bundle=b) for q in qs]
        return {"campaign": b["decision"]["campaign"], "turns": turns, "honesty_note": _CAPTION}

    def generate_examples(self, n: int = 2, random_state: int = 4) -> Path:
        """Copilot Q&A logs for a spread of test campaigns -> deliverable JSON."""
        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.converse(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved copilot logs -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _bundle(self, raw: Optional[dict[str, Any]]) -> dict[str, Any]:
        if raw is None:
            raise ValueError("VentureCopilotQA.ask needs either a bundle= or a raw= campaign.")
        return {"campaign": None, "vi": self.agg.aggregate(raw),
                "decision": self.rec.run(raw), "committee": self.consensus.run(raw)}
