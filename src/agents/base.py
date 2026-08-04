"""Shared agent infrastructure (Day-12, Layer 12).

The uniform ``AgentReport`` contract every specialist agent emits, so the orchestrator and consensus
engine treat all agents identically. Deterministic; no state.
"""

from __future__ import annotations

from typing import Any


def stance_from_score(score: float, positive_at: float = 60, negative_at: float = 40) -> str:
    """Map a 0-100 perspective score to a stance."""
    if score >= positive_at:
        return "positive"
    if score < negative_at:
        return "negative"
    return "cautious"


def agent_report(agent: str, perspective: str, score: float, confidence: float,
                 findings: list[str], key_metric: dict[str, Any], deferred: dict[str, Any],
                 caption: str, positive_at: float = 60, negative_at: float = 40) -> dict[str, Any]:
    """Build the uniform AgentReport dict (the contract shared by all six agents)."""
    return {
        "agent": agent,
        "perspective": perspective,
        "score": round(score),
        "stance": stance_from_score(score, positive_at, negative_at),
        "confidence": round(float(confidence), 3),
        "findings": findings,
        "key_metric": key_metric,
        "deferred": deferred,
        "caption": caption,
    }


def clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return lo if x < lo else hi if x > hi else x


def clip01(x: float) -> float:
    return clip(x, 0.0, 1.0)
