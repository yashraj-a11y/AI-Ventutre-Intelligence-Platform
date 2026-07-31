"""Phase F — Counterfactual Explanation Generator (Day-8, Layer 7).

Turn a ranked, evidence-backed recommendation into a founder-readable action sentence:

    "Lower your funding goal from $38,000 to $5,000. The model estimates your funding
    probability would rise from 3% to 8% (+5 points). Film & Video campaigns asking
    goal <= $5,000 funded at 51% vs a 40% baseline (lift 1.27, n=20,767); funded
    campaigns here had a median goal of $5,000 vs $10,000 for failed. Confidence: high.
    Note: this is the model's estimate based on historical association, not a guarantee."

Three parts, always: the **action** (what to change, in plain words), the **model-estimated
effect** (baseline → new probability, in points), the **real evidence** (the Phase-D
statement), and a **mandatory honesty caption** — a counterfactual is a model-simulated
what-if, never a promise.

Only real, mutable features appear (goal / duration / launch timing / title). Reuses the
readable-feature style of Day-5/7, specialised to *actions*.

Example
-------
>>> gen = CounterfactualExplanationGenerator()
>>> print(gen.explain(ranked[0]))
"""

from __future__ import annotations

from typing import Any, Optional

from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("this is the model's estimate based on historical association, not a guarantee")


def action_phrase(feature: str, current: Any, proposed: Any) -> str:
    """Plain-language action for a recommendation (real mutable features only)."""
    if feature == "goal":
        return f"Lower your funding goal from ${float(current):,.0f} to ${float(proposed):,.0f}"
    if feature == "campaign_duration":
        return f"Shorten your campaign from {current} to {proposed} days"
    if feature == "launch_weekday":
        return f"Launch on {proposed} instead of {current}"
    if feature == "title":
        return "Add a number to your title (e.g. an edition or volume number)"
    return f"Change {feature} from {current} to {proposed}"


class CounterfactualExplanationGenerator:
    """Render ranked recommendations into founder-readable, honesty-captioned advice.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.cf.explanation")

    # ------------------------------------------------------------------ public
    def explain(self, rec: dict[str, Any]) -> str:
        """One ranked recommendation → a founder-readable action sentence."""
        action = action_phrase(rec["feature"], rec.get("current"), rec.get("proposed"))
        base_p = rec.get("baseline_p")
        new_p = rec.get("new_p")
        pts = round(float(rec.get("delta", 0.0)) * 100)

        effect = ""
        if base_p is not None and new_p is not None:
            effect = (f" The model estimates your funding probability would rise from "
                      f"{base_p:.0%} to {new_p:.0%} (+{pts} points).")
        ev = rec.get("evidence", {}) or {}
        evidence = f" {ev['statement']}" if ev.get("available") and ev.get("statement") else ""
        conf = rec.get("confidence")
        conf_txt = f" Confidence: {conf}." if conf else ""

        return f"{action}.{effect}{evidence}{conf_txt} Note: {_CAPTION}."

    def explain_all(self, ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach an ``explanation`` string to each ranked recommendation."""
        out = [{**r, "explanation": self.explain(r)} for r in ranked]
        self.logger.info("Generated %d counterfactual explanations.", len(out))
        return out
