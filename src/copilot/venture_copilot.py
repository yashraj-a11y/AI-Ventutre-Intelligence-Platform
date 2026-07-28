"""Phase H — Venture Copilot Explanation Layer (Day-5, honesty-tiered).

The founder-facing advisor. For one campaign it produces a structured **brief** that
composes everything Day-5 built:

- Phase-E :class:`LocalExplainer` → the campaign's calibrated probability + signed
  real-feature SHAP contributions,
- Phase-G :func:`humanize` → those contributions as English phrases,
- Day-4 :class:`VentureScoreGenerator` scores (carried on the explanation) → funding /
  startup / readiness / risk numbers with their provenance.

**Honesty tiers — the whole point of this redesign:**

- **GENERATED** (grounded in real SHAP contributions on real features):
  Strengths, Weaknesses, Risks, and plain-language score explanations.
- **DEFERRED** (no data basis in this dataset — returned as explicit "deferred"
  placeholders, *never* invented prose): Market Analysis, Competitor Analysis, Founder
  Intelligence, Market Opportunities. These await Layers 2–3 (retrieval / richer data).

A copilot that *made up* a market or founder narrative would be worse than useless — it
would look authoritative while being fabricated. So this one says "pending" and means it.

Example
-------
>>> copilot = VentureCopilot()
>>> brief = copilot.advise_raw(one_campaign_df)[0]
>>> brief.strengths[0]
'Being a Theater project'
>>> brief.deferred["market_analysis"]["status"]
'deferred'
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.explainability.explanation_generator import ExplanationGenerator, humanize
from src.explainability.local_explainer import LocalExplainer, LocalExplanation
from src.utils.config import load_config
from src.utils.logger import get_logger

# Analyses this dataset cannot support — deferred, never fabricated.
_DEFERRED_REASON = ("no data basis in this dataset (no market / competitor / founder / "
                    "team signals) — awaits Layers 2–3 (retrieval + richer data); not fabricated")
_DEFERRED_SECTIONS = ["market_analysis", "competitor_analysis",
                      "founder_intelligence", "market_opportunities"]


@dataclass
class VentureBrief:
    """A per-campaign advisory brief with explicit honesty tiers."""

    headline: str
    funding_probability: float
    predicted_label: str
    investment_readiness: str
    # GENERATED tier (grounded in real SHAP contributions):
    strengths: list[str]
    weaknesses: list[str]
    risks: list[str]
    score_explanations: dict[str, str]
    # DEFERRED tier (no data — placeholders, not prose):
    deferred: dict[str, dict[str, str]] = field(default_factory=dict)
    honesty_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VentureCopilot:
    """Generate honest, SHAP-grounded venture briefs (real-feature SWOT + deferred tiers).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    explainer, generator:
        Optional pre-built collaborators (share a warm SHAP engine across calls).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 explainer: Optional[LocalExplainer] = None,
                 generator: Optional[ExplanationGenerator] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.copilot")

        c = self.config.get("day5", {}).get("copilot", {})
        self.max_factors = int(c.get("max_factors", 4))
        self.n_examples = int(c.get("n_examples", 4))
        self.briefs_path = Path(
            c.get("briefs_path", "artifacts/explainability/copilot_briefs.json")
        )

        self.explainer = explainer or LocalExplainer(self.config, self.logger)
        self.generator = generator or ExplanationGenerator(
            self.config, self.logger, explainer=self.explainer
        )

    # ------------------------------------------------------------------ public
    def advise_raw(self, raw) -> list[VentureBrief]:
        """Advisory brief(s) for raw campaign(s)."""
        return [self._brief(e) for e in self.explainer.explain_raw(raw)]

    def advise_transformed(self, X_row, proba: Optional[float] = None) -> VentureBrief:
        """Advisory brief for one already-transformed campaign row."""
        return self._brief(self.explainer.explain_transformed(X_row, proba))

    def brief(self, exp: LocalExplanation) -> VentureBrief:
        """Build a brief from an already-computed explanation (shares the SHAP pass).

        Lets an orchestrator (Phase I) reuse one :class:`LocalExplanation` across the
        readable generator and the copilot instead of re-running SHAP.
        """
        return self._brief(exp)

    def generate_examples(self, n: Optional[int] = None, random_state: int = 11) -> Path:
        """Brief a spread of real campaigns and save the deliverable JSON."""
        import pandas as pd
        import numpy as np

        n = n or self.n_examples
        feat_cols = ["name", "category", "main_category", "currency", "deadline",
                     "goal", "launched", "country", "usd_goal_real", "name_clean"]
        df = pd.read_csv("data/processed/kickstarter_processed.csv")
        sample = df.sample(max(n * 12, 60), random_state=random_state).reset_index(drop=True)
        briefs = self.advise_raw(sample[feat_cols])

        order = np.argsort([b.funding_probability for b in briefs])
        picks = [briefs[order[int(t)]] for t in np.linspace(0, len(order) - 1, n)]
        self.briefs_path.parent.mkdir(parents=True, exist_ok=True)
        self.briefs_path.write_text(json.dumps([b.to_dict() for b in picks], indent=2))
        self.logger.info("Saved %d copilot briefs -> %s", len(picks), self.briefs_path)
        return self.briefs_path

    # --------------------------------------------------------------- internals
    def _brief(self, exp: LocalExplanation) -> VentureBrief:
        pct = round(exp.funding_probability * 100)
        readiness = exp.scores["investment_readiness"]["value"]

        strengths = self._phrases(exp.top_positive)
        weaknesses = self._phrases(exp.top_negative)
        risks = self._risks(exp, weaknesses)

        return VentureBrief(
            headline=(f"{readiness} investment readiness — {pct}% modeled funding chance "
                      f"({exp.predicted_label})."),
            funding_probability=exp.funding_probability,
            predicted_label=exp.predicted_label,
            investment_readiness=readiness,
            strengths=strengths,
            weaknesses=weaknesses,
            risks=risks,
            score_explanations=self._score_explanations(exp.scores),
            deferred={s: {"status": "deferred", "reason": _DEFERRED_REASON}
                      for s in _DEFERRED_SECTIONS},
            honesty_note=exp.honesty_note,
        )

    def _phrases(self, contributions) -> list[str]:
        """Humanized, capitalized real-feature phrases (Phase-G labels)."""
        out = []
        for c in contributions[: self.max_factors]:
            p = humanize(c, self.generator.z_high)
            out.append(p[0].upper() + p[1:])
        return out

    def _risks(self, exp: LocalExplanation, weaknesses: list[str]) -> list[str]:
        """Risks grounded in the failure-risk score + the real negative factors."""
        fail_pct = round(exp.scores["failure_risk"]["value"] * 100)
        risks = [f"Modeled funding risk: {fail_pct}% chance this campaign does not fund."]
        for w in weaknesses[:2]:
            risks.append(f"{w} is weighing the model's estimate down.")
        return risks

    @staticmethod
    def _score_explanations(scores: dict[str, dict[str, Any]]) -> dict[str, str]:
        """Plain-language, provenance-carrying blurbs for the DERIVED scores only."""
        def blurb(key: str, template: str) -> str:
            s = scores[key]
            return template.format(value=s["value"]) + f" (basis: {s['basis']})."

        return {
            "funding_probability": blurb(
                "funding_probability",
                "Calibrated {value} probability of funding if the campaign runs to completion"),
            "startup_score": blurb(
                "startup_score", "Startup score {value}/100, a rescale of funding probability"),
            "investment_readiness": blurb(
                "investment_readiness", "Investment readiness: {value}"),
            "failure_risk": blurb(
                "failure_risk", "Failure risk {value}, the complement of funding probability"),
        }
