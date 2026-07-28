"""Phase I — Explainability Pipeline (Day-5, Layer 5 — the single entry point).

One object that turns a **raw campaign** into the *complete* explainability bundle,
chaining everything Day 5 built on top of the Day-4 prediction stack:

    raw campaign DataFrame
      → Day-4 PredictionEngine        (Day-3 transform → calibrated probability; train/serve-safe)
      → Phase-D/E SHAP local explanation   (signed real-feature contributions)
      → Phase-G readable explanation       (founder-facing sentences)
      → Phase-H Venture Copilot brief       (honesty-tiered SWOT + deferred sections)
    plus Phase-B/C global context           (top drivers + cross-model agreement)

Two design guarantees:

- **One SHAP pass per batch.** The pipeline builds a single warm ``LocalExplainer``
  (interventional background computed once) and *shares* it with the readable generator
  and the copilot. Each campaign is explained exactly once; the generator and copilot
  reuse that ``LocalExplanation`` — no redundant recompute.
- **Train/serve consistency inherited.** Because the chain starts at the Day-4
  ``PredictionEngine`` (which wraps the Day-3 feature pipeline), a new campaign goes
  through the identical transforms that built the training matrix. The explanation
  describes the *same* representation the model was trained and calibrated on.

Example
-------
>>> pipe = ExplainabilityPipeline()
>>> report = pipe.explain(one_campaign_df)[0]
>>> report.readable["explanation_text"]
'This campaign is predicted successful, with an 88% chance of funding. ...'
>>> report.brief["deferred"]["market_analysis"]["status"]
'deferred'
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.copilot.venture_copilot import VentureCopilot
from src.explainability.explanation_generator import ExplanationGenerator
from src.explainability.local_explainer import LocalExplainer, LocalExplanation
from src.utils.config import load_config
from src.utils.logger import get_logger


@dataclass
class ExplainabilityReport:
    """The full per-campaign explainability bundle (JSON-ready)."""

    funding_probability: float
    predicted_label: str
    decision_threshold: float
    scores: dict[str, Any]                 # Day-4 venture scores (derived + deferred)
    local_explanation: dict[str, Any]      # Phase E structured contributions
    readable: dict[str, Any]               # Phase G founder-readable text + factors
    brief: dict[str, Any]                  # Phase H copilot brief (SWOT + deferred)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExplainabilityPipeline:
    """Raw campaign → complete explainability bundle (global + local), one SHAP pass.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.explain.pipeline")

        c = self.config.get("day5", {}).get("pipeline", {})
        self.top_k_global = int(c.get("top_k_global", 15))
        self.global_csv = Path(c.get("global_importance_csv",
                                      "artifacts/explainability/global_importance.csv"))
        self.coef_csv = Path(c.get("coefficients_csv",
                                    "artifacts/explainability/coefficients.csv"))
        self.report_path = Path(c.get("report_path",
                                      "artifacts/explainability/explainability_report_examples.json"))

        # One warm explainer shared by every layer -> one SHAP background, one pass.
        self.explainer = LocalExplainer(self.config, self.logger)
        self.generator = ExplanationGenerator(self.config, self.logger, explainer=self.explainer)
        self.copilot = VentureCopilot(
            self.config, self.logger, explainer=self.explainer, generator=self.generator
        )
        self.logger.info("ExplainabilityPipeline ready (shared warm SHAP engine).")

    # ------------------------------------------------------------------ public
    def explain(self, raw) -> list[ExplainabilityReport]:
        """Full bundle for raw campaign(s). Explains each campaign exactly once."""
        exps = self.explainer.explain_raw(raw)          # single SHAP pass for the batch
        return [self._report(e) for e in exps]

    def global_context(self, top_k: Optional[int] = None) -> dict[str, Any]:
        """Model-level view: top SHAP drivers, top ± coefficients, cross-model agreement."""
        import pandas as pd

        k = top_k or self.top_k_global
        ctx: dict[str, Any] = {"available": True}
        if self.global_csv.exists():
            gi = pd.read_csv(self.global_csv).nlargest(k, "mean_abs_shap")
            ctx["top_shap_features"] = gi[["feature", "group", "mean_abs_shap"]].to_dict("records")
        else:
            ctx["top_shap_features"] = []
        if self.coef_csv.exists():
            co = pd.read_csv(self.coef_csv)
            ctx["top_positive_coef"] = (co[co["coef"] > 0].nlargest(k, "coef")
                                        [["feature", "coef"]].to_dict("records"))
            ctx["top_negative_coef"] = (co[co["coef"] < 0].nsmallest(k, "coef")
                                        [["feature", "coef"]].to_dict("records"))
        if not (self.global_csv.exists() or self.coef_csv.exists()):
            ctx = {"available": False, "reason": "run Phases B/C first"}
        return ctx

    def explain_and_save(self, raw, path: Optional[Path] = None) -> Path:
        """Explain campaign(s) and persist the bundle list as JSON."""
        reports = self.explain(raw)
        path = Path(path) if path else self.report_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([r.to_dict() for r in reports], indent=2))
        self.logger.info("Saved %d explainability reports -> %s", len(reports), path)
        return path

    # --------------------------------------------------------------- internals
    def _report(self, exp: LocalExplanation) -> ExplainabilityReport:
        return ExplainabilityReport(
            funding_probability=exp.funding_probability,
            predicted_label=exp.predicted_label,
            decision_threshold=exp.decision_threshold,
            scores=exp.scores,
            local_explanation=exp.to_dict(),                 # Phase E
            readable=self.generator.to_readable(exp),        # Phase G (reuses exp)
            brief=self.copilot.brief(exp).to_dict(),         # Phase H (reuses exp)
            meta={"base_value": exp.base_value, "margin": exp.margin,
                  "shap_additivity_error": exp.additivity_error,
                  "shap_space": "log_odds_margin (calibration is a monotone squash on top)"},
        )
