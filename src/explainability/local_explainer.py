"""Phase E — Local Startup Explanation (Day-5, Layer 5).

*Why did THIS campaign get this score?* The founder-facing counterpart to the global
rankings. For one campaign we return its calibrated **funding probability** plus the
**top real-feature contributions** that pushed it up or down — from per-instance SHAP
on the base LightGBM.

Composition, not reimplementation:

- **:class:`~src.models.prediction_engine.PredictionEngine`** (Day-4) → the Day-3
  feature transform + the *calibrated* funding probability + decision threshold +
  venture scores. Guarantees train/serve consistency (the same transforms that built
  the training matrix).
- **:class:`~src.explainability.shap_explainer.SHAPExplainer`** (Phase D) → the signed
  per-feature contributions for that one transformed row (instant: ~20 ms).

Two honesty rules, enforced structurally:

1. **Every factor is a REAL feature.** All 865 features are campaign metadata or title
   TF-IDF tokens — there is no founder/team/market/revenue signal to leak. A text
   token with a positive value means *the word is present in the title*; we say exactly
   that ("title term 'documentary'"), never invent a narrative.
2. **Probability is calibrated; contributions explain the margin.** SHAP attributes the
   uncalibrated log-odds; calibration is a monotone squash on top, so the sign and
   ordering of contributions are the story, and the probability is the calibrated number
   the founder should trust. Both are reported, and their roles are labelled.

Example
-------
>>> lx = LocalExplainer()
>>> exp = lx.explain_raw(one_campaign_df)[0]
>>> exp.funding_probability, exp.top_positive[0].feature
(0.31, 'usd_goal_real')
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.explainability.global_importance import is_tabular
from src.explainability.shap_explainer import SHAPExplainer
from src.models.prediction_engine import PredictionEngine
from src.utils.config import load_config
from src.utils.logger import get_logger


@dataclass
class Contribution:
    """One real feature's signed push on the funding margin."""

    feature: str
    group: str                    # "text" | "tabular"
    contribution: float           # signed SHAP (log-odds); + pushes funding up
    feature_value: float          # the (standardized) value the model saw
    descriptor: str               # light human hint (Phase G polishes into prose)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LocalExplanation:
    """A full per-campaign explanation."""

    funding_probability: float    # calibrated
    predicted_label: str
    decision_threshold: float
    base_value: float             # expected margin (φ₀), log-odds
    margin: float                 # this campaign's log-odds (φ₀ + Σ contributions)
    top_positive: list[Contribution]
    top_negative: list[Contribution]
    additivity_error: float
    honesty_note: str
    scores: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["top_positive"] = [c.to_dict() for c in self.top_positive]
        d["top_negative"] = [c.to_dict() for c in self.top_negative]
        return d


_HONESTY = (
    "Factors are real campaign features (goal, category, timing, title words) the model "
    "learned to associate with funding — not causal levers or founder/market signals "
    "(the dataset has none). Probability is calibrated; contributions explain the log-odds."
)


class LocalExplainer:
    """Explain one campaign: calibrated probability + top ± real-feature contributions.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    engine, shap_explainer:
        Optional pre-built collaborators (handy for tests / to share a warm explainer).
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        logger=None,
        engine: Optional[PredictionEngine] = None,
        shap_explainer: Optional[SHAPExplainer] = None,
    ) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.explain.local")

        c = self.config.get("day5", {}).get("local", {})
        self.top_k = int(c.get("top_k", 8))
        self.examples_path = Path(
            c.get("examples_path", "artifacts/explainability/local_explanation_examples.json")
        )

        self.engine = engine or PredictionEngine(self.config, self.logger)
        self.shap = shap_explainer or SHAPExplainer(self.config, self.logger).load().build_explainer()

    # ------------------------------------------------------------------ public
    def explain_raw(self, raw: pd.DataFrame) -> list[LocalExplanation]:
        """Explain raw campaign(s): transform → calibrated prob + SHAP contributions."""
        X = self.engine.pipeline.transform(raw)          # Day-3 transforms → (n, 865) sparse
        proba = self.engine.model.predict_proba(X)[:, 1]  # calibrated
        return self._explain_matrix(X, proba)

    def explain_transformed(
        self, X_row: Any, proba: Optional[float] = None
    ) -> LocalExplanation:
        """Explain an already-transformed row (e.g. a stored test-set campaign).

        ``X_row`` may be a 1-row sparse matrix or a dense vector. If ``proba`` is not
        given it is computed with the calibrated model.
        """
        X = X_row if sp.issparse(X_row) else sp.csr_matrix(np.asarray(X_row).reshape(1, -1))
        if X.shape[0] != 1:
            raise ValueError("explain_transformed expects exactly one row.")
        p = float(proba) if proba is not None else float(self.engine.model.predict_proba(X)[0, 1])
        return self._explain_matrix(X, np.array([p]))[0]

    def save_examples(self, explanations: list[LocalExplanation]) -> Path:
        """Persist a list of explanations as JSON (the Phase-E deliverable)."""
        self.examples_path.parent.mkdir(parents=True, exist_ok=True)
        self.examples_path.write_text(
            json.dumps([e.to_dict() for e in explanations], indent=2)
        )
        self.logger.info("Saved %d local explanations -> %s",
                         len(explanations), self.examples_path)
        return self.examples_path

    # --------------------------------------------------------------- internals
    def _explain_matrix(self, X, proba: np.ndarray) -> list[LocalExplanation]:
        dense = X.toarray() if sp.issparse(X) else np.asarray(X)
        shap_res = self.shap.explain(dense)               # additivity-guarded (Phase D)
        names = shap_res.feature_names
        threshold = self.engine.threshold

        out: list[LocalExplanation] = []
        for i in range(dense.shape[0]):
            sv = shap_res.values[i]
            values = dense[i]
            margin = float(shap_res.base_value + sv.sum())
            order = np.argsort(sv)
            pos = [j for j in order[::-1] if sv[j] > 0][: self.top_k]
            neg = [j for j in order if sv[j] < 0][: self.top_k]
            p = float(proba[i])
            out.append(
                LocalExplanation(
                    funding_probability=round(p, 4),
                    predicted_label="successful" if p >= threshold else "failed",
                    decision_threshold=round(threshold, 4),
                    base_value=round(shap_res.base_value, 4),
                    margin=round(margin, 4),
                    top_positive=[self._contrib(names, sv, values, j) for j in pos],
                    top_negative=[self._contrib(names, sv, values, j) for j in neg],
                    additivity_error=shap_res.additivity_error,
                    honesty_note=_HONESTY,
                    scores=self.engine.scorer.score_campaign(p),
                )
            )
        return out

    def _contrib(self, names, sv, values, j: int) -> Contribution:
        name = names[j]
        group = "tabular" if is_tabular(name) else "text"
        val = float(values[j])
        if group == "text":
            # positive TF-IDF value ⇒ the token is present in the title
            descriptor = f"title term '{name}'" + ("" if val > 0 else " (absent)")
        else:
            descriptor = name
        return Contribution(
            feature=name, group=group,
            contribution=round(float(sv[j]), 4),
            feature_value=round(val, 4),
            descriptor=descriptor,
        )
