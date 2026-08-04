"""Phase G — Confidence Scoring Engine (Day-11, Layer 11).

The genuinely new intellectual content of Day 11: estimate **how much to trust** the final
recommendation, from *real* agreement signals only — never a fabricated number. Six factors, each
in ``[0, 1]``, combined as a normalized weighted mean and mapped to ``[floor, ceil]``:

1. **prediction_margin** — how far the calibrated probability sits from the decision bar
   (decisive vs borderline).
2. **cohort_agreement** — do similar campaigns (same-category cohort) actually fund the way the
   model predicts?
3. **pattern_support** — Day-7 historical strength (lift x sample size) behind the recommendations.
4. **scenario_stability** — do the Day-10 scenarios behave monotonically (no erratic swings)?
5. **cf_stability** — does the Day-8 *greedy* optimum agree with the Day-10 *joint* best (two
   independent estimates of the reachable optimum)?
6. **model_agreement** — do the three Day-4 models (calibrated / best / logistic regression) agree
   on the label? (Optional — dropped and renormalized if the extra models can't be loaded.)

Confidence is a property of the **ensemble**, not any one model — so it can only be computed here,
where every signal is visible. The per-factor breakdown and the model probabilities are returned, so
nothing is a black box. No faiss.

Example
-------
>>> cs = ConfidenceScorer()
>>> out = cs.score(raw_campaign)
>>> 0.5 <= out["score"] <= 0.99, list(out["factors"])
(True, ['prediction_margin', 'cohort_agreement', 'pattern_support', 'scenario_stability', 'cf_stability', 'model_agreement'])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.intelligence_aggregator import IntelligenceAggregator
from src.utils.config import load_config
from src.utils.logger import get_logger

_LABELS = {
    "prediction_margin": "a clear prediction margin",
    "cohort_agreement": "agreement among similar campaigns",
    "pattern_support": "strong historical pattern support",
    "scenario_stability": "stable scenario behavior",
    "cf_stability": "consistent reachability across methods",
    "model_agreement": "agreement across models",
}
_CAPTION = ("Confidence is a transparent weighted blend of real agreement signals (each shown) — "
            "prediction margin, similar-campaign agreement, historical pattern support, scenario "
            "stability, cross-method reachability, and model agreement. Not a probability; a "
            "reliability estimate over the ensemble.")


class ConfidenceScorer:
    """Transparent ensemble-confidence from real agreement signals.

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
        self.logger = logger or get_logger("avip.reco.confidence")

        c = self.config.get("day11", {}).get("confidence", {})
        self.weights = dict(c.get("weights", {}))
        self.margin_scale = float(c.get("margin_scale", 0.30))
        self.top_k = int(c.get("pattern_top_k", 3))
        self.lift_span = float(c.get("pattern_lift_span", 0.6))
        self.min_n = int(c.get("pattern_min_n", 100))
        self.stability_scale = float(c.get("stability_scale", 0.20))
        self.floor = float(c.get("floor", 0.50))
        self.ceil = float(c.get("ceil", 0.99))
        self.use_model_agreement = bool(c.get("use_model_agreement", True))
        self.out_path = Path(c.get("out_path", "artifacts/recommendation/confidence_scores.json"))

        self.agg = aggregator or IntelligenceAggregator(self.config, self.logger, cf_pipeline=cf_pipeline)
        self.engine = self.agg.cf.simulator.engine          # shared warm PredictionEngine
        self._extra_models = self._load_extra_models(c.get("extra_models", {}))

    # ------------------------------------------------------------------ public
    def score(self, raw: dict[str, Any],
              intelligence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Confidence in the final recommendation for one campaign."""
        vi = intelligence if intelligence is not None else self.agg.aggregate(raw)
        p = vi["prediction"]["funding_probability"]
        t = vi["prediction"]["threshold"]
        label_success = p >= t

        factors = {
            "prediction_margin": _clip(abs(p - t) / self.margin_scale),
            "cohort_agreement": self._cohort_agreement(vi, label_success),
            "pattern_support": self._pattern_support(vi),
            "scenario_stability": self._scenario_stability(vi),
            "cf_stability": self._cf_stability(vi),
        }
        model_probs = None
        if self.use_model_agreement and self._extra_models:
            agree, model_probs = self._model_agreement(raw, t)
            if agree is not None:
                factors["model_agreement"] = agree

        raw_score = self._weighted(factors)
        score = round(self.floor + (self.ceil - self.floor) * raw_score, 4)

        out = {
            "campaign": vi["campaign"],
            "score": score,
            "factors": {k: round(v, 3) for k, v in factors.items()},
            "weights": {k: self.weights.get(k) for k in factors},
            "reason": self._reason(factors),
            "honesty_note": _CAPTION,
        }
        if model_probs is not None:
            out["model_probabilities"] = {k: round(v, 4) for k, v in model_probs.items()}
        self.logger.info("Confidence %.2f (%s).", score, out["reason"][:60])
        return out

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Confidence scores for a spread of test campaigns -> deliverable JSON."""
        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.score(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved confidence scores -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- factors
    @staticmethod
    def _cohort_agreement(vi: dict[str, Any], label_success: bool) -> float:
        rate = float(vi["cohort"]["cohort_funding_rate"])
        return rate if label_success else 1.0 - rate

    def _pattern_support(self, vi: dict[str, Any]) -> float:
        recs = vi["counterfactual"]["ranked_recommendations"][:self.top_k]
        if not recs:
            return 0.5
        strengths = []
        for r in recs:
            ev = r.get("evidence") or {}
            lift = float(ev.get("lift", 1.0)); n = int(ev.get("n", 0))
            s = _clip((lift - 1.0) / self.lift_span)
            strengths.append(s * 0.5 if n < self.min_n else s)
        return float(np.mean(strengths))

    @staticmethod
    def _scenario_stability(vi: dict[str, Any]) -> float:
        marg = [float(m["marginal_gain"]) for m in vi["scenario"]["marginal_values"]]
        if not marg:
            return 0.5
        return float(np.mean([1.0 if m >= -0.005 else 0.0 for m in marg]))   # monotone / non-harmful

    def _cf_stability(self, vi: dict[str, Any]) -> float:
        opt = float(vi["evolution"]["stage"]["optimized_probability"])       # Day-8 greedy optimum
        best = float(vi["scenario"]["best_scenario"]["probability"])         # Day-10 joint best
        return _clip(1.0 - abs(opt - best) / self.stability_scale)

    def _model_agreement(self, raw: dict[str, Any], t: float):
        """Fraction of the 3 Day-4 models whose label matches the majority (real cross-model check)."""
        try:
            X = self.engine.pipeline.transform(pd.DataFrame([raw]))
            probs = {"calibrated": float(self.engine.model.predict_proba(X)[:, 1][0])}
            for name, m in self._extra_models.items():
                probs[name] = float(m.predict_proba(X)[:, 1][0])
            labels = [p >= t for p in probs.values()]
            majority = max(set(labels), key=labels.count)
            return sum(1 for l in labels if l == majority) / len(labels), probs
        except Exception as e:                                                # graceful: drop the factor
            self.logger.warning("model_agreement unavailable (%s) — dropping the factor.", e)
            return None, None

    # --------------------------------------------------------------- internals
    def _weighted(self, factors: dict[str, float]) -> float:
        w = self.weights
        tot = sum(w.get(k, 0.0) for k in factors)                            # renormalizes if a factor is absent
        return sum(w.get(k, 0.0) * v for k, v in factors.items()) / tot if tot else 0.0

    def _reason(self, factors: dict[str, float]) -> str:
        strong = [_LABELS[k] for k, v in factors.items() if v >= 0.6 and k in _LABELS]
        weak = [_LABELS[k] for k, v in factors.items() if v < 0.4 and k in _LABELS]
        parts = ("Supported by " + _join(strong) + "." if strong
                 else "No single signal is strongly corroborating.")
        if weak:
            parts += " Main uncertainty: " + _join(weak) + "."
        return parts

    def _load_extra_models(self, spec: dict[str, str]) -> dict[str, Any]:
        models: dict[str, Any] = {}
        for name, path in spec.items():
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                self.logger.warning("Could not load extra model %s (%s).", name, e)
        return models


def _clip(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]
