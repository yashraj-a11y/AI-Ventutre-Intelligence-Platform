"""Phase H — Venture Intelligence Pipeline (Day-6, Layers 2/3 entry point).

One object that turns a **raw campaign** into prediction **plus** historical evidence:

    raw campaign ──▶ Day-4 PredictionEngine ─▶ calibrated funding probability + scores
                └──▶ serialize ─▶ SBERT ─▶ FAISS ─▶ retrieve ─▶ EvidenceGenerator
                                                                    │
                                                    prediction + evidence, together

The cross-layer join is the point of Day 6: the retrieval prior (neighbours' funding
rate) is reported **next to** the model's probability, so the evidence corroborates or
challenges the prediction. This is **non-circular** because the knowledge base is
*train-only* — a query campaign's own outcome is never in the KB it searches (and for a
KB-member query we self-exclude by ``campaign_id``).

Design:

- **One warm stack, shared.** A single ``EvidenceGenerator`` (holding the retriever →
  search engine → vector DB + embedder) is built once; the optional Day-4
  ``PredictionEngine`` loads once. Each ``analyze`` call is one encode + one FAISS search
  (+ one model forward pass).
- **Prediction is optional and fault-tolerant.** If the Day-4 artifacts are missing or a
  raw campaign lacks the prediction columns, the pipeline degrades to **evidence-only**
  rather than failing.

Example
-------
>>> pipe = VentureIntelligencePipeline()
>>> rep = pipe.analyze(raw_campaign)
>>> rep.prediction["funding_probability"], rep.evidence["retrieval_prior"]
(0.31, 0.34)
>>> rep.evidence["prior_vs_model"]
'corroborates the model'
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from src.retrieval.evidence_generator import EvidenceGenerator
from src.retrieval.venture_retriever import _json_safe
from src.utils.config import load_config
from src.utils.logger import get_logger

# Raw feature columns the Day-4 PredictionEngine expects (label/state excluded).
_PREDICT_COLS = ["name", "category", "main_category", "currency", "deadline",
                 "goal", "launched", "country", "usd_goal_real", "name_clean"]


@dataclass
class VentureIntelligenceReport:
    """Prediction (optional) + historical evidence for one campaign."""

    query: dict[str, Any]
    prediction: Optional[dict[str, Any]]     # Day-4 funding probability + label + scores
    evidence: dict[str, Any]                 # Phase-G evidence pack
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VentureIntelligencePipeline:
    """Raw campaign → prediction + retrieved historical evidence (one entry point).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    evidence_generator:
        Optional pre-built :class:`EvidenceGenerator` (share a warm retrieval stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 evidence_generator: Optional[EvidenceGenerator] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.retrieval.pipeline")

        c = self.config.get("day6", {}).get("pipeline", {})
        self.attach_prediction = bool(c.get("attach_prediction", True))
        self.n_examples = int(c.get("n_examples", 3))
        self.report_path = Path(c.get("report_path",
                                      "artifacts/retrieval/intelligence_report_examples.json"))

        # ORDER MATTERS (native OpenMP coexistence): load the Day-4 predictor (lightgbm,
        # which links its own libomp) BEFORE building the retrieval/faiss stack. Loading
        # faiss first and then calling lightgbm predict segfaults on macOS; lightgbm-first
        # initialises cleanly. (Combined with OMP_NUM_THREADS=1 from the package __init__,
        # which covers the torch+faiss case.)
        self.predictor = self._load_predictor() if self.attach_prediction else None
        self.evidence = evidence_generator or EvidenceGenerator(self.config, self.logger)

    # ------------------------------------------------------------------ public
    def analyze(self, raw: Union[dict, pd.Series],
                exclude_campaign_ids: Optional[list[int]] = None) -> VentureIntelligenceReport:
        """Predict + retrieve evidence for one raw campaign (evidence corroborates model)."""
        raw = dict(raw)
        prediction, model_prob = self._predict(raw)
        pack = self.evidence.generate(raw, model_probability=model_prob,
                                      exclude_campaign_ids=exclude_campaign_ids)
        return VentureIntelligenceReport(
            query=pack.pop("query", None) or self._query_view(raw),
            prediction=prediction,
            evidence=pack,
            meta={"has_prediction": prediction is not None,
                  "kb": "train-only (non-circular)",
                  "retrieval_prior": pack.get("retrieval_prior"),
                  "prior_vs_model": pack.get("prior_vs_model")},
        )

    def generate_examples(self, n: Optional[int] = None) -> Path:
        """Analyze a spread of unseen TEST campaigns (not in the KB) -> deliverable JSON."""
        import numpy as np

        n = n or self.n_examples
        kb_cfg = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb_cfg.get("processed_path", "data/processed/kickstarter_processed.csv"))
        test_idx = np.load(Path(kb_cfg.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[test_idx].sample(n, random_state=3).reset_index(drop=True)
        reports = [self.analyze(r.to_dict()).to_dict() for _, r in sample.iterrows()]

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(reports, indent=2, default=_json_safe))
        self.logger.info("Saved %d intelligence reports -> %s", len(reports), self.report_path)
        return self.report_path

    # --------------------------------------------------------------- internals
    def _load_predictor(self):
        """Load the Day-4 PredictionEngine; None if artifacts/deps are unavailable."""
        try:
            from src.models.prediction_engine import PredictionEngine
            pe = PredictionEngine(self.config, self.logger)
            self.logger.info("Day-4 PredictionEngine attached.")
            return pe
        except Exception as e:                       # missing artifacts / deps -> evidence-only
            self.logger.warning("Prediction unavailable (%s) — evidence-only mode.", e)
            return None

    def _predict(self, raw: dict) -> tuple[Optional[dict], Optional[float]]:
        if self.predictor is None:
            return None, None
        missing = [c for c in _PREDICT_COLS if c not in raw]
        if missing:
            self.logger.warning("Raw campaign missing %s — evidence-only.", missing)
            return None, None
        try:
            out = self.predictor.predict(pd.DataFrame([{c: raw[c] for c in _PREDICT_COLS}]))[0]
            return out.to_dict(), float(out.funding_probability)
        except Exception as e:
            self.logger.warning("Prediction failed (%s) — evidence-only.", e)
            return None, None

    @staticmethod
    def _query_view(raw: dict) -> dict[str, Any]:
        return {k: raw.get(k) for k in ("name", "category", "main_category",
                                        "country", "usd_goal_real")}
