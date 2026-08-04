"""Phase H — Venture Copilot Pipeline (Day-9, Layer 8 entry point).

One object that turns a **raw campaign** into a full investor-grade copilot brief, integrating
every prior layer:

    Day-4 prediction + Day-5 SHAP + Day-7 patterns + Day-8 counterfactuals
      → investor objections + risk analysis + SWOT + category analysis
        + competition analysis + strategic recommendations

Every section is a reframing of the *same* real signals, so the brief is coherent (a top
objection is a top risk is a SWOT weakness is a fixable lever).

**One warm stack, shared.** A single Day-8 :class:`CounterfactualPipeline` (calibrated model +
Day-7 item frame) and a single Day-5 :class:`LocalExplainer` (SHAP, sharing that model) are
built once; the six Day-9 engines borrow them. The counterfactual is computed **once per
campaign** and passed to the objection / SWOT / recommendation engines (no redundant re-runs).
No faiss/torch → **no OpenMP concern** (preflight-verified).

Deterministic templates throughout — no LLM, so the brief is reproducible and cannot
hallucinate a competitor or a market number.

Example
-------
>>> copilot = VentureCopilotPipeline()
>>> brief = copilot.analyze(raw_campaign)
>>> brief["investor_objections"][0]["dimension"], brief["swot"]["strengths"][0]["point"]
('goal_size', 'A modest funding goal')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.copilot.competitor_analysis import CompetitionAnalysisEngine
from src.copilot.market_analysis import CategoryAnalysisEngine
from src.copilot.objection_generator import ObjectionGenerator
from src.copilot.recommendation_generator import RecommendationEngine
from src.copilot.risk_analysis import RiskAnalysisEngine
from src.copilot.swot_analysis import SWOTAnalysisEngine
from src.counterfactual.pipeline import CounterfactualPipeline
from src.explainability.local_explainer import LocalExplainer
from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("An AI copilot brief: model predictions + historical associations reframed as "
            "investor-style reasoning via deterministic templates — not a real investor's "
            "judgement, and not verified market/competitor facts. Founder, market-size, named "
            "competition, and financials are deferred (no data).")


class VentureCopilotPipeline:
    """Integrate all Day-9 engines into one investor-grade copilot brief (warm shared stack).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    cf_pipeline:
        Optional pre-built Day-8 :class:`CounterfactualPipeline` (share the warm stack — e.g. so the
        Day-11 recommendation engine runs copilot + evolution + scenario on ONE loaded model).
        If ``None`` (default), a fresh one is built — existing behavior is unchanged.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional["CounterfactualPipeline"] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.copilot.pipeline")

        c = self.config.get("day9", {}).get("copilot_pipeline", {})
        self.n_examples = int(c.get("n_examples", 3))
        self.out_path = Path(c.get("out_path", "artifacts/copilot/venture_copilot_brief.json"))

        # One warm stack shared by every engine (injected for Day-11, else self-built).
        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)  # model + item frame
        self.lx = LocalExplainer(self.config, self.logger, engine=self.cf.simulator.engine)  # SHAP
        frame, base = self.cf.evidence.frame, self.cf.evidence.base

        self.objections = ObjectionGenerator(self.config, self.logger, cf_pipeline=self.cf)
        self.risk = RiskAnalysisEngine(self.config, self.logger, cf_pipeline=self.cf)
        self.swot = SWOTAnalysisEngine(self.config, self.logger, cf_pipeline=self.cf, local_explainer=self.lx)
        self.category = CategoryAnalysisEngine(self.config, self.logger, frame=frame, base=base)
        self.competition = CompetitionAnalysisEngine(self.config, self.logger, frame=frame, base=base)
        self.recommend = RecommendationEngine(self.config, self.logger, cf_pipeline=self.cf)
        self.logger.info("VentureCopilotPipeline ready (warm shared stack, no faiss).")

    # ------------------------------------------------------------------ public
    def analyze(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Full copilot brief for one campaign."""
        raw = dict(raw)
        analysis = self.cf.analyze(raw)                 # counterfactual ONCE -> reused
        base_p = analysis["baseline_p"]
        threshold = float(self.cf.simulator.engine.threshold)

        return {
            "campaign": {k: raw.get(k) for k in ("name", "main_category", "category",
                                                 "country", "usd_goal_real")},
            "funding_probability": round(base_p, 4),
            "predicted_label": "successful" if base_p >= threshold else "failed",
            "decision_threshold": round(threshold, 4),
            "investor_objections": self.objections.generate(raw, analysis=analysis),
            "risk_analysis": self.risk.analyze(raw),
            "swot": self.swot.analyze(raw, analysis=analysis),
            "category_analysis": self.category.analyze(raw),
            "competition_analysis": self.competition.analyze(raw),
            "recommendations": self.recommend.generate(raw, analysis=analysis),
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: Optional[int] = None, random_state: int = 3) -> Path:
        """Copilot briefs for a spread of improvable TEST campaigns -> deliverable JSON."""
        import numpy as np

        n = n or self.n_examples
        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        briefs = [self.analyze(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(briefs, indent=2, default=str))
        self.logger.info("Saved %d copilot briefs -> %s", len(briefs), self.out_path)
        return self.out_path
