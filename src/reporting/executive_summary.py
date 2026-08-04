"""Phase E — Executive Summary Generator (Day-13, Layer 13).

The most compressed projection of the bundle — five dashboard numbers and a one-line headline, for a
quick decision or a portfolio dashboard. Presentation only, no new analysis.

    Startup score (readiness index)  <- Day-11
    Funding probability              <- Day-4
    Risk                             <- Day-11 risk level
    Recommendation                   <- Day-12 committee
    Confidence                       <- Day-12 committee

Deterministic, no LLM, no faiss.

Example
-------
>>> gen = ExecutiveSummaryGenerator()
>>> gen.generate(raw_campaign)["report_type"]
'Executive Summary'
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.agents.consensus import ConsensusEngine
from src.counterfactual.pipeline import CounterfactualPipeline
from src.recommendation.pipeline import RecommendationPipeline
from src.reporting.base import section
from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("Executive Summary — five headline numbers reformatted from the Day 4-12 analysis (no "
            "new analysis). Model-simulated, not causal.")


class ExecutiveSummaryGenerator:
    """Render the dashboard executive summary from the ReportBundle.

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
        self.logger = logger or get_logger("avip.reporting.executive")

        c = self.config.get("day13", {}).get("executive", {})
        self.out_path = Path(c.get("out_path", "artifacts/reports/executive_summary.json"))

        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)
        self.rec = rec_pipeline or RecommendationPipeline(self.config, self.logger, cf_pipeline=self.cf)
        self.consensus = consensus or ConsensusEngine(self.config, self.logger, cf_pipeline=self.cf)
        self.agg = self.rec.agg

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any], bundle: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Executive summary for one campaign (uses a prebuilt bundle if given)."""
        if bundle is not None:
            decision, committee = bundle["decision"], bundle["committee"]
        else:
            decision = self.rec.run(raw)
            committee = self.consensus.run(raw)

        ir = decision["investment_readiness"]
        camp = decision["campaign"]
        risk_level = decision["risks"].get("overall_risk_level", "Unknown")
        band = committee["committee_recommendation"]
        conf = committee["overall_confidence"]

        headline = (f"{camp['name']}: {band} — {ir['stage']} at {ir['funding_probability']:.0%} "
                    f"(readiness {ir['readiness_index']}), {conf:.0%} confidence.")
        metrics = {
            "Startup score": ir["readiness_index"],
            "Funding probability": f"{ir['funding_probability']:.0%}",
            "Risk": risk_level,
            "Recommendation": band,
            "Confidence": f"{conf:.0%}",
        }

        report = {
            "report_type": "Executive Summary",
            "campaign": camp,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "headline": headline,
            "sections": [
                section("headline", "Executive Summary", "Day-12 committee + Day-11", "text", headline),
                section("dashboard", "Dashboard", "Days 4 / 10 / 11 / 12", "metrics", metrics),
            ],
            "deferred_appendix": {
                "note": "Not covered — the dataset has no data for these; not fabricated.",
                "items": sorted(decision["deferred"].keys())},
            "honesty_note": _CAPTION,
        }
        self.logger.info("Executive summary: %s -> %s (%.0f%%).", camp["name"], band, conf * 100)
        return report

    def generate_examples(self, n: int = 5, random_state: int = 4) -> Path:
        """Executive summaries for a spread of test campaigns -> deliverable JSON."""
        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.generate(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved executive summaries -> %s", self.out_path)
        return self.out_path
