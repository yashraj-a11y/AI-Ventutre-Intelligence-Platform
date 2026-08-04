"""Phase H — Reporting Pipeline (Day-13, Layer 13). The unified entry point.

Assemble the :data:`ReportBundle` **once** on one warm faiss-free stack (Day-11 recommendation +
Day-12 committee), then generate all four reports (founder / investor / memo / executive) and the
copilot Q&A from that single bundle, and optionally export them. Because every generator is a pure
function of the bundle, the whole report suite costs **one** bundle, not one per report.

    build_bundle(raw)  ->  {vi, decision, committee}          (the only model-heavy work)
      -> FounderReport / InvestorReport / InvestmentMemo / ExecutiveSummary / CopilotQA  (all pure)
      -> export (JSON / MD / HTML; PDF optional)

Deterministic, no LLM, no faiss. This is where the whole platform ties together for a human reader.

Example
-------
>>> pipe = ReportingPipeline()
>>> out = pipe.run(raw_campaign)
>>> list(out["reports"])
['founder', 'investor', 'memo', 'executive']
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
from src.reporting.copilot import VentureCopilotQA
from src.reporting.executive_summary import ExecutiveSummaryGenerator
from src.reporting.export import ReportExporter
from src.reporting.founder_report import FounderReportGenerator
from src.reporting.investment_memo import InvestmentMemoGenerator
from src.reporting.investor_report import InvestorReportGenerator
from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("A full report suite reformatted from the Day 4-12 analysis on one warm faiss-free stack "
            "(no new analysis). Every section names its source; deferred items are appended, not "
            "fabricated. Deterministic, no LLM. Model-simulated, not causal.")


class ReportingPipeline:
    """Build the ReportBundle once, then generate + export the full report suite.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    cf_pipeline:
        Optional pre-built Day-8 :class:`CounterfactualPipeline` (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reporting.pipeline")

        c = self.config.get("day13", {}).get("pipeline", {})
        self.out_dir = Path(c.get("out_dir", "outputs/reports"))
        self.out_path = Path(c.get("out_path", "artifacts/reports/report_suite.json"))

        # ONE warm stack: cf -> Day-11 recommendation + Day-12 committee, shared into every generator.
        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)
        self.rec = RecommendationPipeline(self.config, self.logger, cf_pipeline=self.cf)
        self.consensus = ConsensusEngine(self.config, self.logger, cf_pipeline=self.cf)
        self.agg = self.rec.agg

        self.founder = FounderReportGenerator(self.config, self.logger, rec_pipeline=self.rec)
        self.investor = InvestorReportGenerator(self.config, self.logger, rec_pipeline=self.rec,
                                                consensus=self.consensus)
        self.memo = InvestmentMemoGenerator(self.config, self.logger, rec_pipeline=self.rec,
                                            consensus=self.consensus)
        self.executive = ExecutiveSummaryGenerator(self.config, self.logger, rec_pipeline=self.rec,
                                                   consensus=self.consensus)
        self.copilot = VentureCopilotQA(self.config, self.logger, rec_pipeline=self.rec,
                                        consensus=self.consensus)
        self.exporter = ReportExporter(self.config, self.logger)
        self.logger.info("ReportingPipeline ready (one warm faiss-free stack, 4 reports + copilot).")

    # ------------------------------------------------------------------ public
    def build_bundle(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Assemble the ReportBundle once (the only model-heavy work)."""
        vi = self.agg.aggregate(raw)
        decision = self.rec.run(raw)
        committee = self.consensus.run(raw)
        return {"campaign": decision["campaign"], "vi": vi, "decision": decision, "committee": committee}

    def run(self, raw: dict[str, Any], export: bool = False,
            formats: Optional[list[str]] = None) -> dict[str, Any]:
        """Generate the full report suite (+ copilot) for one campaign from a single bundle."""
        bundle = self.build_bundle(raw)
        reports = {
            "founder": self.founder.generate(raw, bundle=bundle),
            "investor": self.investor.generate(raw, bundle=bundle),
            "memo": self.memo.generate(raw, bundle=bundle),
            "executive": self.executive.generate(raw, bundle=bundle),
        }
        copilot = self.copilot.converse(bundle=bundle)

        result = {"campaign": bundle["campaign"], "reports": reports, "copilot": copilot,
                  "honesty_note": _CAPTION}
        if export:
            result["exported"] = {name: self.exporter.export(rep, formats=formats, out_dir=self.out_dir)
                                  for name, rep in reports.items()}
        self.logger.info("Report suite: %s (%d reports, %d copilot turns%s).",
                         bundle["campaign"]["name"], len(reports), len(copilot["turns"]),
                         ", exported" if export else "")
        return result

    def generate_examples(self, n: int = 2, random_state: int = 4, export: bool = True) -> Path:
        """Full report suites for a spread of test campaigns -> deliverable JSON (+ exported files)."""
        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.run(r.to_dict(), export=export) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved report suites -> %s", self.out_path)
        return self.out_path
