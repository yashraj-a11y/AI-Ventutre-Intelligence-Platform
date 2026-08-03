"""Phase B — Funding-Readiness Stage Detection (Day-10, Layer 9).

The honest reframe of "startup stage." We can't observe an MVP/Validation/PMF lifecycle in a
single pre-launch Kickstarter snapshot — so instead of fabricating one, :class:`StageDetector`
classifies a **funding-readiness stage** derived from the model itself:

- **Not ready** — even the best controllable changes leave it below the funding bar (the
  fundamentals, e.g. category, likely need to change).
- **Fixable** — below the bar as-is, but controllable changes could clear it before launch.
- **Ready** — already above the funding bar.
- **Strong** — comfortably above the bar.

Uses three real numbers: the calibrated probability (Day-4), the decision threshold (Day-4
F1-optimal), and the **optimized** probability (the Day-8 counterfactual's reachable optimum
over controllable features). Startup lifecycle stages (Idea/MVP/PMF/Series A) are **deferred** —
the data can't observe them.

Reuses the Day-8 :class:`CounterfactualPipeline`; no faiss → no OpenMP.

Example
-------
>>> det = StageDetector()
>>> s = det.detect(raw_campaign)
>>> s["stage"], s["confidence"]
('Fixable', 0.7)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.pipeline import CounterfactualPipeline
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED_STAGES = {
    "startup_lifecycle": {"status": "deferred",
                          "reason": ("Idea/MVP/Validation/PMF/Growth/Series-A require observing a "
                                     "company over time; the dataset is a single pre-launch snapshot")}
}
_CAPTION = ("Funding-readiness stage (model-derived from the calibrated probability + the "
            "counterfactual's reachable optimum) — not a startup lifecycle stage, and not a "
            "causal guarantee.")


class StageDetector:
    """Classify a campaign's funding-readiness stage.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    cf_pipeline:
        Optional pre-built Day-8 :class:`CounterfactualPipeline` (share the warm stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 cf_pipeline: Optional[CounterfactualPipeline] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.evolution.stage")

        c = self.config.get("day10", {}).get("stage", {})
        self.strong_margin = float(c.get("strong_margin", 0.15))
        self.out_path = Path(c.get("out_path", "artifacts/evolution/startup_stage.json"))

        self.cf = cf_pipeline or CounterfactualPipeline(self.config, self.logger)

    # ------------------------------------------------------------------ public
    def detect(self, raw: dict[str, Any],
               analysis: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Classify the funding-readiness stage for one campaign."""
        raw = dict(raw)
        analysis = analysis if analysis is not None else self.cf.analyze(raw)
        base_p = analysis["baseline_p"]
        optimized_p = analysis["optimization_plan"].get("final_p", base_p)
        threshold = float(self.cf.simulator.engine.threshold)

        stage, margin, reason = self._classify(base_p, optimized_p, threshold)
        return {
            "campaign": {k: raw.get(k) for k in ("name", "main_category", "usd_goal_real")},
            "funding_probability": round(base_p, 4),
            "optimized_probability": round(optimized_p, 4),
            "reachable_gain": round(optimized_p - base_p, 4),
            "threshold": round(threshold, 4),
            "stage": stage,
            "confidence": self._confidence(margin),
            "reason": reason,
            "deferred_stages": _DEFERRED_STAGES,
            "honesty_note": _CAPTION,
        }

    def stage_of(self, base_p: float, optimized_p: float) -> str:
        """Just the readiness-stage label for a (probability, reachable-optimum) pair."""
        threshold = float(self.cf.simulator.engine.threshold)
        return self._classify(base_p, optimized_p, threshold)[0]

    def generate_examples(self, n: int = 4, random_state: int = 7) -> Path:
        """Stage detection for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti].sample(n, random_state=random_state)
        out = [self.detect(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved stage detections -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _classify(self, base_p, optimized_p, threshold) -> tuple[str, float, str]:
        t_pct, o_pct, b_pct = round(threshold * 100), round(optimized_p * 100), round(base_p * 100)
        if base_p >= threshold + self.strong_margin:
            return ("Strong", base_p - (threshold + self.strong_margin),
                    f"The model estimates {b_pct}% funding probability — comfortably above the "
                    f"{t_pct}% bar.")
        if base_p >= threshold:
            return ("Ready", base_p - threshold,
                    f"The model estimates {b_pct}% — already above the {t_pct}% funding bar.")
        if optimized_p >= threshold:
            return ("Fixable", optimized_p - threshold,
                    f"As-is the model estimates {b_pct}% (below the {t_pct}% bar), but controllable "
                    f"changes could reach ~{o_pct}% — above the bar. Fixable before launch.")
        return ("Not ready", threshold - optimized_p,
                f"Even with the best controllable changes the model estimates ~{o_pct}% — still "
                f"below the {t_pct}% bar. The fundamentals (e.g. category, positioning) likely "
                f"need to change.")

    @staticmethod
    def _confidence(margin: float) -> float:
        """Confidence from distance to the decisive boundary (clipped 0.5–0.99)."""
        return round(min(0.99, 0.5 + min(abs(margin), 0.4) * 1.2), 2)
