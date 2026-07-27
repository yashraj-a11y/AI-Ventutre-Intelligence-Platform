"""Phase G — Venture Scores (Day-4, honesty tiers).

The platform vision lists many "venture intelligence scores" (funding, growth,
founder strength, market fit, ...). But this dataset has exactly **one** ground
truth: funding success. So :class:`VentureScoreGenerator` splits scores into two
honest tiers instead of emitting fabricated numbers:

**DERIVED** — transparent, monotonic functions of the *calibrated* funding
probability ``p`` (the one validated quantity). No hidden magic:
  - ``funding_probability`` = p
  - ``failure_risk``        = 1 − p
  - ``startup_score``       = round(100·p)            (0–100 rescale)
  - ``investment_score``    = round(100·p)            (same basis, product-facing)
  - ``investment_readiness``= band(p): High / Medium / Low

**DEFERRED** — scores with **no data basis** here (the dataset has no
founder/team/market/business-model/growth-trajectory columns). These return
``value=None, status="deferred"`` with a reason — *never a fabricated number*.
They await Layer 2/3 (retrieval, pattern mining) and richer data:
  - ``founder_strength``, ``market_fit``, ``business_model``, ``scalability``,
    ``growth_potential``

Every returned score carries its ``basis`` so a consumer (Founder/Investor
Report, Copilot) can see exactly how it was produced and never mistakes a
heuristic for a validated model output.

Example
-------
>>> gen = VentureScoreGenerator()
>>> gen.score_campaign(0.81)["startup_score"]
{'value': 81, 'scale': '0-100', 'status': 'derived', 'basis': '100 * funding_probability'}
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.utils.config import load_config
from src.utils.logger import get_logger


class VentureScoreGenerator:
    """Derive honest venture scores from the calibrated funding probability."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.models.venture_scores")

        v = self.config.get("day4", {}).get("venture_scores", {})
        self.readiness_high: float = float(v.get("readiness_high", 0.66))
        self.readiness_medium: float = float(v.get("readiness_medium", 0.40))
        self.deferred: list[str] = list(
            v.get("deferred", ["founder_strength", "market_fit", "business_model",
                               "scalability", "growth_potential"])
        )

    # ------------------------------------------------------------------ public
    def score_campaign(self, p: float) -> dict[str, dict[str, Any]]:
        """Full score report for one campaign, given its calibrated probability p."""
        p = float(np.clip(p, 0.0, 1.0))
        scores: dict[str, dict[str, Any]] = {
            "funding_probability": self._derived(round(p, 4), "0-1", "calibrated model output"),
            "failure_risk": self._derived(round(1 - p, 4), "0-1", "1 - funding_probability"),
            "startup_score": self._derived(int(round(100 * p)), "0-100", "100 * funding_probability"),
            "investment_score": self._derived(int(round(100 * p)), "0-100", "100 * funding_probability"),
            "investment_readiness": self._derived(
                self._readiness(p), "band",
                f"band(p): High>={self.readiness_high}, Medium>={self.readiness_medium}",
            ),
        }
        for name in self.deferred:
            scores[name] = {
                "value": None,
                "status": "deferred",
                "basis": "no data basis in this dataset — awaits Layer 2/3 (retrieval / richer data)",
            }
        return scores

    def score_batch(self, proba) -> dict[str, np.ndarray]:
        """Vectorised derived scores for many campaigns (deferred ones omitted)."""
        p = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
        return {
            "funding_probability": np.round(p, 4),
            "failure_risk": np.round(1 - p, 4),
            "startup_score": np.round(100 * p).astype(int),
            "investment_score": np.round(100 * p).astype(int),
            "investment_readiness": np.array([self._readiness(v) for v in p]),
        }

    # --------------------------------------------------------------- internals
    def _readiness(self, p: float) -> str:
        if p >= self.readiness_high:
            return "High"
        if p >= self.readiness_medium:
            return "Medium"
        return "Low"

    @staticmethod
    def _derived(value, scale: str, basis: str) -> dict[str, Any]:
        return {"value": value, "scale": scale, "status": "derived", "basis": basis}
