"""Phase D — Funding Pattern Discovery (Day-7, Layer 6).

Where B/C ask "what funds / fails," Phase D asks the **funding-specific, quantitative**
question: *given a category, how much should a campaign ask for?* :class:`FundingPatternAnalyzer`
produces three real, actionable views:

1. **Category funding profiles** — per ``main_category``: success rate + lift, and the
   **median goal of funded vs failed** campaigns (the raw dollar gap, not just a band), plus
   the best-performing goal band within that category.
2. **Goal thresholds** — the funded-vs-failed median-goal gap gives an actionable target
   ("in Games, winners asked a median $X; losers $Y").
3. **Top funding combos** — the strongest 2-feature drivers (largest |lift − 1|) from the
   shared miner, in both directions.

**Honesty tiers** (same as Days 4–6): funding **rounds, stage, team composition, revenue,
business model, market size** have no data → returned as explicit ``deferred`` placeholders,
never invented. Everything reported is a historical association over real fields.

Example
-------
>>> fa = FundingPatternAnalyzer()
>>> out = fa.analyze()
>>> out["category_profiles"][0]["main_category"], out["category_profiles"][0]["funded_median_goal_usd"]
('Dance', 3000.0)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.pattern_mining.mining import Rule, build_item_frame, mine_rules
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED = {
    b: {"status": "deferred", "reason": "no data in this dataset — awaits richer venture data"}
    for b in ["funding_rounds", "funding_stage", "team_composition",
              "revenue", "business_model", "market_size"]
}


class FundingPatternAnalyzer:
    """Category goal-threshold profiles + strongest funding combos (real fields only).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.patterns.funding")

        m = self.config.get("day7", {}).get("mining", {})
        self.min_support_count = int(m.get("min_support_count", 500))
        self.single_features = list(m.get("single_features", []))
        self.pair_features = list(m.get("pair_features", []))

        c = self.config.get("day7", {}).get("funding", {})
        self.category_min_support = int(c.get("category_min_support", 100))
        self.top_combos = int(c.get("top_combos", 15))
        self.out_path = Path(c.get("out_path", "artifacts/patterns/funding_patterns.json"))

        self.base_rate_: Optional[float] = None
        self.result_: dict[str, Any] = {}

    # ------------------------------------------------------------------ public
    def analyze(self, frame: Optional[pd.DataFrame] = None, base: Optional[float] = None,
                rules: Optional[list[Rule]] = None) -> dict[str, Any]:
        """Compute category funding profiles + top funding combos."""
        if frame is None:
            frame, base = build_item_frame(self.config, self.logger)
        if rules is None:
            rules = mine_rules(frame, base, self.single_features,
                               self.pair_features, self.min_support_count)
        self.base_rate_ = base

        self.result_ = {
            "pattern_type": "funding",
            "base_rate": round(base, 4),
            "note": ("Per-category goal thresholds + strongest 2-feature funding drivers, "
                     "over real fields only. Historical associations, not causes."),
            "category_profiles": self._category_profiles(frame, base),
            "top_funding_combos": self._top_combos(rules),
            "deferred": _DEFERRED,
        }
        self.logger.info("Funding patterns: %d category profiles, %d top combos.",
                         len(self.result_["category_profiles"]),
                         len(self.result_["top_funding_combos"]))
        return self.result_

    def save(self) -> Path:
        """Persist the funding patterns as JSON."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(self.result_, indent=2))
        self.logger.info("Saved funding patterns -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _category_profiles(self, frame: pd.DataFrame, base: float) -> list[dict[str, Any]]:
        profiles = []
        for cat, sub in frame.groupby("main_category", observed=True):
            n = len(sub)
            sr = float(sub["target"].mean())
            funded = sub.loc[sub["target"] == 1, "usd_goal_real"]
            failed = sub.loc[sub["target"] == 0, "usd_goal_real"]
            # best goal band within this category (min support per cell)
            band = sub.groupby("goal_band", observed=True)["target"].agg(["mean", "count"])
            band = band[band["count"] >= self.category_min_support]
            best_band = band["mean"].idxmax() if len(band) else None
            profiles.append({
                "main_category": cat,
                "n": n,
                "success_rate": round(sr, 4),
                "lift": round(sr / base, 3),
                "funded_median_goal_usd": round(float(funded.median()), 2) if len(funded) else None,
                "failed_median_goal_usd": round(float(failed.median()), 2) if len(failed) else None,
                "best_goal_band": best_band,
                "best_goal_band_success_rate": round(float(band["mean"].max()), 4) if len(band) else None,
            })
        profiles.sort(key=lambda p: p["lift"], reverse=True)
        return profiles

    def _top_combos(self, rules: list[Rule]) -> list[dict[str, Any]]:
        pairs = [r for r in rules if r.order == 2 and r.p_value <= 0.05]
        pairs.sort(key=lambda r: abs(r.lift - 1.0), reverse=True)   # strongest drivers, either way
        return [{**r.to_dict(), "driver": r.direction} for r in pairs[: self.top_combos]]
