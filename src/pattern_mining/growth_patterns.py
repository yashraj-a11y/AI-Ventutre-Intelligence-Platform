"""Phase E — Temporal / Timing Patterns (Day-7, Layer 6).

**Honest reframe of "growth pattern mining."** Kickstarter records a *single* outcome
(funded/failed), not a growth trajectory — there is no user-growth, revenue-growth, or
retention time series to mine. So true "growth patterns" are **deferred**, and this engine
mines the timing patterns we *can* measure honestly:

- **Year trend** — how the funding rate moved across ``launch_year`` (the real "market
  timing" signal: Kickstarter's success rate shifted as the platform matured/saturated).
- **Seasonality** — funding rate by ``launch_month`` / ``launch_quarter`` / ``launch_weekday``
  and the **weekend effect**.

Each timing value gets success rate + lift (vs the 0.4038 base) + support. Real
associations, not causes.

Example
-------
>>> eng = TemporalPatternEngine()
>>> out = eng.analyze()
>>> out["highlights"]["best_weekday"], out["deferred"]["growth_trajectory"]["status"]
('Tuesday', 'deferred')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.pattern_mining.mining import build_item_frame
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED = {
    "growth_trajectory": {"status": "deferred",
                          "reason": ("Kickstarter records one funded/failed outcome, not a "
                                     "time series — no user/revenue/retention growth to mine")},
    "user_growth": {"status": "deferred", "reason": "no data"},
    "revenue_growth": {"status": "deferred", "reason": "no data"},
    "retention": {"status": "deferred", "reason": "no data"},
    "team_expansion": {"status": "deferred", "reason": "no data"},
    "product_iteration": {"status": "deferred", "reason": "no data"},
}


class TemporalPatternEngine:
    """Mine launch-timing patterns (year trend + seasonality); defer true growth.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.patterns.temporal")

        c = self.config.get("day7", {}).get("temporal", {})
        self.min_support_count = int(c.get("min_support_count", 300))
        self.out_path = Path(c.get("out_path", "artifacts/patterns/growth_patterns.json"))

        self.base_rate_: Optional[float] = None
        self.result_: dict[str, Any] = {}

    # ------------------------------------------------------------------ public
    def analyze(self, frame: Optional[pd.DataFrame] = None,
                base: Optional[float] = None) -> dict[str, Any]:
        """Compute year trend + seasonality dimensions + highlights."""
        if frame is None:
            frame, base = build_item_frame(self.config, self.logger)
        self.base_rate_ = base

        year = self._dimension(frame, base, "launch_year", sort="value")
        month = self._dimension(frame, base, "launch_month", sort="value")
        quarter = self._dimension(frame, base, "launch_quarter", sort="value")
        weekday = self._dimension(frame, base, "launch_weekday", sort="lift")
        weekend = self._dimension(frame, base, "is_weekend_launch", sort="lift")

        self.result_ = {
            "pattern_type": "temporal",
            "base_rate": round(base, 4),
            "note": ("Launch-timing patterns (real). True growth trajectories are deferred — "
                     "the dataset has one outcome per campaign, not a time series. "
                     "Historical associations, not causes."),
            "year_trend": year,
            "month": month,
            "quarter": quarter,
            "weekday": weekday,
            "weekend": weekend,
            "highlights": self._highlights(year, month, quarter, weekday, weekend),
            "deferred": _DEFERRED,
        }
        self.logger.info("Temporal patterns: %d years, %d months; best weekday %s.",
                         len(year), len(month),
                         self.result_["highlights"].get("best_weekday"))
        return self.result_

    def save(self) -> Path:
        """Persist the temporal patterns as JSON."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(self.result_, indent=2))
        self.logger.info("Saved temporal patterns -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _dimension(self, frame: pd.DataFrame, base: float, col: str,
                   sort: str = "value") -> list[dict[str, Any]]:
        """Success rate + lift + support for each value of a timing feature."""
        g = frame.groupby(col, observed=True)["target"].agg(["mean", "size"])
        g = g[(g["size"] >= self.min_support_count) & (g.index.astype(str) != "<NA>")]
        rows = [{"value": str(v), "success_rate": round(float(r["mean"]), 4),
                 "lift": round(float(r["mean"] / base), 3), "n": int(r["size"])}
                for v, r in g.iterrows()]
        if sort == "lift":
            rows.sort(key=lambda x: x["lift"], reverse=True)
        else:
            rows.sort(key=lambda x: x["value"])
        return rows

    @staticmethod
    def _highlights(year, month, quarter, weekday, weekend) -> dict[str, Any]:
        def best(rows):
            return max(rows, key=lambda x: x["lift"])["value"] if rows else None
        def worst(rows):
            return min(rows, key=lambda x: x["lift"])["value"] if rows else None

        h = {"best_year": best(year), "worst_year": worst(year),
             "best_month": best(month), "worst_month": worst(month),
             "best_quarter": best(quarter), "best_weekday": best(weekday),
             "worst_weekday": worst(weekday)}
        if year:
            h["year_trend_direction"] = ("declining" if year[-1]["lift"] < year[0]["lift"]
                                         else "rising")
        wd = {r["value"]: r["lift"] for r in weekend}
        if "weekend" in wd and "weekday" in wd:
            h["weekend_vs_weekday_lift"] = round(wd["weekend"] - wd["weekday"], 3)
        return h
