"""Phase E — Category & Segment Analysis (Day-9, Layer 8).

The honest "market intelligence" reframe. We have **no external market data** (TAM/SAM/SOM,
CAGR), so we don't invent it. Instead we read the campaign's **category as a segment within
Kickstarter**, entirely from our own train data:

- **funding rate + lift** — how the segment performs vs the 40.4% baseline.
- **year-trend** — how the segment's funding rate moved over time (the Day-7 saturation story)
  → the honest "market timing" signal.
- **crowding / volume** — how many campaigns compete in the segment.
- **goal norms** — funded vs failed median goal in the segment.

From those we derive a **market_opportunity** (favorable / neutral / challenging, from lift) and
**market_timing** (from the trend). TAM/SAM/SOM/CAGR and external market data are **deferred**.

Pure category statistics over the Day-7 item frame — no prediction engine, no faiss.

Example
-------
>>> eng = CategoryAnalysisEngine()
>>> a = eng.analyze(raw_campaign)
>>> a["market_opportunity"], a["market_timing"]
('Challenging', 'Unfavorable')
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
    "tam_sam_som": {"status": "deferred", "reason": "no market-size data"},
    "market_cagr_growth": {"status": "deferred", "reason": "no external growth data"},
    "external_market_data": {"status": "deferred", "reason": "dataset is Kickstarter only"},
}
_CAPTION = ("A within-Kickstarter category read from our own data — NOT external market size, "
            "CAGR, or industry analysis (those are deferred: no data).")


class CategoryAnalysisEngine:
    """Own-data category/segment analysis (funding rate, trend, crowding, goal norms).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    frame, base:
        Optional pre-built Day-7 item frame + base rate (share across the pipeline).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 frame: Optional[pd.DataFrame] = None, base: Optional[float] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.copilot.market")

        c = self.config.get("day9", {}).get("market", {})
        self.min_year_n = int(c.get("min_year_n", 100))
        self.crowd_hi = int(c.get("crowded_high", 25000))
        self.crowd_mid = int(c.get("crowded_mid", 12000))
        self.crowd_lo = int(c.get("crowded_low", 5000))
        self.favorable_lift = float(c.get("favorable_lift", 1.1))
        self.challenging_lift = float(c.get("challenging_lift", 0.9))
        self.out_path = Path(c.get("out_path", "artifacts/copilot/market_analysis.json"))

        if frame is None:
            frame, base = build_item_frame(self.config, self.logger)
        self.frame = frame
        self.base = float(base)

    # ------------------------------------------------------------------ public
    def analyze(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Category/segment analysis for one campaign."""
        raw = dict(raw)
        main = str(raw.get("main_category"))
        cat = self.frame[self.frame["main_category"] == main]
        n = len(cat)
        rate = float(cat["target"].mean()) if n else self.base
        lift = rate / self.base
        funded = cat.loc[cat.target == 1, "usd_goal_real"]
        failed = cat.loc[cat.target == 0, "usd_goal_real"]

        by_year, direction = self._year_trend(cat)
        opportunity = self._opportunity(lift)
        timing = self._timing(direction)

        return {
            "campaign": {k: raw.get(k) for k in ("name", "main_category", "usd_goal_real")},
            "category": main,
            "category_funding_rate": round(rate, 4),
            "category_lift": round(lift, 3),
            "n_campaigns": n,
            "market_opportunity": opportunity,
            "crowding": self._crowding(n),
            "year_trend": {"direction": direction, "by_year": by_year},
            "market_timing": timing,
            "goal_norms": {
                "funded_median_goal_usd": round(float(funded.median()), 2) if len(funded) else None,
                "failed_median_goal_usd": round(float(failed.median()), 2) if len(failed) else None,
            },
            "summary": self._summary(main, rate, lift, n, direction, timing, opportunity),
            "deferred": _DEFERRED,
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 3) -> Path:
        """Category analyses for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti].sample(n, random_state=random_state)
        out = [self.analyze(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved category analyses -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _year_trend(self, cat: pd.DataFrame) -> tuple[list[dict[str, Any]], str]:
        g = cat.groupby("launch_year", observed=True)["target"].agg(["mean", "size"])
        g = g[(g["size"] >= self.min_year_n) & (g.index.astype(str) != "<NA>")].sort_index()
        rows = [{"year": str(y), "funding_rate": round(float(r["mean"]), 4),
                 "lift": round(float(r["mean"] / self.base), 3), "n": int(r["size"])}
                for y, r in g.iterrows()]
        if len(rows) < 2:
            return rows, "insufficient-data"
        early, late = rows[0]["lift"], rows[-1]["lift"]
        direction = ("declining" if late < early * 0.95 else
                     "rising" if late > early * 1.05 else "stable")
        return rows, direction

    def _opportunity(self, lift: float) -> str:
        return ("Favorable" if lift >= self.favorable_lift else
                "Challenging" if lift < self.challenging_lift else "Neutral")

    def _timing(self, direction: str) -> str:
        return {"declining": "Unfavorable", "rising": "Favorable",
                "stable": "Neutral"}.get(direction, "Unknown")

    def _crowding(self, n: int) -> str:
        return ("Highly crowded" if n >= self.crowd_hi else
                "Crowded" if n >= self.crowd_mid else
                "Moderate" if n >= self.crowd_lo else "Niche")

    def _summary(self, main, rate, lift, n, direction, timing, opportunity) -> str:
        trend_txt = {"declining": "declined over time", "rising": "risen over time",
                     "stable": "held steady"}.get(direction, "an unclear trend")
        return (f"The {main} segment has a {rate:.0%} historical funding rate (lift {lift:.2f}, "
                f"{opportunity.lower()} opportunity), across {n:,} campaigns "
                f"({self._crowding(n).lower()}). Funding rates have {trend_txt}, making market "
                f"timing {timing.lower()}. (Within-Kickstarter read; external market size/CAGR "
                f"not available.)")
