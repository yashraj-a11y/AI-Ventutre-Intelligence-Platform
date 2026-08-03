"""Phase F — Category Competition Analysis (Day-9, Layer 8).

The honest "competitor intelligence" reframe. We have **no named competitors** — so we don't
invent "OpenAI / Microsoft". The real competition for a Kickstarter campaign is **the other
campaigns in its category**, and we can say useful, grounded things about them:

- **Competitive intensity** — how crowded the category is (volume + share).
- **What winners do differently** — funded-vs-failed norms (goal, duration, goal band) — the
  data-grounded "competitive advantage / gap".
- **Your position** — where this campaign sits vs the funded cohort (positioned like the winners,
  or like the strugglers?).

**Deferred** (no data): named competitors, moat / defensibility, proprietary-data / network-effects
analysis.

Pure category statistics over the Day-7 item frame — no prediction engine, no faiss.

Example
-------
>>> eng = CompetitionAnalysisEngine()
>>> a = eng.analyze(raw_campaign)
>>> a["your_position"]["positioned_like"]
'strugglers'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.evidence_validator import duration_band
from src.knowledge_base.venture_database import goal_band
from src.pattern_mining.mining import build_item_frame
from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED = {
    "named_competitors": {"status": "deferred", "reason": "no competitor data in this dataset"},
    "moat_defensibility": {"status": "deferred", "reason": "no product/IP data"},
    "proprietary_data_network_effects": {"status": "deferred", "reason": "no such data"},
}
_CAPTION = ("Competition = other campaigns in your category (our own data). Named competitors, "
            "moat, and defensibility are deferred — no data.")


class CompetitionAnalysisEngine:
    """Own-data category competition: crowding, winner norms, the campaign's position.

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
        self.logger = logger or get_logger("avip.copilot.competition")

        c = self.config.get("day9", {}).get("competition", {})
        self.crowd_hi = int(c.get("crowded_high", 25000))
        self.crowd_mid = int(c.get("crowded_mid", 12000))
        self.crowd_lo = int(c.get("crowded_low", 5000))
        self.struggler_ratio = float(c.get("struggler_goal_ratio", 1.3))
        self.out_path = Path(c.get("out_path", "artifacts/copilot/competitor_analysis.json"))

        if frame is None:
            frame, base = build_item_frame(self.config, self.logger)
        self.frame = frame
        self.base = float(base)

    # ------------------------------------------------------------------ public
    def analyze(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Category competition analysis for one campaign."""
        raw = dict(raw)
        main = str(raw.get("main_category"))
        goal = float(raw.get("usd_goal_real", 0.0))
        launched = pd.to_datetime(raw.get("launched"), errors="coerce")
        deadline = pd.to_datetime(raw.get("deadline"), errors="coerce")
        dur = int((deadline - launched).days) if not (pd.isna(launched) or pd.isna(deadline)) else None

        cat = self.frame[self.frame["main_category"] == main]
        n = len(cat)
        funded = cat[cat.target == 1]
        failed = cat[cat.target == 0]
        f_goal = float(funded["usd_goal_real"].median()) if len(funded) else None
        x_goal = float(failed["usd_goal_real"].median()) if len(failed) else None

        return {
            "campaign": {k: raw.get(k) for k in ("name", "main_category", "usd_goal_real")},
            "category": main,
            "competitive_intensity": {
                "level": self._crowding(n),
                "n_campaigns": n,
                "share_of_all": round(n / len(self.frame), 4),
            },
            "what_winners_do": self._winner_norms(funded, failed, f_goal, x_goal),
            "your_position": self._position(goal, dur, f_goal, x_goal, funded),
            "deferred": _DEFERRED,
            "honesty_note": _CAPTION,
        }

    def generate_examples(self, n: int = 3, random_state: int = 3) -> Path:
        """Competition analyses for a spread of test campaigns -> deliverable JSON."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = [self.analyze(r.to_dict()) for _, r in sample.reset_index(drop=True).iterrows()]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved competition analyses -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _winner_norms(self, funded, failed, f_goal, x_goal) -> list[str]:
        norms = []
        if f_goal is not None and x_goal is not None:
            norms.append(f"Ask for less: funded campaigns' median goal is ${f_goal:,.0f} "
                         f"vs ${x_goal:,.0f} for those that failed.")
        f_dur = self._mode(funded["duration_band"]) if len(funded) else None
        x_dur = self._mode(failed["duration_band"]) if len(failed) else None
        if f_dur and f_dur != x_dur:
            norms.append(f"Favor {f_dur} campaigns (winners' most common length; losers lean {x_dur}).")
        return norms

    def _position(self, goal, dur, f_goal, x_goal, funded) -> dict[str, Any]:
        like = "unknown"
        note = []
        if f_goal:
            if goal > f_goal * self.struggler_ratio:
                like = "strugglers"
                note.append(f"Your ${goal:,.0f} goal is {goal/f_goal:.1f}x the winners' median "
                            f"(${f_goal:,.0f}) — priced like the campaigns that failed.")
            else:
                like = "winners"
                note.append(f"Your ${goal:,.0f} goal is near the winners' median (${f_goal:,.0f}).")
        if dur is not None and len(funded):
            f_dur = self._mode(funded["duration_band"])
            db = duration_band(dur)
            if f_dur and db != f_dur:
                note.append(f"Your {db} length differs from winners' typical {f_dur}.")
        return {"goal_usd": round(goal, 2),
                "winners_median_goal_usd": round(f_goal, 2) if f_goal else None,
                "positioned_like": like, "notes": note}

    def _crowding(self, n: int) -> str:
        return ("Highly crowded" if n >= self.crowd_hi else
                "Crowded" if n >= self.crowd_mid else
                "Moderate" if n >= self.crowd_lo else "Niche")

    @staticmethod
    def _mode(s: pd.Series) -> Optional[str]:
        vc = s.value_counts()
        return str(vc.index[0]) if len(vc) else None
