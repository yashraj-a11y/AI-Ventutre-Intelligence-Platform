"""Phase B — Counterfactual Candidate Generator (Day-8, Layer 7).

Generate **feasible, bounded** improvement candidates for a campaign — changes to features a
founder can actually control, with target values grounded in Day-7 patterns rather than
arbitrary grids.

Each candidate is a single-feature change described as **raw-input edits** (what to apply to
the raw campaign), so the Phase-C simulator just applies them and re-runs the Day-4 model —
coupled features (usd_goal_real, goal_per_day, duration, timing, title-stats) recompute
consistently through the Day-3 pipeline.

**Mutable features only** (structurally enforced):
- **goal** → the category's *funded-median* goal (Day-7 `funding_patterns.json`) + a moderate
  reduction; only ever proposes **lower** goals (all evidence: smaller goals fund more), floored.
- **campaign_duration** → 15 / 30 days (Day-7: shorter funds better), via ``deadline``.
- **launch_weekday** → Tuesday (Day-7 best), by shifting ``launched`` while preserving duration.
- **title** → add a number (coarse ``has_number`` flip), *suggestive* tier only.

**Immutable** (``country``, ``currency``) are never touched — recommending them would be gaming,
not advice. Fabricated levers (revenue, retention, founder, market…) don't exist and are never
generated.

Example
-------
>>> gen = CandidateGenerator()
>>> cands = gen.generate(raw_campaign)
>>> cands[0].feature, cands[0].proposed, cands[0].tier
('goal', 5000.0, 'strong')
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

_WEEKDAYS = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
             "Friday": 4, "Saturday": 5, "Sunday": 6}


@dataclass
class Candidate:
    """One feasible single-feature change (described as raw-input edits)."""

    feature: str                       # goal | campaign_duration | launch_weekday | title
    current: Any
    proposed: Any
    raw_changes: dict[str, Any]        # edits to apply to the raw campaign
    tier: str                          # strong | moderate | suggestive
    rationale_key: str                 # for Phase-D/F evidence + explanation
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CandidateGenerator:
    """Generate bounded, mutable-only improvement candidates for a campaign.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.cf.candidates")

        c = self.config.get("day8", {}).get("candidates", {})
        self.goal_floor = float(c.get("goal_floor", 100))
        self.goal_multiples = list(c.get("goal_multiples", [0.5]))
        self.target_durations = list(c.get("target_durations", [15, 30]))
        self.best_weekday = str(c.get("best_weekday", "Tuesday"))
        self.enable_title = bool(c.get("enable_title", True))
        self.immutable = set(c.get("immutable", ["country", "currency"]))
        self._funding_path = Path(c.get("funding_patterns_path",
                                        "artifacts/patterns/funding_patterns.json"))
        self._funded_median = self._load_funded_medians()

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any]) -> list[Candidate]:
        """All feasible improvement candidates for one raw campaign."""
        raw = dict(raw)
        cands: list[Candidate] = []
        cands += self._goal_candidates(raw)
        cands += self._duration_candidates(raw)
        cands += self._weekday_candidate(raw)
        if self.enable_title:
            cands += self._title_candidate(raw)
        # Safety: never touch immutable features.
        cands = [c for c in cands if not (set(c.raw_changes) & self.immutable)]
        self.logger.info("Generated %d candidates (%s).", len(cands),
                         ", ".join(sorted({c.feature for c in cands})) or "none")
        return cands

    # --------------------------------------------------------------- internals
    def _goal_candidates(self, raw) -> list[Candidate]:
        goal = float(raw.get("goal", 0.0))
        usd = float(raw.get("usd_goal_real", goal))
        if usd <= 0 or goal <= 0:
            return []
        fx = usd / goal                                   # usd per goal-unit (currency factor)
        main = str(raw.get("main_category"))
        targets: list[tuple[float, str, str]] = []        # (usd target, tier, rationale)

        median = self._funded_median.get(main)
        if median and usd > median * 1.05:                # only if lowering is a real change
            targets.append((median, "strong", "category_funded_median"))
        for m in self.goal_multiples:
            t = usd * m
            if t < usd * 0.95 and (not median or abs(t - median) > median * 0.1):
                targets.append((t, "moderate", f"reduce_to_{int(m*100)}pct"))

        out = []
        seen = set()
        for t_usd, tier, key in targets:
            t_usd = max(self.goal_floor * fx, round(t_usd, 2))   # floor
            if round(t_usd) in seen or t_usd >= usd * 0.98:
                continue
            seen.add(round(t_usd))
            new_goal = round(t_usd / fx, 2) if fx else round(t_usd, 2)
            out.append(Candidate(
                feature="goal", current=round(usd, 2), proposed=round(t_usd, 2),
                raw_changes={"goal": new_goal, "usd_goal_real": round(t_usd, 2)},
                tier=tier, rationale_key=key,
                meta={"unit": "usd_goal_real", "direction": "lower"}))
        return out

    def _duration_candidates(self, raw) -> list[Candidate]:
        launched = pd.to_datetime(raw.get("launched"), errors="coerce")
        deadline = pd.to_datetime(raw.get("deadline"), errors="coerce")
        if pd.isna(launched) or pd.isna(deadline):
            return []
        cur = int((deadline - launched).days)
        out = []
        for d in self.target_durations:
            if abs(d - cur) < 5:                          # skip trivial changes
                continue
            new_deadline = launched + pd.Timedelta(days=d)
            out.append(Candidate(
                feature="campaign_duration", current=cur, proposed=d,
                raw_changes={"deadline": str(new_deadline)},
                tier="moderate", rationale_key="shorter_campaign",
                meta={"unit": "days"}))
        return out

    def _weekday_candidate(self, raw) -> list[Candidate]:
        launched = pd.to_datetime(raw.get("launched"), errors="coerce")
        deadline = pd.to_datetime(raw.get("deadline"), errors="coerce")
        if pd.isna(launched):
            return []
        cur_wd = launched.day_name()
        if cur_wd == self.best_weekday:
            return []
        target = _WEEKDAYS[self.best_weekday]
        shift = (target - launched.weekday()) % 7 or 7    # move forward to next best-weekday
        new_launched = launched + pd.Timedelta(days=shift)
        changes = {"launched": str(new_launched)}
        if not pd.isna(deadline):                         # preserve duration
            changes["deadline"] = str(deadline + pd.Timedelta(days=shift))
        return [Candidate(
            feature="launch_weekday", current=cur_wd, proposed=self.best_weekday,
            raw_changes=changes, tier="suggestive", rationale_key="best_weekday",
            meta={"shift_days": int(shift)})]

    def _title_candidate(self, raw) -> list[Candidate]:
        name = str(raw.get("name", "") or "")
        if any(ch.isdigit() for ch in name):             # already has a number
            return []
        new_name = f"{name} 2".strip()
        changes = {"name": new_name}
        if "name_clean" in raw:
            changes["name_clean"] = f"{str(raw.get('name_clean') or '')} 2".strip()
        return [Candidate(
            feature="title", current="no number", proposed="has number",
            raw_changes=changes, tier="suggestive", rationale_key="title_has_number",
            meta={"note": "coarse has_number flip"})]

    def _load_funded_medians(self) -> dict[str, float]:
        if not self._funding_path.exists():
            self.logger.warning("Funding patterns missing (%s) — goal targets use multiples only.",
                                self._funding_path)
            return {}
        data = json.loads(self._funding_path.read_text())
        return {p["main_category"]: p["funded_median_goal_usd"]
                for p in data.get("category_profiles", [])
                if p.get("funded_median_goal_usd")}
