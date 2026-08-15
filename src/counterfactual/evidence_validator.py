"""Phase D — Historical Evidence Validation (Day-8, Layer 7).

Every recommendation must be backed by **real** historical evidence, not an invented "214
startups with retention > 55%." :class:`EvidenceValidator` computes the *actual* funding rate
of past campaigns that match a recommendation's target value, directly from the Day-7 mining
substrate (the discretized train item frame — the same data the pattern repository was mined
from).

For a recommendation (e.g. "lower your goal to ~$5,000" for a Film campaign) it finds the
matching item (``goal_band = a small goal``), computes confidence / lift / n for **campaigns
in that category with that value** (falling back to all categories if the category cell is too
small), and — for goal — the funded-vs-failed median goal in the category. All real, all from
train.

No faiss/torch — evidence is pure pandas over the item frame, so no OpenMP concern. (Day-6
retrieval could add per-neighbour evidence later; the Day-7 pattern substrate is the primary,
self-contained source.)

Example
-------
>>> ev = EvidenceValidator()
>>> ev.validate({"feature": "goal", "proposed": 5000}, campaign)["statement"]
"In Film & Video, campaigns with a small goal funded at 62% vs a 40% baseline (lift 1.55, n=...)."
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.knowledge_base.venture_database import goal_band
from src.pattern_mining.mining import build_item_frame
from src.utils.config import load_config
from src.utils.logger import get_logger


def duration_band(days: float) -> str:
    """Match the Day-7 duration bins (pd.cut [-1,15,30,45,60])."""
    for hi, label in [(15, "<=15d"), (30, "16-30d"), (45, "31-45d"), (60, "46-60d")]:
        if days <= hi:
            return label
    return ">60d"


class EvidenceValidator:
    """Attach real historical evidence (confidence/lift/n) to recommendations.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    frame, base:
        Optional pre-built Day-7 item frame + base rate (share across calls / the pipeline).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 frame: Optional[pd.DataFrame] = None, base: Optional[float] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.cf.evidence")
        self.min_subgroup_n = int(self.config.get("day8", {}).get("evidence", {}).get("min_subgroup_n", 100))

        if frame is None:
            frame, base = build_item_frame(self.config, self.logger)
        self.frame = frame
        self.base = float(base)

    # ------------------------------------------------------------------ public
    def validate(self, rec: dict[str, Any], campaign: dict[str, Any]) -> dict[str, Any]:
        """Real historical evidence for one recommendation applied to a campaign's context."""
        feature = rec["feature"]
        main = str(campaign.get("main_category"))

        # Goal uses a THRESHOLD reference class ("campaigns asking <= your proposed goal"),
        # not the coarse goal_band — so $5k and $19k give different, precise evidence.
        if feature == "goal":
            proposed = float(rec.get("proposed"))
            item_label = f"goal <= ${proposed:,.0f}"
            cat_mask = (self.frame["main_category"] == main) & (self.frame["usd_goal_real"] <= proposed)
            glob_col = self.frame["usd_goal_real"] <= proposed
        else:
            item = self._target_item(feature, rec.get("proposed"))
            if item is None:
                return {"available": False, "reason": f"no evidence mapping for {feature}"}
            col, val = item
            item_label = f"{col}={val}"
            cat_mask = (self.frame["main_category"] == main) & (self.frame[col].astype(str) == str(val))
            glob_col = self.frame[col].astype(str) == str(val)

        # Prefer category-specific evidence; fall back to global if the cell is too small.
        n, conf, lift = self._stat(cat_mask)
        scope = f"{main} campaigns"
        if n < self.min_subgroup_n:
            n, conf, lift = self._stat(glob_col)
            scope = "campaigns overall"

        ev = {
            "available": n > 0,
            "source": "pattern_mining (Day-7 train substrate)",
            "scope": scope,
            "item": item_label,
            "n": int(n),
            "confidence": round(conf, 4),
            "lift": round(lift, 3),
            "base_rate": round(self.base, 4),
        }
        if feature == "goal":
            ev.update(self._goal_medians(main))
        ev["statement"] = self._statement(feature, scope, item_label, conf, lift, n, ev)
        return ev

    def validate_all(self, recs: list[dict[str, Any]],
                     campaign: dict[str, Any]) -> list[dict[str, Any]]:
        """Attach evidence to a list of recommendations (returns rec + 'evidence')."""
        out = []
        for r in recs:
            out.append({**r, "evidence": self.validate(r, campaign)})
        return out

    # --------------------------------------------------------------- internals
    def _target_item(self, feature: str, proposed: Any) -> Optional[tuple[str, Any]]:
        if feature == "goal":
            return ("goal_band", goal_band(float(proposed)))
        if feature == "campaign_duration":
            return ("duration_band", duration_band(float(proposed)))
        if feature == "launch_weekday":
            return ("launch_weekday", str(proposed))
        if feature == "title":
            return ("has_number", "has number")
        return None

    def _stat(self, mask) -> tuple[int, float, float]:
        sub = self.frame[mask]
        n = int(len(sub))
        if n == 0:
            return 0, 0.0, 0.0
        conf = float(sub["target"].mean())
        return n, conf, conf / self.base if self.base else 0.0

    def _goal_medians(self, main: str) -> dict[str, Any]:
        sub = self.frame[self.frame["main_category"] == main]
        funded = sub.loc[sub["target"] == 1, "usd_goal_real"]
        failed = sub.loc[sub["target"] == 0, "usd_goal_real"]
        return {
            "funded_median_goal_usd": round(float(funded.median()), 2) if len(funded) else None,
            "failed_median_goal_usd": round(float(failed.median()), 2) if len(failed) else None,
        }

    def _statement(self, feature, scope, item_label, conf, lift, n, ev) -> str:
        base_pct = round(self.base * 100)
        if feature == "goal":
            lead = f"{scope} asking {item_label}"
        elif feature == "campaign_duration":
            lead = f"{scope} with a {item_label.split('=', 1)[1]} campaign"
        elif feature == "launch_weekday":
            lead = f"{scope} launched on {item_label.split('=', 1)[1]}"
        elif feature == "title":
            lead = f"{scope} with a number in the title"
        else:
            lead = f"{scope} with {item_label}"
        core = (f"{lead} funded at {conf:.0%} vs a {base_pct}% baseline "
                f"(lift {lift:.2f}, n={n:,}).")
        if feature == "goal" and ev.get("funded_median_goal_usd") is not None:
            core += (f" Funded campaigns here had a median goal of "
                     f"${ev['funded_median_goal_usd']:,.0f} vs "
                     f"${ev['failed_median_goal_usd']:,.0f} for failed.")
        return core
