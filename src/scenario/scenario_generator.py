"""Phase E — Scenario Generator (Day-10, Layer 10).

Generate named multi-variable **what-if scenarios** — cumulative combinations of the
controllable changes (goal / duration / launch timing / title) — for a founder to compare.
The genuinely new capability over Day-8 (one greedy optimum): *discrete strategies* whose
marginal value can be weighed.

**The canonical composition (mandatory).** The controllable changes touch overlapping *raw*
fields — both "shorten duration" and "launch Tuesday" edit ``deadline`` — so naively merging
Day-8's independently-computed ``raw_changes`` **double-writes ``deadline``** and corrupts the
result (verified: a scenario *regressed* 0.375 → 0.059). :func:`compose_scenario` instead applies
**semantic intents in dependency order** — goal → shift ``launched`` to the target weekday → set
``deadline`` = shifted-launched + duration → title — recomputing derived fields consistently
(verified: 0.384, monotone). Every scenario is composed this way.

Target values are drawn from the Day-8 :class:`CandidateGenerator` (goal → category
funded-median, etc.), so scenarios are grounded, not arbitrary. Real controllable variables
only — no revenue/retention/founder scenarios.

Example
-------
>>> gen = ScenarioGenerator()
>>> scen = gen.generate(raw_campaign)
>>> [s["name"] for s in scen["scenarios"]]
['Current (as-is)', 'Lower goal', 'Lower goal + shorten duration', ...]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.counterfactual.candidate_generator import CandidateGenerator
from src.utils.config import load_config
from src.utils.logger import get_logger

_WEEKDAYS = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
             "Friday": 4, "Saturday": 5, "Sunday": 6}
# Cumulative order (goal first — the dominant lever) and readable labels.
_ORDER = ["goal_usd", "duration_days", "weekday", "add_number"]
_LABEL = {"goal_usd": "Lower goal", "duration_days": "shorten duration",
          "weekday": "Tuesday launch", "add_number": "add title number"}


def compose_scenario(raw: dict[str, Any], goal_usd: Optional[float] = None,
                     duration_days: Optional[int] = None, weekday: Optional[str] = None,
                     add_number: bool = False) -> dict[str, Any]:
    """Apply controllable changes in CANONICAL dependency order (recompute derived fields).

    Order: goal (independent) → shift ``launched`` to target weekday → derive ``deadline`` from
    the (possibly shifted) ``launched`` + duration → title. This is the verified fix for the
    naive-merge ``deadline`` double-write.
    """
    c = dict(raw)
    orig_launched = pd.to_datetime(c.get("launched"), errors="coerce")
    orig_deadline = pd.to_datetime(c.get("deadline"), errors="coerce")
    launched = orig_launched

    if goal_usd is not None:                                  # 1. goal (independent)
        g = float(c.get("goal", 0.0)); usd = float(c.get("usd_goal_real", g))
        fx = usd / g if g else 1.0
        c["usd_goal_real"] = round(float(goal_usd), 2)
        c["goal"] = round(float(goal_usd) / fx, 2) if fx else round(float(goal_usd), 2)

    if weekday is not None and not pd.isna(launched):         # 2. shift launched
        shift = (_WEEKDAYS[weekday] - launched.weekday()) % 7 or 7
        launched = launched + pd.Timedelta(days=shift)
        c["launched"] = str(launched)

    if not pd.isna(launched):                                 # 3. derive deadline (from current launched)
        dur = (duration_days if duration_days is not None
               else int((orig_deadline - orig_launched).days) if not pd.isna(orig_deadline) else 30)
        c["deadline"] = str(launched + pd.Timedelta(days=dur))

    if add_number and not any(ch.isdigit() for ch in str(c.get("name", ""))):  # 4. title
        c["name"] = f"{c.get('name', '')} 2".strip()
        if "name_clean" in c:
            c["name_clean"] = f"{c.get('name_clean', '')} 2".strip()
    return c


class ScenarioGenerator:
    """Generate named cumulative scenarios over controllable variables (canonically composed).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    candidate_generator:
        Optional pre-built Day-8 :class:`CandidateGenerator` (for target values).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 candidate_generator: Optional[CandidateGenerator] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.scenario.generator")

        c = self.config.get("day10", {}).get("scenario", {})
        self.duration_target = int(c.get("duration_target", 30))
        self.out_path = Path(c.get("out_path", "artifacts/scenario/scenarios.json"))

        self.gen = candidate_generator or CandidateGenerator(self.config, self.logger)

    # ------------------------------------------------------------------ public
    def generate(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Named cumulative scenarios (each a composed campaign ready to simulate)."""
        raw = dict(raw)
        targets = self._targets(raw)
        present = [(k, targets[k]) for k in _ORDER if k in targets]

        scenarios = [{"name": "Current (as-is)", "changes": {},
                      "campaign": dict(raw), "description": "no changes"}]
        cum: dict[str, Any] = {}
        for k, v in present:
            cum[k] = v
            name = " + ".join(_LABEL[kk] for kk in cum)
            name = name[0].upper() + name[1:]
            scenarios.append({
                "name": name,
                "changes": dict(cum),
                "campaign": compose_scenario(raw, **cum),
                "description": self._describe(cum, raw),
            })
        self.logger.info("Generated %d scenarios (%d controllable variables).",
                         len(scenarios), len(present))
        return {"campaign": {k: raw.get(k) for k in ("name", "main_category", "usd_goal_real")},
                "n_scenarios": len(scenarios), "scenarios": scenarios,
                "note": ("Multi-variable what-if scenarios over controllable features only, "
                         "composed canonically. Model-simulated (Phase F), not causal.")}

    def generate_examples(self, n: int = 3, random_state: int = 4) -> Path:
        """Scenarios for a spread of improvable test campaigns -> deliverable JSON (compact)."""
        import numpy as np

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        sample = df.iloc[ti][df.iloc[ti]["usd_goal_real"] > 12000].sample(n, random_state=random_state)
        out = []
        for _, r in sample.reset_index(drop=True).iterrows():
            g = self.generate(r.to_dict())
            # compact: drop the full composed campaign from the saved artifact
            g["scenarios"] = [{"name": s["name"], "changes": s["changes"],
                               "description": s["description"]} for s in g["scenarios"]]
            out.append(g)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2, default=str))
        self.logger.info("Saved scenarios -> %s", self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    def _targets(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Target values for each controllable variable, from Day-8 candidates."""
        cands = self.gen.generate(raw)
        by_feat: dict[str, list] = {}
        for c in cands:
            by_feat.setdefault(c.feature, []).append(c)
        t: dict[str, Any] = {}
        if "goal" in by_feat:                                  # strong (category median) if present
            gc = next((c for c in by_feat["goal"] if c.tier == "strong"), by_feat["goal"][0])
            t["goal_usd"] = gc.proposed
        if "campaign_duration" in by_feat:                     # prefer the configured target length
            dc = next((c for c in by_feat["campaign_duration"] if c.proposed == self.duration_target),
                      by_feat["campaign_duration"][0])
            t["duration_days"] = dc.proposed
        if "launch_weekday" in by_feat:
            t["weekday"] = by_feat["launch_weekday"][0].proposed
        if "title" in by_feat:
            t["add_number"] = True
        return t

    @staticmethod
    def _describe(cum: dict[str, Any], raw: dict[str, Any]) -> str:
        parts = []
        if "goal_usd" in cum:
            parts.append(f"goal → ${float(cum['goal_usd']):,.0f}")
        if "duration_days" in cum:
            parts.append(f"duration → {cum['duration_days']}d")
        if "weekday" in cum:
            parts.append(f"launch → {cum['weekday']}")
        if "add_number" in cum:
            parts.append("title → add number")
        return "; ".join(parts)
