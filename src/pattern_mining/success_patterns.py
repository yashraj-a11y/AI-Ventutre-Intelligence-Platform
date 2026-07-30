"""Phase B — Success Pattern Mining (Day-7, Layer 6).

*What do funded campaigns repeatedly have in common?* :class:`SuccessPatternMiner` mines
association rules ``{real items} → funded`` and keeps the ones that are **success-leaning**:
lift clearly above 1 (funding more likely than the 0.4038 base rate), above minimum support,
and statistically significant.

Rigor (Phase-A commitments): every kept rule clears **support** (≥ min count, set in
``day7.mining``), **effect size** (lift ≥ ``min_lift``), and **significance** (chi-square
p ≤ ``max_pvalue``) — jointly. Ranked by lift, with support shown as the evidence behind it.

Real items only — category / goal band / country / timing / title. No founder / market /
revenue patterns (they don't exist). Rules are historical associations, not causes.

Example
-------
>>> miner = SuccessPatternMiner()
>>> pats = miner.mine()          # ranked success rules
>>> pats[0]["label"], pats[0]["lift"]
('main_category=Dance ∧ goal_band=a small goal', 1.76)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.pattern_mining.mining import Rule, build_item_frame, mine_rules
from src.utils.config import load_config
from src.utils.logger import get_logger


class SuccessPatternMiner:
    """Mine and rank success-leaning association rules (lift > 1).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.patterns.success")

        m = self.config.get("day7", {}).get("mining", {})
        self.min_support_count = int(m.get("min_support_count", 500))
        self.single_features = list(m.get("single_features", []))
        self.pair_features = list(m.get("pair_features", []))

        c = self.config.get("day7", {}).get("success", {})
        self.min_lift = float(c.get("min_lift", 1.15))
        self.max_pvalue = float(c.get("max_pvalue", 0.05))
        self.top_k = int(c.get("top_k", 25))
        self.out_path = Path(c.get("out_path", "artifacts/patterns/success_patterns.json"))

        self.base_rate_: Optional[float] = None
        self.patterns_: list[Rule] = []

    # ------------------------------------------------------------------ public
    def mine(self, frame: Optional[pd.DataFrame] = None, base: Optional[float] = None,
             rules: Optional[list[Rule]] = None) -> list[dict[str, Any]]:
        """Mine (or reuse) rules, keep success-leaning + significant, rank by lift."""
        if rules is None:
            if frame is None:
                frame, base = build_item_frame(self.config, self.logger)
            rules = mine_rules(frame, base, self.single_features,
                               self.pair_features, self.min_support_count)
        self.base_rate_ = base

        kept = [r for r in rules
                if r.lift >= self.min_lift and r.p_value <= self.max_pvalue]
        kept.sort(key=lambda r: (r.lift, r.support), reverse=True)
        self.patterns_ = kept[: self.top_k]
        self.logger.info("Success patterns: %d rules pass (lift>=%.2f, p<=%.3g); top lift %.2f.",
                         len(kept), self.min_lift, self.max_pvalue,
                         self.patterns_[0].lift if self.patterns_ else 0.0)
        return [r.to_dict() for r in self.patterns_]

    def save(self) -> Path:
        """Persist the ranked success patterns as JSON (with provenance)."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pattern_type": "success",
            "base_rate": round(self.base_rate_, 4) if self.base_rate_ else None,
            "thresholds": {"min_support_count": self.min_support_count,
                           "min_lift": self.min_lift, "max_pvalue": self.max_pvalue},
            "note": ("Historical associations from train campaigns (metadata + titles only), "
                     "not causal drivers."),
            "n_patterns": len(self.patterns_),
            "patterns": [r.to_dict() for r in self.patterns_],
        }
        self.out_path.write_text(json.dumps(payload, indent=2))
        self.logger.info("Saved %d success patterns -> %s", len(self.patterns_), self.out_path)
        return self.out_path
