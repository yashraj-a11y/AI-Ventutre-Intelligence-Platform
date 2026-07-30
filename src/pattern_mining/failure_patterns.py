"""Phase C — Failure Pattern Mining (Day-7, Layer 6).

The symmetric mirror of Phase B: *what do FAILED campaigns repeatedly have in common?*
:class:`FailurePatternMiner` reuses the shared :func:`~src.pattern_mining.mining.mine_rules`
output and keeps the **failure-leaning** rules — funding **lift < 1** (funding *less* likely
than the 0.4038 base rate), above minimum support and statistically significant.

Because ``Rule`` measures everything toward *funded*, failure is just the other side: a rule
with funding-lift 0.25 has **failure-lift** ``(1 − confidence) / (1 − base_rate)`` well above
1. We surface both so the pattern reads naturally as a warning ("very large goals fund at
10% vs 40% base — a strong failure signal").

Real items only; historical associations, not causes (a large goal is *associated* with
failure — it doesn't strictly *cause* it, though the confound is intuitive here).

Example
-------
>>> miner = FailurePatternMiner()
>>> pats = miner.mine()
>>> pats[0]["label"], pats[0]["lift"], pats[0]["failure_lift"]
('goal_band=a very large goal', 0.25, 1.51)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.pattern_mining.mining import Rule, build_item_frame, mine_rules
from src.utils.config import load_config
from src.utils.logger import get_logger


class FailurePatternMiner:
    """Mine and rank failure-leaning association rules (funding lift < 1).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.patterns.failure")

        m = self.config.get("day7", {}).get("mining", {})
        self.min_support_count = int(m.get("min_support_count", 500))
        self.single_features = list(m.get("single_features", []))
        self.pair_features = list(m.get("pair_features", []))

        c = self.config.get("day7", {}).get("failure", {})
        self.max_lift = float(c.get("max_lift", 0.85))
        self.max_pvalue = float(c.get("max_pvalue", 0.05))
        self.top_k = int(c.get("top_k", 25))
        self.out_path = Path(c.get("out_path", "artifacts/patterns/failure_patterns.json"))

        self.base_rate_: Optional[float] = None
        self.patterns_: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ public
    def mine(self, frame: Optional[pd.DataFrame] = None, base: Optional[float] = None,
             rules: Optional[list[Rule]] = None) -> list[dict[str, Any]]:
        """Keep failure-leaning + significant rules; rank by strongest failure signal."""
        if rules is None:
            if frame is None:
                frame, base = build_item_frame(self.config, self.logger)
            rules = mine_rules(frame, base, self.single_features,
                               self.pair_features, self.min_support_count)
        self.base_rate_ = base

        kept = [r for r in rules
                if r.lift <= self.max_lift and r.p_value <= self.max_pvalue]
        kept.sort(key=lambda r: (r.lift, -r.support))       # lowest funding-lift = worst first
        self.patterns_ = [self._failure_view(r, base) for r in kept[: self.top_k]]
        self.logger.info("Failure patterns: %d rules pass (lift<=%.2f, p<=%.3g); worst lift %.2f.",
                         len(kept), self.max_lift, self.max_pvalue,
                         kept[0].lift if kept else 0.0)
        return self.patterns_

    def save(self) -> Path:
        """Persist the ranked failure patterns as JSON (with provenance)."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pattern_type": "failure",
            "base_rate": round(self.base_rate_, 4) if self.base_rate_ else None,
            "thresholds": {"min_support_count": self.min_support_count,
                           "max_lift": self.max_lift, "max_pvalue": self.max_pvalue},
            "note": ("Historical associations from train campaigns (metadata + titles only), "
                     "not causal drivers. failure_lift = P(failed|items)/P(failed)."),
            "n_patterns": len(self.patterns_),
            "patterns": self.patterns_,
        }
        self.out_path.write_text(json.dumps(payload, indent=2))
        self.logger.info("Saved %d failure patterns -> %s", len(self.patterns_), self.out_path)
        return self.out_path

    # --------------------------------------------------------------- internals
    @staticmethod
    def _failure_view(r: Rule, base: float) -> dict[str, Any]:
        """Augment a rule dict with failure-oriented measures for readability."""
        d = r.to_dict()
        failure_confidence = 1.0 - r.confidence            # P(failed | items)
        d["failure_confidence"] = round(failure_confidence, 4)
        d["failure_lift"] = round(failure_confidence / (1.0 - base), 3) if base < 1 else None
        return d
