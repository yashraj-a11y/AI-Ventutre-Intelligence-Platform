"""Phase F — Statistical Pattern Ranking (Day-7, Layer 6).

The shared **rigor layer**. B/C/D each mined and filtered rules with a fixed p ≤ 0.05.
But mining *m* rules and testing each at 0.05 guarantees ≈ 0.05·m false positives — the
multiple-testing problem from Phase A. :class:`PatternRanker` takes the **full** mined rule
set and applies the honest correction:

- **Bonferroni**: a rule is significant only if ``p ≤ α / m`` (α family-wise, m = rules
  tested), which controls the chance of *any* false positive across the whole search.
- **Effect size**: keep only ``|lift − 1| ≥ min_abs_lift`` — a pattern must be *big*, not
  just statistically detectable (with 265k rows, trivially small effects are "significant").
- **Unified ranking**: rank all surviving patterns (success *and* failure) by effect size =
  distance of lift from 1, so the most **influential** venture factors surface first.

Effect size (lift) and the p-value answer *different* questions: the p-value says "is this
real?" (and grows with n), lift says "is this big?" (independent of n). A pattern must pass
both. Output: ``ranked_patterns.csv`` + a multiple-testing summary.

Example
-------
>>> ranker = PatternRanker()
>>> ranked = ranker.rank()
>>> ranker.summary_["n_bonferroni_significant"], ranker.summary_["alpha_bonferroni"]
(..., 1.2e-04)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.pattern_mining.mining import Rule, build_item_frame, mine_rules
from src.utils.config import load_config
from src.utils.logger import get_logger


class PatternRanker:
    """Apply Bonferroni + effect-size gates to all mined rules and rank them.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.patterns.ranker")

        m = self.config.get("day7", {}).get("mining", {})
        self.min_support_count = int(m.get("min_support_count", 500))
        self.single_features = list(m.get("single_features", []))
        self.pair_features = list(m.get("pair_features", []))

        c = self.config.get("day7", {}).get("ranker", {})
        self.alpha = float(c.get("alpha", 0.05))
        self.min_abs_lift = float(c.get("min_abs_lift", 0.15))
        self.top_k = int(c.get("top_k", 40))
        self.out_path = Path(c.get("out_path", "artifacts/patterns/ranked_patterns.csv"))

        self.base_rate_: Optional[float] = None
        self.ranked_: list[dict[str, Any]] = []
        self.summary_: dict[str, Any] = {}

    # ------------------------------------------------------------------ public
    def rank(self, frame: Optional[pd.DataFrame] = None, base: Optional[float] = None,
             rules: Optional[list[Rule]] = None) -> list[dict[str, Any]]:
        """Rank all mined rules by effect size, gated by Bonferroni significance."""
        if frame is None:
            frame, base = build_item_frame(self.config, self.logger)
        if rules is None:
            rules = mine_rules(frame, base, self.single_features,
                               self.pair_features, self.min_support_count)
        self.base_rate_ = base

        m = len(rules)                                   # number of hypotheses tested
        alpha_bonf = self.alpha / m if m else self.alpha

        rows = []
        for r in rules:
            effect = abs(r.lift - 1.0)
            sig_raw = r.p_value <= self.alpha
            sig_bonf = r.p_value <= alpha_bonf
            if sig_bonf and effect >= self.min_abs_lift:
                d = r.to_dict()
                d.update({"effect_size": round(effect, 3),
                          "significant_bonferroni": True})
                rows.append(d)

        rows.sort(key=lambda d: (d["effect_size"], d["support"]), reverse=True)
        self.ranked_ = rows[: self.top_k]

        n_raw = sum(1 for r in rules if r.p_value <= self.alpha)
        n_bonf = sum(1 for r in rules if r.p_value <= alpha_bonf)
        self.summary_ = {
            "n_rules_tested": m,
            "alpha": self.alpha,
            "alpha_bonferroni": float(f"{alpha_bonf:.3e}"),
            "n_raw_significant": n_raw,               # would-be "findings" at p<=0.05
            "n_bonferroni_significant": n_bonf,       # survive multiple-testing correction
            "n_rejected_by_correction": n_raw - n_bonf,
            "min_abs_lift": self.min_abs_lift,
            "n_ranked": len(self.ranked_),
        }
        self.logger.info(
            "Ranked %d patterns. Multiple testing: %d rules tested, %d sig@0.05, "
            "%d survive Bonferroni (α=%.2e), %d rejected by correction.",
            len(self.ranked_), m, n_raw, n_bonf, alpha_bonf, n_raw - n_bonf,
        )
        return self.ranked_

    def save(self) -> Path:
        """Persist the ranked patterns as a flat CSV (+ log the multiple-testing summary)."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["label", "direction", "order", "support", "confidence", "lift",
                "effect_size", "p_value", "significant_bonferroni", "n", "n_funded"]
        df = pd.DataFrame(self.ranked_)
        df = df[[c for c in cols if c in df.columns]]
        df.insert(0, "rank", range(1, len(df) + 1))
        df.to_csv(self.out_path, index=False)
        self.logger.info("Saved %d ranked patterns -> %s", len(df), self.out_path)
        return self.out_path
