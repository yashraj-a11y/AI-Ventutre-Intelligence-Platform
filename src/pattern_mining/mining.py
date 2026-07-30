"""Shared mining core (Day-7, Layer 6) — item frame + association-rule statistics.

Internal infrastructure used by every pattern miner (success / failure / funding /
temporal) and the ranker. Two things:

1. :func:`build_item_frame` — discretize the **train** campaigns' real pre-launch features
   into categorical **items** (the only thing association rules can consume). Numeric
   features are binned into human bands so "a very large goal" is an item, not a float.
2. :func:`mine_rules` — enumerate 1- and 2-item rules ``{items} → funded`` via vectorized
   ``groupby`` counting, and attach the full statistics per rule: support, confidence,
   **lift** (confidence / base rate), and a **chi-square p-value** (is the deviation from
   the base rate real, or sampling noise?).

Only **real** features are mined — category, country, currency, goal band, launch timing,
campaign duration, title word-count, has-number. No founder / market / revenue / growth
items exist. Every rule is a historical **association**, not a cause.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.knowledge_base.venture_database import goal_band


@dataclass
class Rule:
    """One association rule ``{items} → funded`` with its statistics.

    ``lift > 1`` = funding more likely than the base rate (success-leaning);
    ``lift < 1`` = less likely (failure-leaning). ``p_value`` from a 2×2 chi-square.
    """

    items: dict[str, Any]        # e.g. {"main_category": "Dance", "goal_band": "a small goal"}
    order: int                   # number of items (1 or 2)
    n: int                       # campaigns matching the antecedent
    n_funded: int
    support: float               # n / N
    confidence: float            # P(funded | items)
    lift: float                  # confidence / base_rate
    p_value: float               # chi-square significance
    direction: str               # "success" (lift>1) | "failure" (lift<1)

    def label(self) -> str:
        return " ∧ ".join(f"{k}={v}" for k, v in self.items.items())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label()
        d["support"] = round(self.support, 5)
        d["confidence"] = round(self.confidence, 4)
        d["lift"] = round(self.lift, 3)
        d["p_value"] = float(f"{self.p_value:.3e}")
        return d


def build_item_frame(config: dict[str, Any], logger=None) -> tuple[pd.DataFrame, float]:
    """Discretize train campaigns into categorical item columns + target. Returns (frame, base_rate)."""
    m = config.get("day7", {}).get("mining", {})
    df = pd.read_csv(m.get("processed_path", "data/processed/kickstarter_processed.csv"))
    idx = np.load(Path(m.get("split_dir", "artifacts/splits")) / f"{m.get('use_split','train')}_index.npy")
    d = df.iloc[idx].copy()

    launched = pd.to_datetime(d["launched"], errors="coerce")
    deadline = pd.to_datetime(d["deadline"], errors="coerce")
    duration = (deadline - launched).dt.days
    words = d["name"].fillna("").astype(str).str.split().str.len()

    frame = pd.DataFrame({
        "main_category": d["main_category"].astype(str).values,
        "category": d["category"].astype(str).values,
        "country": d["country"].astype(str).values,
        "currency": d["currency"].astype(str).values,
        "goal_band": d["usd_goal_real"].apply(goal_band).values,
        "launch_year": launched.dt.year.astype("Int64").astype(str).values,
        "launch_month": launched.dt.month.astype("Int64").astype(str).str.zfill(2).values,
        "launch_quarter": ("Q" + launched.dt.quarter.astype("Int64").astype(str)).values,
        "launch_weekday": launched.dt.day_name().values,
        "is_weekend_launch": np.where(launched.dt.weekday >= 5, "weekend", "weekday"),
        "duration_band": pd.cut(duration, [-1, 15, 30, 45, 60, 10_000],
                                labels=["<=15d", "16-30d", "31-45d", "46-60d", ">60d"]).astype(str).values,
        "word_count_band": pd.cut(words, [-1, 3, 6, 10, 1000],
                                  labels=["1-3 words", "4-6 words", "7-10 words", "11+ words"]).astype(str).values,
        "has_number": np.where(d["name"].fillna("").astype(str).str.contains(r"\d"),
                               "has number", "no number"),
        "usd_goal_real": d["usd_goal_real"].astype(float).values,   # raw (for goal thresholds; not an item)
        "target": d["target"].astype(int).values,
    })
    base = float(frame["target"].mean())
    if logger:
        # non-item columns: usd_goal_real (raw, for thresholds) + target
        logger.info("Item frame: %d campaigns, %d item features, base funding rate %.4f.",
                    len(frame), frame.shape[1] - 2, base)
    return frame, base


def mine_rules(frame: pd.DataFrame, base: float,
               single_features: list[str], pair_features: list[str],
               min_support_count: int) -> list[Rule]:
    """Enumerate 1- and 2-item rules ``{items} → funded`` with full statistics."""
    N = len(frame)
    total_funded = int(frame["target"].sum())
    rules: list[Rule] = []

    # 1-item rules: one groupby per feature counts every value at once (vectorized).
    for f in single_features:
        g = frame.groupby(f, observed=True)["target"].agg(["size", "sum"])
        for val, row in g.iterrows():
            r = _make_rule({f: val}, 1, int(row["size"]), int(row["sum"]),
                           N, total_funded, base, min_support_count)
            if r:
                rules.append(r)

    # 2-item rules: curated feature pairs (keeps the itemset space small + meaningful).
    for f1, f2 in itertools.combinations(pair_features, 2):
        g = frame.groupby([f1, f2], observed=True)["target"].agg(["size", "sum"])
        for (v1, v2), row in g.iterrows():
            r = _make_rule({f1: v1, f2: v2}, 2, int(row["size"]), int(row["sum"]),
                           N, total_funded, base, min_support_count)
            if r:
                rules.append(r)
    return rules


def _make_rule(items, order, n, n_funded, N, total_funded, base, min_count) -> Optional[Rule]:
    if n < min_count:                          # support gate — prune small-sample noise early
        return None
    confidence = n_funded / n
    lift = confidence / base if base > 0 else 0.0
    # 2×2 chi-square: (funded/not) × (in-antecedent/out).
    out_funded = total_funded - n_funded
    table = [[n_funded, n - n_funded], [out_funded, (N - n) - out_funded]]
    try:
        p = float(chi2_contingency(table)[1])
    except ValueError:
        p = 1.0
    return Rule(items=dict(items), order=order, n=n, n_funded=n_funded,
                support=n / N, confidence=confidence, lift=lift, p_value=p,
                direction="success" if lift >= 1 else "failure")
