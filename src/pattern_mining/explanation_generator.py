"""Phase G — Pattern Explanation Generator (Day-7, Layer 6).

Turn a ranked statistical pattern into a business-readable sentence — the last-mile
translation from ``main_category=Comics ∧ goal_band=a very small goal, lift 1.92`` into

    "Historically, campaigns in the Comics category seeking a very small goal were more
    likely to secure funding: they funded at 77% versus a 40% baseline, a lift of 1.92,
    across 1,461 campaigns. This is a historical association in campaign metadata and
    titles, not a proven cause of funding."

Two rules (same discipline as Day-5/6 explanation layers):

- **Readable-feature map** — each item ``feature=value`` becomes an English clause
  ("in the Comics category", "seeking a very small goal", "launched on a weekend"). Only
  real features have entries; there is nothing to say about founders/market/revenue.
- **Association, not causation** — every explanation ends with the mandatory caption. A
  pattern is a historical regularity, never a proven lever (Comics-with-small-goals fund
  more partly *because* small goals are easy to hit — confounded, not causal).

Example
-------
>>> gen = PatternExplanationGenerator()
>>> exps = gen.generate_examples()      # explains the top ranked patterns
>>> print(exps[0]["explanation"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.pattern_mining.pattern_ranker import PatternRanker
from src.utils.config import load_config
from src.utils.logger import get_logger

_MONTHS = {"01": "January", "02": "February", "03": "March", "04": "April",
           "05": "May", "06": "June", "07": "July", "08": "August",
           "09": "September", "10": "October", "11": "November", "12": "December"}

_CAPTION = ("This is a historical association in campaign metadata and titles, "
            "not a proven cause of funding.")


def item_phrase(feature: str, value: Any) -> str:
    """Turn one pattern item (feature=value) into an English clause. Real features only."""
    v = str(value)
    if feature in ("main_category", "category"):
        return f"in the {v} category"
    if feature == "goal_band":
        return f"seeking {v}"                      # e.g. "seeking a very small goal"
    if feature == "country":
        return f"based in {v}"
    if feature == "currency":
        return f"raising funds in {v}"
    if feature == "launch_year":
        return f"launched in {v}"
    if feature == "launch_month":
        return f"launched in {_MONTHS.get(v, v)}"
    if feature == "launch_quarter":
        return f"launched in {v}"
    if feature == "launch_weekday":
        return f"launched on {v}"
    if feature == "is_weekend_launch":
        return "launched on a weekend" if v == "weekend" else "launched on a weekday"
    if feature == "duration_band":
        return f"running a {v} campaign"
    if feature == "word_count_band":
        return f"with a {v} title"
    if feature == "has_number":
        return "with a number in the title" if v == "has number" else "without a number in the title"
    return f"{feature}={v}"                          # fallback — never fabricate


class PatternExplanationGenerator:
    """Render ranked patterns into founder-readable, honesty-captioned sentences.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    ranker:
        Optional pre-built :class:`PatternRanker` (share the mined rules).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 ranker: Optional[PatternRanker] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.patterns.explanation")

        c = self.config.get("day7", {}).get("explanation", {})
        self.top_k = int(c.get("top_k", 15))
        self.out_path = Path(c.get("out_path", "artifacts/patterns/pattern_explanations.json"))
        self.ranker = ranker or PatternRanker(self.config, self.logger)

    # ------------------------------------------------------------------ public
    def explain(self, pattern: dict[str, Any], base_rate: float) -> str:
        """Render one ranked pattern dict into a business-readable sentence."""
        ante = " and ".join(item_phrase(f, v) for f, v in pattern["items"].items())
        conf = pattern["confidence"]
        lift = pattern["lift"]
        n = pattern["n"]
        base_pct = round(base_rate * 100)

        if pattern.get("direction") == "success":
            outcome = "were more likely to secure funding"
            stat = (f"they funded at {conf:.0%} versus a {base_pct}% baseline, "
                    f"a lift of {lift:.2f}")
        else:
            outcome = "were more likely to fail"
            stat = (f"they funded at only {conf:.0%} versus a {base_pct}% baseline "
                    f"(lift {lift:.2f})")
        return (f"Historically, campaigns {ante} {outcome}: {stat}, across {n:,} campaigns. "
                f"{_CAPTION}")

    def generate_examples(self, ranked: Optional[list[dict[str, Any]]] = None,
                          base: Optional[float] = None) -> list[dict[str, Any]]:
        """Rank patterns (or reuse a shared ranking), explain the top-k, and save."""
        if ranked is None:
            ranked = self.ranker.rank()
            base = self.ranker.base_rate_
        out = [{"label": p["label"], "direction": p.get("direction"),
                "lift": p["lift"], "support": p["support"], "n": p["n"],
                "explanation": self.explain(p, base)}
               for p in ranked[: self.top_k]]

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(
            {"base_rate": round(base, 4), "note": _CAPTION,
             "n_explanations": len(out), "explanations": out}, indent=2))
        self.logger.info("Saved %d pattern explanations -> %s", len(out), self.out_path)
        return out
