"""Phase G — Human-readable Explanation Generator (Day-5, Layer 5).

Phase E returns *structured* contributions: raw feature names, signed SHAP values, and
**standardized** feature values (``usd_goal_real = 1.66`` = 1.66 SD above the mean
goal). Truthful, but not something you put in front of a founder. Phase G is the
**translation layer**: it turns each contribution into an honest English phrase via a
**readable-label map**, and assembles the phrases into a short paragraph.

Design rules (all in service of *honesty*):

- **Only real features are ever named.** The label map covers exactly the 65 tabular
  features plus the "title term '<token>'" pattern for the 800 TF-IDF words. There is no
  entry — and can be no entry — for founder/market/revenue, because those features do not
  exist.
- **Standardized value → qualitative word, never a fake number.** A one-SD-above goal
  becomes "a large funding goal", not "goal = 1.66". Cyclical features (launch
  month/weekday/quarter) get a *neutral* phrase — "the launch month" — because "high
  month" is meaningless; direction is carried by whether it sits in the ↑ or ↓ list.
- **One-hot direction is respected.** ``main_category_Theater`` with a *positive*
  (present) value reads "being a Theater project"; a *negative* (absent) value reads
  "not being a Theater project" — saying "being Theater" when the flag is 0 would be a lie.
- **Every rendering ends with the honesty caption**: these are learned associations from
  metadata + titles, not causal levers.

Example
-------
>>> gen = ExplanationGenerator()
>>> print(gen.render(local_explanation))
This campaign is predicted successful, with a 88% chance of funding.
Factors raising its odds: being a Theater project; the title term 'fringe'; a modest
funding goal; a short campaign length.
Factors lowering its odds: a recent launch year.
Note: these are patterns the model learned to associate with funding from campaign
metadata and title text — not guaranteed causes of success.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from src.explainability.local_explainer import Contribution, LocalExplainer, LocalExplanation
from src.utils.config import load_config
from src.utils.logger import get_logger

# --- readable-label map ------------------------------------------------------
# Numeric tabular: (subject noun, adjective when value HIGH, adjective when LOW).
_NUMERIC_LABELS: dict[str, tuple[str, str, str]] = {
    "usd_goal_real":   ("funding goal", "large", "modest"),
    "goal_per_day":    ("daily funding target", "steep", "gentle"),
    "campaign_duration": ("campaign length", "long", "short"),
    "category_freq":   ("project category", "common", "niche"),
    "launch_year":     ("launch year", "recent", "earlier"),
    "title_length":    ("title length", "long", "short"),
    "word_count":      ("title word count", "high", "low"),
    "avg_word_length": ("average word length in the title", "long", "short"),
    "uppercase_ratio": ("use of capital letters in the title", "heavy", "light"),
}
# Cyclical / flag features: neutral phrase (no high/low), direction from the ± list.
_NEUTRAL_LABELS: dict[str, str] = {
    "launch_month":      "the launch month",
    "launch_weekday":    "the launch day of the week",
    "launch_quarter":    "the launch quarter",
    "is_weekend_launch": "a weekend launch",
    "has_number":        "a number in the title",
}
# One-hot prefixes: (present template, absent template). {} = the decoded level.
_PREFIX_LABELS: dict[str, tuple[str, str]] = {
    "main_category_": ("being a {} project", "not being a {} project"),
    "currency_":      ("raising funds in {}", "not raising funds in {}"),
    "country_":       ("being based in {}", "not being based in {}"),
}
# A few friendly country/currency names; codes are an acceptable fallback.
_COUNTRY_NAMES = {
    "US": "the US", "GB": "the UK", "CA": "Canada", "AU": "Australia", "DE": "Germany",
    "FR": "France", "IT": "Italy", "ES": "Spain", "NL": "the Netherlands", "SE": "Sweden",
    "NZ": "New Zealand", "JP": "Japan", "MX": "Mexico", "DK": "Denmark", "IE": "Ireland",
    "CH": "Switzerland", "NO": "Norway", "HK": "Hong Kong", "SG": "Singapore",
    "AT": "Austria", "BE": "Belgium", "LU": "Luxembourg",
}

_HONESTY = ("these are patterns the model learned to associate with funding from campaign "
            "metadata and title text — not guaranteed causes of success")


def humanize(contribution: Contribution, z_high: float = 0.4) -> str:
    """Turn one structured :class:`Contribution` into an honest English phrase.

    Reusable by the Venture Copilot (Phase H). ``z_high`` is the |standardized value|
    threshold above which a numeric feature is called "high"/"low" rather than "typical".
    """
    name, val = contribution.feature, contribution.feature_value

    # 1) TF-IDF title token.
    if contribution.group == "text":
        kind = "phrase" if " " in name else "term"
        return (f"the title {kind} '{name}'" if val > 0
                else f"the absence of the title {kind} '{name}'")

    # 2) One-hot categorical (prefix match).
    for prefix, (present_t, absent_t) in _PREFIX_LABELS.items():
        if name.startswith(prefix):
            level = name[len(prefix):]
            if prefix == "country_":
                level = _COUNTRY_NAMES.get(level, level)
            return _fix_article((present_t if val > 0 else absent_t).format(level))

    # 3) Numeric with a meaningful high/low direction.
    if name in _NUMERIC_LABELS:
        subject, hi, lo = _NUMERIC_LABELS[name]
        if val >= z_high:
            return _fix_article(f"a {hi} {subject}")
        if val <= -z_high:
            return _fix_article(f"a {lo} {subject}")
        return _fix_article(f"a typical {subject}")

    # 4) Cyclical / flag: neutral phrase.
    if name in _NEUTRAL_LABELS:
        return _NEUTRAL_LABELS[name]

    # 5) Fallback: never fabricate — name the raw feature.
    return f"the feature '{name}'"


def _fix_article(phrase: str) -> str:
    """'a' → 'an' before a vowel-initial word (our vocab has no 'a university' traps)."""
    return re.sub(r"\ba ([aeiouAEIOU])", r"an \1", phrase)


class ExplanationGenerator:
    """Render :class:`LocalExplanation` objects into founder-readable prose.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    explainer:
        Optional pre-built :class:`LocalExplainer` (shares a warm SHAP engine).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 explainer: Optional[LocalExplainer] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.explain.generator")

        c = self.config.get("day5", {}).get("explanation", {})
        self.max_factors = int(c.get("max_factors", 4))
        self.z_high = float(c.get("z_high", 0.4))
        self.n_examples = int(c.get("n_examples", 6))
        self.examples_path = Path(
            c.get("examples_path", "artifacts/explainability/explanation_examples.json")
        )
        self._explainer = explainer   # lazily built only if generate_examples() is called

    # ------------------------------------------------------------------ public
    def readable_factors(self, contributions: list[Contribution]) -> list[str]:
        """Verbalize a list of contributions (capped at ``max_factors``)."""
        return [humanize(c, self.z_high) for c in contributions[: self.max_factors]]

    def render(self, exp: LocalExplanation) -> str:
        """Assemble a short founder-facing paragraph for one campaign."""
        pct = round(exp.funding_probability * 100)
        up = self.readable_factors(exp.top_positive)
        down = self.readable_factors(exp.top_negative)

        lines = [f"This campaign is predicted {exp.predicted_label}, "
                 f"with a {pct}% chance of funding."]
        if up:
            lines.append("Factors raising its odds: " + self._join(up) + ".")
        if down:
            lines.append("Factors lowering its odds: " + self._join(down) + ".")
        lines.append(f"Note: {_HONESTY}.")
        return "\n".join(lines)

    def to_readable(self, exp: LocalExplanation) -> dict[str, Any]:
        """Structured readable payload: text + verbalized factors + kept numbers."""
        return {
            "funding_probability": exp.funding_probability,
            "predicted_label": exp.predicted_label,
            "decision_threshold": exp.decision_threshold,
            "explanation_text": self.render(exp),
            "positive_factors": self.readable_factors(exp.top_positive),
            "negative_factors": self.readable_factors(exp.top_negative),
            "honesty_note": _HONESTY,
        }

    def generate_examples(self, n: Optional[int] = None, random_state: int = 7) -> Path:
        """Verbalize a spread of real campaigns and save ``explanation_examples.json``."""
        import pandas as pd
        import numpy as np

        n = n or self.n_examples
        if self._explainer is None:
            self._explainer = LocalExplainer(self.config, self.logger)

        feat_cols = ["name", "category", "main_category", "currency", "deadline",
                     "goal", "launched", "country", "usd_goal_real", "name_clean"]
        df = pd.read_csv("data/processed/kickstarter_processed.csv")
        sample = df.sample(max(n * 10, 60), random_state=random_state).reset_index(drop=True)
        exps = self._explainer.explain_raw(sample[feat_cols])

        # Spread across the probability range so examples aren't all similar.
        order = np.argsort([e.funding_probability for e in exps])
        picks = [exps[order[int(t)]] for t in np.linspace(0, len(order) - 1, n)]
        payload = [self.to_readable(e) for e in picks]

        self.examples_path.parent.mkdir(parents=True, exist_ok=True)
        self.examples_path.write_text(json.dumps(payload, indent=2))
        self.logger.info("Saved %d readable explanations -> %s", len(payload), self.examples_path)
        return self.examples_path

    # --------------------------------------------------------------- internals
    @staticmethod
    def _join(items: list[str]) -> str:
        """Oxford-style join: 'a; b; c'."""
        return "; ".join(items)
