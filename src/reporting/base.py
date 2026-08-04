"""Shared reporting infrastructure (Day-13, Layer 13).

A uniform **section** schema so every report is a list of typed sections the exporter (Phase G) can
render generically to Markdown / HTML / JSON. Deterministic; no state.

Section kinds:
- ``text``    — body is a string (a paragraph).
- ``metrics`` — body is a dict {label: value} (a small key/value block).
- ``list``    — body is a list of strings (bullets).
- ``table``   — body is {"columns": [...], "rows": [[...], ...]}.

Every section carries a ``source`` tag (the upstream layer it came from) — the provenance that
enforces the NO-NEW-CLAIMS rule (a section that can't name its source can't be in the report).
"""

from __future__ import annotations

from typing import Any


def section(id: str, title: str, source: str, kind: str, body: Any) -> dict[str, Any]:
    """Build one uniform, provenance-tagged report section."""
    assert kind in ("text", "metrics", "list", "table"), kind
    return {"id": id, "title": title, "source": source, "kind": kind, "body": body}


# Human-readable prose for the common Day-5 SHAP features (Strengths / Weaknesses read as prose,
# not raw feature names). Falls back to the descriptor, then a de-underscored feature name.
_FEATURE_PROSE = {
    "category_freq": "your category's overall popularity",
    "usd_goal_real": "your funding goal size",
    "goal_per_day": "your goal relative to the campaign length",
    "uppercase_ratio": "your title's capitalization style",
    "title_length": "your title length",
    "name_word_count": "your title word count",
    "campaign_duration": "your campaign length",
    "launch_weekday": "your launch-day timing",
    "deadline_weekday": "your deadline timing",
    "blurb_length": "your blurb length",
}


def feature_prose(feature: str, descriptor: str | None = None) -> str:
    """Founder-readable phrase for a SHAP feature."""
    if feature in _FEATURE_PROSE:
        return _FEATURE_PROSE[feature]
    if feature.startswith("main_category_"):
        return f"being in the {feature.split('_', 2)[-1]} category"
    if descriptor and descriptor != feature:
        return descriptor                                    # e.g. "title term 'game'"
    return feature.replace("_", " ")
