"""Phase F — Domain Feature Engineering (Day-3 Stage-2).

Runs **first** in the pipeline (Phase L): it *creates* the engineered columns
that the numeric branch later logs/caps/scales. Every feature here is
**pre-launch and leakage-free** — computable from information a creator has at
the moment of launch (the goal, the two dates, the title). Nothing derived from
the *outcome* (pledged, backers, funding velocity) is allowed; those live in the
future relative to the prediction and would leak the label.

Features built
--------------
Temporal (from ``launched`` / ``deadline``):
  - ``campaign_duration``  = (deadline - launched).days
  - ``launch_year`` / ``launch_month`` / ``launch_weekday`` / ``launch_quarter``
  - ``is_weekend_launch``  = weekday in {Sat, Sun}
Derived numeric:
  - ``goal_per_day``       = goal / max(campaign_duration, 1)   ← guarded
Title style (from the *raw* ``name``, before text-cleaning strips casing/digits):
  - ``title_length`` (chars), ``word_count``, ``avg_word_length``,
    ``uppercase_ratio``, ``has_number``

**The divide-by-zero guard.** Phase A found 74 campaigns with
``campaign_duration <= 0`` (deadline == launch day). ``goal / 0`` is ``inf``,
which would poison log1p and the scaler downstream. The denominator is clipped
to ``>= min_duration_days`` (default 1) so ``goal_per_day`` is always finite.

**Stateless / leakage-free by construction.** Unlike the scaler or encoder, this
builder learns *nothing* from the data — every output is a row-wise function of
that row alone (category frequency, which *is* learned, was moved to Phase B;
``goal_log`` to Phase C). ``fit`` is therefore a no-op that just records the
output schema; it exists for pipeline-interface uniformity.

Example
-------
>>> fb = FeatureBuilder().fit(X_train)
>>> Xf = fb.transform(X_train)
>>> sorted(fb.feature_names_)[:2]
['avg_word_length', 'campaign_duration']
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger


class FeatureBuilder:
    """Create pre-launch, leakage-free engineered features (row-wise, stateless).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.features.feature_builder")

        fb = self.config.get("day3", {}).get("feature_builder", {})
        self.launched_col: str = fb.get("launched_col", "launched")
        self.deadline_col: str = fb.get("deadline_col", "deadline")
        self.title_col: str = fb.get("title_col", "name")  # raw title, keeps casing/digits
        self.goal_col: str = fb.get("goal_col", "goal")
        self.min_duration_days: int = int(fb.get("min_duration_days", 1))

        self.feature_names_: list[str] = []
        self._fitted = False

    # -------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureBuilder":
        """No-op fit (stateless). Records the output schema for interface parity."""
        self.feature_names_ = [
            "campaign_duration",
            "goal_per_day",
            "launch_year",
            "launch_month",
            "launch_weekday",
            "launch_quarter",
            "is_weekend_launch",
            "title_length",
            "word_count",
            "avg_word_length",
            "uppercase_ratio",
            "has_number",
        ]
        self._fitted = True
        self.logger.info("FeatureBuilder ready — will build %d features.", len(self.feature_names_))
        return self

    # -------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` plus the engineered columns (row-preserving, additive)."""
        self._check_fitted()
        out = X.copy()

        self._build_temporal(out)
        self._build_goal_per_day(out)
        self._build_title_stats(out)

        self.logger.info(
            "Built %d features on %d rows.", len(self.feature_names_), len(out)
        )
        return out

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # --------------------------------------------------------------- builders
    def _build_temporal(self, out: pd.DataFrame) -> None:
        launched = pd.to_datetime(out[self.launched_col], errors="coerce")
        deadline = pd.to_datetime(out[self.deadline_col], errors="coerce")

        out["campaign_duration"] = (deadline - launched).dt.days
        out["launch_year"] = launched.dt.year
        out["launch_month"] = launched.dt.month
        out["launch_weekday"] = launched.dt.weekday        # Mon=0 .. Sun=6
        out["launch_quarter"] = launched.dt.quarter
        out["is_weekend_launch"] = (launched.dt.weekday >= 5).astype("int64")

    def _build_goal_per_day(self, out: pd.DataFrame) -> None:
        goal = pd.to_numeric(out[self.goal_col], errors="coerce").astype(float)
        # GUARD: clip denominator to >= min_duration_days so goal/0 never happens.
        safe_duration = out["campaign_duration"].clip(lower=self.min_duration_days)
        out["goal_per_day"] = goal / safe_duration

    def _build_title_stats(self, out: pd.DataFrame) -> None:
        # Use the RAW title: casing and digits are the signal for style features.
        name = out[self.title_col].fillna("").astype(str)

        out["title_length"] = name.str.len()
        out["word_count"] = name.str.split().str.len().fillna(0).astype("int64")

        letters_only = name.str.replace(r"\s+", "", regex=True).str.len()
        wc = out["word_count"].replace(0, np.nan)
        out["avg_word_length"] = (letters_only / wc).fillna(0.0)

        n_upper = name.str.count(r"[A-Z]")
        n_alpha = name.str.count(r"[A-Za-z]").replace(0, np.nan)
        out["uppercase_ratio"] = (n_upper / n_alpha).fillna(0.0)

        out["has_number"] = name.str.contains(r"\d", regex=True).astype("int64")

    # --------------------------------------------------------------- internals
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder must be fit() before transform().")
