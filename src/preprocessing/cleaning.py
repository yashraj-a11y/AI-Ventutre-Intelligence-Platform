"""Phase B — Basic Data Cleaning & Column Governance.

After Phase A confirms the data is *shaped* right, this module performs the first
transformations that actually **change** the data — but only structural cleaning,
never feature engineering (that is Day 3).

Four row-preserving handlers:

- :class:`ColumnCleaner`      — canonicalize column names to snake_case
                                (``"usd pledged"`` -> ``"usd_pledged"``, ``"ID"`` -> ``"id"``).
- :class:`TextCleaner`        — create a cleaned title column ``name_clean`` while
                                **preserving the raw ``name``** (Day 3 needs the raw
                                case/punctuation for features like ``uppercase_ratio``).
- :class:`CategoryNormalizer` — trim whitespace on categorical labels, preserving
                                the ``main_category -> category`` hierarchy.
- :class:`ColumnRemover`      — drop the identifier and target-leakage columns,
                                keeping ``usd_goal_real`` (known at launch).

All four are row-preserving, so they follow the sklearn-style
``fit`` / ``transform`` / ``fit_transform`` interface and never drop rows.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger


def normalize_column_name(name: str) -> str:
    """Canonicalize a column name: strip, lowercase, spaces -> underscores.

    >>> normalize_column_name("usd pledged")
    'usd_pledged'
    >>> normalize_column_name(" ID ")
    'id'
    """
    return str(name).strip().lower().replace(" ", "_")


class ColumnCleaner:
    """Standardize column names to a canonical snake_case form."""

    def __init__(self, logger=None) -> None:
        self.logger = logger or get_logger("avip.preprocessing.cleaning")

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnCleaner":
        return self  # stateless: nothing to learn

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        old = list(X.columns)
        X.columns = [normalize_column_name(c) for c in X.columns]
        renamed = {o: n for o, n in zip(old, X.columns) if o != n}
        if renamed:
            self.logger.info("ColumnCleaner renamed %d column(s): %s", len(renamed), renamed)
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class TextCleaner:
    """Produce a cleaned title column while preserving the raw text.

    Cleaning steps: lowercase, strip outcome markers ("(canceled)"), remove
    punctuation (keeping letters/digits/spaces), collapse whitespace. Digits are
    kept so Day 3 can still build ``has_number``. No tokenization / stopwords /
    TF-IDF — those belong to Day 3.
    """

    _OUTCOME_MARKERS = ["(canceled)", "(suspended)", "(cancelled)"]

    def __init__(
        self,
        text_column: str = "name",
        output_column: str = "name_clean",
        logger=None,
    ) -> None:
        self.text_column = text_column
        self.output_column = output_column
        self.logger = logger or get_logger("avip.preprocessing.cleaning")

    def fit(self, X: pd.DataFrame, y=None) -> "TextCleaner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.text_column not in X.columns:
            self.logger.warning(
                "TextCleaner: column '%s' absent; skipping.", self.text_column
            )
            return X

        cleaned = X[self.text_column].fillna("").astype(str).str.lower()
        for marker in self._OUTCOME_MARKERS:
            cleaned = cleaned.str.replace(marker, " ", regex=False)
        cleaned = cleaned.str.replace(r"[^a-z0-9\s]", " ", regex=True)  # drop punctuation
        cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()  # collapse

        X[self.output_column] = cleaned  # raw `name` is left untouched
        self.logger.info(
            "TextCleaner: wrote cleaned titles to '%s' (raw '%s' preserved).",
            self.output_column,
            self.text_column,
        )
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class CategoryNormalizer:
    """Trim whitespace on categorical label columns (preserves case/hierarchy)."""

    def __init__(
        self,
        category_columns: Optional[list[str]] = None,
        config: Optional[dict[str, Any]] = None,
        logger=None,
    ) -> None:
        self.logger = logger or get_logger("avip.preprocessing.cleaning")
        cfg = config if config is not None else load_config()
        self.category_columns = category_columns or cfg.get("datatypes", {}).get(
            "category_columns", []
        )

    def fit(self, X: pd.DataFrame, y=None) -> "CategoryNormalizer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        touched = []
        for col in self.category_columns:
            if col in X.columns:
                X[col] = X[col].astype(str).str.strip()
                touched.append(col)
        if touched:
            self.logger.info("CategoryNormalizer trimmed: %s", touched)
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class ColumnRemover:
    """Drop identifier and target-leakage columns.

    Matching is done on *normalized* names, so it works whether or not
    :class:`ColumnCleaner` has already run (``"usd pledged"`` and ``"usd_pledged"``
    both match).
    """

    def __init__(
        self,
        drop_columns: Optional[list[str]] = None,
        config: Optional[dict[str, Any]] = None,
        logger=None,
    ) -> None:
        self.logger = logger or get_logger("avip.preprocessing.cleaning")
        cfg = config if config is not None else load_config()
        cols = cfg.get("columns", {})
        if drop_columns is None:
            drop_columns = list(cols.get("identifier", [])) + list(
                cols.get("leakage_drop", [])
            )
        # canonicalize the drop-list once
        self.drop_normalized = {normalize_column_name(c) for c in drop_columns}

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnRemover":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        to_drop = [c for c in X.columns if normalize_column_name(c) in self.drop_normalized]
        X = X.drop(columns=to_drop)
        self.logger.info(
            "ColumnRemover dropped %d column(s): %s | kept %d.",
            len(to_drop),
            to_drop,
            X.shape[1],
        )
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
