"""Phase E — Datatype Correction.

Raw CSVs store everything loosely: dates as strings, categories as bulky objects.
:class:`DataTypeHandler` coerces each column to its correct production dtype:

- datetime columns (``launched``, ``deadline``) -> ``datetime64``
- category columns  (``category``, ``main_category``, ``currency``, ``country``) -> ``category``
- numeric columns   (``goal``, ``usd_goal_real``) -> ``float`` via ``pd.to_numeric``
- text columns      (``name``, ``name_clean``) -> ``string``

Correct dtypes give: less memory (``category`` is a big win), faster ops, and —
importantly — they enable Phase F's date rules (you can't compare string "dates").
So this handler must run **before** integrity validation.

``errors="coerce"`` turns un-parseable values into ``NaT``/``NaN`` rather than
crashing; those invalid values are then caught by Phase F (e.g. the 1970 epoch
sentinel, launch-after-deadline). This handler is row-preserving.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger


class DataTypeHandler:
    """Coerce columns to their correct production datatypes (row-preserving)."""

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        text_columns: Optional[list[str]] = None,
        logger=None,
    ) -> None:
        self.logger = logger or get_logger("avip.preprocessing.datatype")
        cfg = config if config is not None else load_config()
        dt = cfg.get("datatypes", {})
        self.datetime_columns = list(dt.get("datetime_columns", []))
        self.category_columns = list(dt.get("category_columns", []))
        self.numeric_columns = list(dt.get("numeric_columns", []))
        # text columns default to name + its cleaned form
        self.text_columns = text_columns or ["name", "name_clean"]

    def fit(self, X: pd.DataFrame, y=None) -> "DataTypeHandler":
        return self  # stateless

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        conversions: dict[str, str] = {}
        coerced_na: dict[str, int] = {}

        for col in self.datetime_columns:
            if col in X.columns:
                before_na = int(X[col].isna().sum())
                X[col] = pd.to_datetime(X[col], errors="coerce")
                new_na = int(X[col].isna().sum()) - before_na
                conversions[col] = "datetime64"
                if new_na:
                    coerced_na[col] = new_na

        for col in self.numeric_columns:
            if col in X.columns:
                before_na = int(X[col].isna().sum())
                X[col] = pd.to_numeric(X[col], errors="coerce")
                new_na = int(X[col].isna().sum()) - before_na
                conversions[col] = "float"
                if new_na:
                    coerced_na[col] = new_na

        for col in self.category_columns:
            if col in X.columns:
                X[col] = X[col].astype("category")
                conversions[col] = "category"

        for col in self.text_columns:
            if col in X.columns:
                X[col] = X[col].astype("string")
                conversions[col] = "string"

        self.logger.info("DataTypeHandler converted: %s", conversions)
        if coerced_na:
            self.logger.warning(
                "DataTypeHandler: coercion produced new NaN/NaT (caught later by "
                "integrity): %s",
                coerced_na,
            )
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def memory_report(self, X: pd.DataFrame) -> dict[str, Any]:
        """Return per-column and total memory (MB), deep for object/string cols."""
        usage = X.memory_usage(deep=True)
        per_col = {c: round(int(usage[c]) / 1024**2, 3) for c in X.columns}
        total_mb = round(float(usage.sum()) / 1024**2, 3)
        return {"total_mb": total_mb, "per_column_mb": per_col}
