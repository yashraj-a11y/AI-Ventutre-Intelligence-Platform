"""Phase C — Missing Value Treatment.

Fills missing values with **configurable, type-aware** strategies:

- Numeric columns    -> ``mean`` or ``median`` (default median: robust to skew)
- Categorical columns-> ``mode`` or a constant ``"Unknown"``
- Text columns       -> empty string ``""``

The critical idea: :meth:`MissingValueHandler.fit` **learns** the fill values from
the *training* data (e.g. the training median of ``goal``) and stores them; then
:meth:`transform` **applies those stored values** to any dataset (train or test).
Fitting on train and transforming test with the *same* stored statistics is what
prevents statistical leakage.

Note: ``country == 'N,0"'`` is a non-null string sentinel — it is NOT missing, so
this handler ignores it. Phase F (integrity) removes that cohort.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger


class MissingValueHandler:
    """Configurable, type-aware missing-value imputer (row-preserving).

    Parameters
    ----------
    config:
        Parsed config; loaded from ``configs/preprocessing.yaml`` if ``None``.
    numeric_strategy, categorical_strategy:
        Override the config strategies (``"mean"``/``"median"``,
        ``"mode"``/``"unknown"``).
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        numeric_strategy: Optional[str] = None,
        categorical_strategy: Optional[str] = None,
        logger=None,
    ) -> None:
        self.logger = logger or get_logger("avip.preprocessing.missing")
        cfg = config if config is not None else load_config()

        missing_cfg = cfg.get("missing", {})
        self.numeric_strategy = numeric_strategy or missing_cfg.get(
            "numeric_strategy", "median"
        )
        self.categorical_strategy = categorical_strategy or missing_cfg.get(
            "categorical_strategy", "mode"
        )
        self.categorical_fill = missing_cfg.get("categorical_fill", "Unknown")
        self.text_fill = missing_cfg.get("text_fill", "")
        self.text_columns = list(missing_cfg.get("text_columns", []))

        dtypes = cfg.get("datatypes", {})
        self.numeric_columns = list(dtypes.get("numeric_columns", []))
        self.category_columns = list(dtypes.get("category_columns", []))

        # Learned during fit(): column -> fill value.
        self.fill_values_: dict[str, Any] = {}

    # -------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y=None) -> "MissingValueHandler":
        """Learn per-column fill values from the (training) data."""
        self.fill_values_ = {}

        for col in self.numeric_columns:
            if col in X.columns:
                if self.numeric_strategy == "mean":
                    self.fill_values_[col] = float(X[col].mean())
                else:
                    self.fill_values_[col] = float(X[col].median())

        for col in self.category_columns:
            if col in X.columns:
                if self.categorical_strategy == "unknown":
                    self.fill_values_[col] = self.categorical_fill
                else:
                    mode = X[col].mode(dropna=True)
                    self.fill_values_[col] = (
                        mode.iloc[0] if not mode.empty else self.categorical_fill
                    )

        for col in self.text_columns:
            if col in X.columns:
                self.fill_values_[col] = self.text_fill

        self.logger.info("MissingValueHandler learned fill values: %s", self.fill_values_)
        return self

    # --------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the stored fill values (must call :meth:`fit` first)."""
        if not self.fill_values_:
            self.logger.warning("MissingValueHandler.transform called before fit(); no-op.")
            return X.copy()

        X = X.copy()
        filled_counts = {}
        for col, value in self.fill_values_.items():
            if col in X.columns:
                n_missing = int(X[col].isna().sum())
                if n_missing:
                    X[col] = X[col].fillna(value)
                    filled_counts[col] = n_missing
        if filled_counts:
            self.logger.info("MissingValueHandler filled: %s", filled_counts)
        else:
            self.logger.info("MissingValueHandler: no missing values to fill.")
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------ report
    def missing_report(self, X: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Return per-column missing counts and percentages (columns with >0)."""
        n = len(X)
        report = {}
        for col in X.columns:
            miss = int(X[col].isna().sum())
            if miss:
                report[col] = {"count": miss, "percent": round(100 * miss / n, 4)}
        return report
