"""
scaling.py

Production-ready feature scaling module for the
AI Venture Intelligence Platform.

Author: Yash
"""

from typing import List, Optional

import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from .base import BasePreprocessor


class ScalingHandler(BasePreprocessor):
    """
    Handles feature scaling for numerical columns.

    Supported scalers
    -----------------
    • StandardScaler
    • MinMaxScaler
    • RobustScaler

    Parameters
    ----------
    columns : list[str], default=None
        Numerical columns to scale.
        If None, all numeric columns are used.

    strategy : str, default="standard"
        Scaling strategy.

        Supported values:
            - standard
            - minmax
            - robust
    """

    VALID_STRATEGIES = {
        "standard",
        "minmax",
        "robust",
    }

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        strategy: str = "standard",
    ):

        self.columns = columns
        self.strategy = strategy

        self.scaler = None

        self._validate_strategy()

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    def _validate_strategy(self):

        if self.strategy not in self.VALID_STRATEGIES:

            raise ValueError(
                f"Unsupported scaling strategy: {self.strategy}"
            )

    @staticmethod
    def _validate_dataframe(X):

        if not isinstance(X, pd.DataFrame):

            raise TypeError(
                "Input must be a pandas DataFrame."
            )

    # -------------------------------------------------
    # Create Scaler
    # -------------------------------------------------

    def _create_scaler(self):

        if self.strategy == "standard":

            return StandardScaler()

        elif self.strategy == "minmax":

            return MinMaxScaler()

        elif self.strategy == "robust":

            return RobustScaler()

    # -------------------------------------------------
    # Fit
    # -------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None):

        self._validate_dataframe(X)

        if self.columns is None:

            self.columns = (
                X.select_dtypes(include="number")
                .columns
                .tolist()
            )

        self.scaler = self._create_scaler()

        self.scaler.fit(
            X[self.columns]
        )

        return self

    # -------------------------------------------------
    # Transform
    # -------------------------------------------------

    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:

        self._validate_dataframe(X)

        X = X.copy()

        X[self.columns] = self.scaler.transform(
            X[self.columns]
        )

        return X

