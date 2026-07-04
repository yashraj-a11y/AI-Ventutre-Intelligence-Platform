"""
outliers.py

Production-ready outlier handling module for the
AI Venture Intelligence Platform.

Author: Yash
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from .base import BasePreprocessor


class OutlierHandler(BasePreprocessor):
    """
    Handles outlier detection and treatment.

    Supported methods
    -----------------
    • IQR
    • Z-Score

    Supported actions
    -----------------
    • remove
    • clip

    Parameters
    ----------
    columns : list[str], default=None
        Numerical columns to process.
        If None, all numeric columns are used.

    method : str, default="iqr"
        Outlier detection method.

        Supported:
            - iqr
            - zscore

    action : str, default="clip"

        Supported:
            - clip
            - remove

    threshold : float, default=3.0
        Z-score threshold.
    """

    VALID_METHODS = {
        "iqr",
        "zscore",
    }

    VALID_ACTIONS = {
        "clip",
        "remove",
    }

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        action: str = "clip",
        threshold: float = 3.0,
    ):

        self.columns = columns
        self.method = method
        self.action = action
        self.threshold = threshold

        self.bounds = {}

        self._validate_arguments()

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    def _validate_arguments(self):

        if self.method not in self.VALID_METHODS:
            raise ValueError(
                f"Unsupported method: {self.method}"
            )

        if self.action not in self.VALID_ACTIONS:
            raise ValueError(
                f"Unsupported action: {self.action}"
            )

    @staticmethod
    def _validate_dataframe(X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

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

        self.bounds = {}

        for column in self.columns:

            if self.method == "iqr":

                q1 = X[column].quantile(0.25)
                q3 = X[column].quantile(0.75)

                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

            else:

                mean = X[column].mean()
                std = X[column].std()

                lower = mean - self.threshold * std
                upper = mean + self.threshold * std

            self.bounds[column] = (
                lower,
                upper,
            )

        return self

    # -------------------------------------------------
    # Transform
    # -------------------------------------------------

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:

        self._validate_dataframe(X)

        X = X.copy()

        if self.action == "clip":

            return self._clip_outliers(X)

        return self._remove_outliers(X)

    # -------------------------------------------------
    # Clip Outliers
    # -------------------------------------------------

    def _clip_outliers(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:

        for column in self.columns:

            lower, upper = self.bounds[column]

            X[column] = X[column].clip(
                lower=lower,
                upper=upper,
            )

        return X

    # -------------------------------------------------
    # Remove Outliers
    # -------------------------------------------------

    def _remove_outliers(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:

        mask = pd.Series(
            True,
            index=X.index,
        )

        for column in self.columns:

            lower, upper = self.bounds[column]

            mask &= (
                (X[column] >= lower)
                &
                (X[column] <= upper)
            )

        return X.loc[mask].reset_index(drop=True)