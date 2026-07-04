"""
missing.py

Production-ready missing value handler for the
AI Venture Intelligence Platform.

Author: Yash
"""

from typing import Optional

import pandas as pd
from sklearn.impute import SimpleImputer

from .base import BasePreprocessor


class MissingValueHandler(BasePreprocessor):
    """
    Handles missing value imputation for tabular datasets.

    Supports separate imputation strategies for:
        • Numeric columns
        • Categorical columns

    Parameters
    ----------
    numeric_strategy : str, default="median"
        Strategy for numeric columns.
        Supported:
            - mean
            - median

    categorical_strategy : str, default="most_frequent"
        Strategy for categorical columns.
        Supported:
            - most_frequent
    """

    VALID_NUMERIC_STRATEGIES = {"mean", "median"}
    VALID_CATEGORICAL_STRATEGIES = {"most_frequent"}

    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
    ):

        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy

        self.numeric_columns = []
        self.categorical_columns = []

        self.numeric_imputer: Optional[SimpleImputer] = None
        self.categorical_imputer: Optional[SimpleImputer] = None

        self._validate_strategies()

    def _validate_strategies(self) -> None:
        """
        Validate the user supplied strategies.
        """

        if self.numeric_strategy not in self.VALID_NUMERIC_STRATEGIES:
            raise ValueError(
                f"Unsupported numeric strategy: {self.numeric_strategy}"
            )

        if (
            self.categorical_strategy
            not in self.VALID_CATEGORICAL_STRATEGIES
        ):
            raise ValueError(
                f"Unsupported categorical strategy: "
                f"{self.categorical_strategy}"
            )

    @staticmethod
    def _validate_dataframe(X) -> None:
        """
        Ensure input is a pandas DataFrame.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learn imputation statistics from training data.

        Parameters
        ----------
        X : pandas.DataFrame
            Training dataframe.

        Returns
        -------
        self
        """

        self._validate_dataframe(X)

        self.numeric_columns = (
            X.select_dtypes(include="number")
            .columns
            .tolist()
        )

        self.categorical_columns = (
            X.select_dtypes(exclude="number")
            .columns
            .tolist()
        )

        if self.numeric_columns:

            self.numeric_imputer = SimpleImputer(
                strategy=self.numeric_strategy
            )

            self.numeric_imputer.fit(
                X[self.numeric_columns]
            )

        if self.categorical_columns:

            self.categorical_imputer = SimpleImputer(
                strategy=self.categorical_strategy
            )

            self.categorical_imputer.fit(
                X[self.categorical_columns]
            )

        return self

    def transform(self, X: pd.DataFrame):
        """
        Apply learned imputations.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """

        self._validate_dataframe(X)

        X = X.copy()

        if self.numeric_imputer is not None:

            X[self.numeric_columns] = (
                self.numeric_imputer.transform(
                    X[self.numeric_columns]
                )
            )

        if self.categorical_imputer is not None:

            X[self.categorical_columns] = (
                self.categorical_imputer.transform(
                    X[self.categorical_columns]
                )
            )

        return X