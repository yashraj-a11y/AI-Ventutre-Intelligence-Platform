"""
duplicates.py

Production-ready duplicate detection module.

Project: AI_Venture_Intelligence
Author: Yash
"""

import pandas as pd

from .base import BasePreprocessor


class DuplicateHandler(BasePreprocessor):
    """
    Detects and removes duplicate rows from a DataFrame.

    Parameters
    ----------
    keep : str, default="first"

        Which duplicate to keep.

        Supported:

        - "first"
        - "last"
        - False
    """

    VALID_KEEP = {"first", "last", False}

    def __init__(
        self,
        keep="first",
    ):

        self.keep = keep

        self.duplicates_removed = 0

        self._validate_keep()

    def _validate_keep(self):

        if self.keep not in self.VALID_KEEP:
            raise ValueError(
                f"Unsupported keep strategy: {self.keep}"
            )

    @staticmethod
    def _validate_dataframe(X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

    def fit(self, X: pd.DataFrame, y=None):
        """
        Duplicate removal does not learn statistics.

        Present only to maintain sklearn compatibility.
        """

        self._validate_dataframe(X)

        return self

    def transform(self, X: pd.DataFrame):
        """
        Remove duplicate rows.

        Returns
        -------
        pandas.DataFrame
        """

        self._validate_dataframe(X)

        X = X.copy()

        original_rows = len(X)

        X = X.drop_duplicates(
            keep=self.keep
        )

        self.duplicates_removed = (
            original_rows - len(X)
        )

        return X.reset_index(drop=True)