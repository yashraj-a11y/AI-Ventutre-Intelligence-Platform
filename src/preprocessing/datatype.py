"""
datatype.py

Production-ready data type correction module.

Project: AI_Venture_Intelligence
Author: Yash
"""

from typing import Dict, Optional

import pandas as pd

from .base import BasePreprocessor


class DataTypeHandler(BasePreprocessor):
    """
    Handles data type correction for tabular datasets.

    Converts columns to their desired data types before
    feature engineering and model training.

    Supported data types:
        - int
        - float
        - string
        - category
        - boolean
        - datetime

    Parameters
    ----------
    dtype_mapping : dict, default=None

        Dictionary containing column names and
        their desired data types.

        Example
        -------
        {
            "goal": "float",
            "deadline": "datetime",
            "country": "category"
        }
    """

    VALID_DTYPES = {
        "int",
        "float",
        "string",
        "category",
        "boolean",
        "datetime",
    }

    def __init__(
        self,
        dtype_mapping: Optional[Dict[str, str]] = None,
    ):

        self.dtype_mapping = dtype_mapping or {}

        self._validate_mapping()

    def _validate_mapping(self):
        """
        Validate requested data types.
        """

        for column, dtype in self.dtype_mapping.items():

            if dtype not in self.VALID_DTYPES:

                raise ValueError(
                    f"Unsupported dtype '{dtype}' "
                    f"for column '{column}'."
                )

    @staticmethod
    def _validate_dataframe(X):

        if not isinstance(X, pd.DataFrame):

            raise TypeError(
                "Input must be a pandas DataFrame."
            )

    def fit(self, X: pd.DataFrame, y=None):
        """
        Data type correction does not learn statistics.

        Present only for sklearn compatibility.
        """

        self._validate_dataframe(X)

        return self

    def transform(self, X: pd.DataFrame):
        """
        Apply data type corrections.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """

        self._validate_dataframe(X)

        X = X.copy()

        for column, dtype in self.dtype_mapping.items():

            if column not in X.columns:

                raise ValueError(
                    f"Column '{column}' not found."
                )

            if dtype == "datetime":

                X[column] = pd.to_datetime(
                    X[column],
                    errors="coerce",
                )

            elif dtype == "boolean":

                X[column] = X[column].astype("boolean")

            elif dtype == "category":

                X[column] = X[column].astype("category")

            elif dtype == "string":

                X[column] = X[column].astype("string")

            elif dtype == "int":

                X[column] = X[column].astype("Int64")

            elif dtype == "float":

                X[column] = X[column].astype("float")

        return X