"""
transformations.py

Production-ready feature transformation module for the
AI Venture Intelligence Platform.

Author: Yash
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
)

from .base import BasePreprocessor


class TransformationHandler(BasePreprocessor):
    """
    Handles feature distribution transformations.

    Supported transformations
    -------------------------
    • log
    • boxcox
    • yeojohnson
    • quantile

    Parameters
    ----------
    transformations : dict

        Example

        {
            "goal": "log",
            "campaign_duration": "yeojohnson",
            "backers": "quantile"
        }
    """

    VALID_TRANSFORMATIONS = {
        "log",
        "boxcox",
        "yeojohnson",
        "quantile",
    }

    def __init__(
        self,
        transformations: Optional[Dict[str, str]] = None,
    ):

        self.transformations = transformations or {}

        self.transformers = {}

        self._validate_transformations()

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    def _validate_transformations(self):

        for method in self.transformations.values():

            if method not in self.VALID_TRANSFORMATIONS:

                raise ValueError(
                    f"Unsupported transformation: {method}"
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

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
    ):

        self._validate_dataframe(X)

        self.transformers = {}

        for column, method in self.transformations.items():

            if column not in X.columns:
                continue

            if method == "log":

                continue

            elif method == "boxcox":

                transformer = PowerTransformer(
                    method="box-cox"
                )

                transformer.fit(
                    X[[column]]
                )

            elif method == "yeojohnson":

                transformer = PowerTransformer(
                    method="yeo-johnson"
                )

                transformer.fit(
                    X[[column]]
                )

            elif method == "quantile":

                transformer = QuantileTransformer(
                    output_distribution="normal",
                    random_state=42,
                )

                transformer.fit(
                    X[[column]]
                )

            self.transformers[column] = transformer

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

        for column, method in self.transformations.items():

            if column not in X.columns:
                continue

            if method == "log":

                X[column] = np.log1p(
                    X[column]
                )

            else:

                transformer = self.transformers[column]

                X[column] = transformer.transform(
                    X[[column]]
                )

        return X