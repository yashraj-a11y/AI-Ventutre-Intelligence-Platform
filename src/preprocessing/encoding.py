"""
encoding.py

Production-ready categorical encoding module.

Project: AI_Venture_Intelligence
Author: Yash
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .base import BasePreprocessor


class EncodingHandler(BasePreprocessor):
    """
    Handles categorical feature encoding in a production ML pipeline.

    Supported encoders:
    -------------------
    - label encoding
    - one-hot encoding

    Parameters
    ----------
    encoding_mapping : dict
        Example:
        {
            "country": "onehot",
            "currency": "onehot",
            "state": "label"
        }
    """

    VALID_ENCODERS = {"label", "onehot"}

    def __init__(self, encoding_mapping: Optional[Dict[str, str]] = None):

        self.encoding_mapping = encoding_mapping or {}

        self.label_encoders = {}
        self.onehot_encoder = None
        self.onehot_columns = []

        self._validate_mapping()

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    def _validate_mapping(self):

        for column, encoder in self.encoding_mapping.items():
            if encoder not in self.VALID_ENCODERS:
                raise ValueError(
                    f"Unsupported encoder '{encoder}' for column '{column}'."
                )

    @staticmethod
    def _validate_dataframe(X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

    # -------------------------------------------------
    # Fit
    # -------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None):

        self._validate_dataframe(X)

        for column, encoder in self.encoding_mapping.items():

            if column not in X.columns:
                raise ValueError(f"Column '{column}' not found in dataframe.")

            # -----------------------
            # LABEL ENCODING
            # -----------------------
            if encoder == "label":

                le = LabelEncoder()

                values = X[column].astype(str).fillna("missing")

                le.fit(values)

                self.label_encoders[column] = le

        # -----------------------
        # ONE HOT ENCODING
        # -----------------------
        self.onehot_columns = [
            col for col, enc in self.encoding_mapping.items()
            if enc == "onehot"
        ]

        if self.onehot_columns:

            self.onehot_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )

            onehot_data = X[self.onehot_columns].astype(str).fillna("missing")

            self.onehot_encoder.fit(onehot_data)

        return self

    # -------------------------------------------------
    # Transform
    # -------------------------------------------------

    def transform(self, X: pd.DataFrame):

        self._validate_dataframe(X)

        X = X.copy()

        # -----------------------
        # LABEL ENCODING
        # -----------------------
        for column, encoder in self.encoding_mapping.items():

            if encoder == "label":

                le = self.label_encoders[column]

                values = X[column].astype(str).fillna("missing")

                X[column] = le.transform(values)

        # -----------------------
        # ONE HOT ENCODING
        # -----------------------
        if self.onehot_encoder is not None:

            onehot_data = X[self.onehot_columns].astype(str).fillna("missing")

            encoded = self.onehot_encoder.transform(onehot_data)

            feature_names = self.onehot_encoder.get_feature_names_out(
                self.onehot_columns
            )

            encoded_df = pd.DataFrame(
                encoded,
                columns=feature_names,
                index=X.index
            )

            X = X.drop(columns=self.onehot_columns)
            X = pd.concat([X, encoded_df], axis=1)

        return X