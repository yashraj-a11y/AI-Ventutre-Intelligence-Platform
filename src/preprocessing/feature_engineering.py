
"""
feature_engineering.py

Production-ready feature engineering module for the
AI Venture Intelligence Platform.

Author: Yash
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base import BasePreprocessor


class FeatureEngineeringHandler(BasePreprocessor):
    """
    Feature Engineering Handler.

    Creates new features from existing columns while preserving structure.

    Feature groups:
    • Temporal Features
    • Numerical Features
    • Text Features
    • Interaction Features
    • Funding Features
    """

    def __init__(
        self,
        temporal: bool = True,
        numeric: bool = True,
        text: bool = True,
        interaction: bool = True,
        funding: bool = True,
    ):

        self.temporal = temporal
        self.numeric = numeric
        self.text = text
        self.interaction = interaction
        self.funding = funding

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    @staticmethod
    def _validate_dataframe(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

    # -------------------------------------------------
    # Fit
    # -------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_dataframe(X)
        return self

    # -------------------------------------------------
    # Transform
    # -------------------------------------------------

    def transform(self, X: pd.DataFrame):
        self._validate_dataframe(X)

        X = X.copy()

        if self.temporal:
            X = self._add_temporal_features(X)

        if self.numeric:
            X = self._add_numeric_features(X)

        if self.text:
            X = self._add_text_features(X)

        if self.interaction:
            X = self._add_interaction_features(X)

        if self.funding:
            X = self._add_funding_features(X)

        return X

    # -------------------------------------------------
    # Temporal Features
    # -------------------------------------------------

    def _add_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:

        required_columns = ["launched", "deadline"]

        if all(col in X.columns for col in required_columns):

            X["launched"] = pd.to_datetime(X["launched"], errors="coerce")
            X["deadline"] = pd.to_datetime(X["deadline"], errors="coerce")

            X["campaign_duration"] = (X["deadline"] - X["launched"]).dt.days

            X["launch_year"] = X["launched"].dt.year
            X["launch_month"] = X["launched"].dt.month
            X["launch_weekday"] = X["launched"].dt.dayofweek
            X["launch_quarter"] = X["launched"].dt.quarter

            X["is_weekend_launch"] = (X["launch_weekday"] >= 5).astype(int)

        return X

    # -------------------------------------------------
    # Numerical Features
    # -------------------------------------------------

    def _add_numeric_features(self, X: pd.DataFrame) -> pd.DataFrame:

        if "goal" in X.columns:
            X["goal_log"] = np.log1p(X["goal"])

            X["is_high_goal"] = (
                X["goal"] > X["goal"].median()
            ).astype(int)

        return X

    # -------------------------------------------------
    # Funding Features (HIGH IMPACT)
    # -------------------------------------------------

    def _add_funding_features(self, X: pd.DataFrame) -> pd.DataFrame:

        if "goal" in X.columns and "pledged" in X.columns:

            X["pledged_log"] = np.log1p(X["pledged"])

            X["pledge_ratio"] = X["pledged"] / (X["goal"] + 1e-6)

            X["is_overfunded"] = (X["pledge_ratio"] >= 1).astype(int)

            X["log_goal_pledge_gap"] = (
                X["goal_log"] - X["pledged_log"]
            )

        return X

    # -------------------------------------------------
    # Text Features
    # -------------------------------------------------

    def _add_text_features(self, X: pd.DataFrame) -> pd.DataFrame:

        if "name" in X.columns:

            names = X["name"].fillna("")

            name_length = names.str.len()
            word_count = names.str.split().str.len().replace(0, np.nan)

            X["name_length"] = name_length
            X["name_word_count"] = word_count.fillna(0)

            X["avg_word_length"] = (
                name_length / word_count
            ).replace([np.inf, -np.inf], np.nan).fillna(0)

            X["uppercase_ratio"] = names.apply(self._uppercase_ratio)

        return X

    # -------------------------------------------------
    # Interaction Features
    # -------------------------------------------------

    def _add_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:

        if "goal" in X.columns and "campaign_duration" in X.columns:

            duration = X["campaign_duration"].replace(0, np.nan)

            X["goal_per_day"] = (
                X["goal"] / duration
            ).replace([np.inf, -np.inf], np.nan).fillna(0)

        if "goal_log" in X.columns and "campaign_duration" in X.columns:
            X["goal_duration_interaction"] = (
                X["goal_log"] * X["campaign_duration"]
            )

        if "goal_log" in X.columns and "launch_month" in X.columns:
            X["goal_month_interaction"] = (
                X["goal_log"] * X["launch_month"]
            )

        return X

    # -------------------------------------------------
    # Utility
    # -------------------------------------------------

    @staticmethod
    def _uppercase_ratio(text: str) -> float:

        if not text:
            return 0.0

        uppercase = sum(c.isupper() for c in text)

        return uppercase / max(len(text), 1)