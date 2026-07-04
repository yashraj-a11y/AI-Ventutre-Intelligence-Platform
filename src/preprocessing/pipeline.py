"""
pipeline.py

Production-ready preprocessing pipeline for the
AI Venture Intelligence Platform.

Author: Yash
"""

from typing import List

import pandas as pd

from .base import BasePreprocessor


class PreprocessingPipeline:
    """
    A custom preprocessing pipeline that chains together
    multiple preprocessing components.

    Parameters
    ----------
    transformers : list[BasePreprocessor]

        Ordered list of preprocessing transformers.

    Example
    -------
    pipeline = PreprocessingPipeline(
        [
            MissingValueHandler(),
            DuplicateHandler(),
            DataTypeHandler(),
            FeatureEngineeringHandler(),
            EncodingHandler(),
            OutlierHandler(),
            TransformationHandler(),
            ScalingHandler(),
        ]
    )
    """

    def __init__(
        self,
        transformers: List[BasePreprocessor],
    ):

        self.transformers = transformers

        self._validate_transformers()

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    def _validate_transformers(self) -> None:
        """
        Ensure every transformer inherits
        from BasePreprocessor.
        """

        if not isinstance(self.transformers, list):

            raise TypeError(
                "transformers must be provided as a list."
            )

        for transformer in self.transformers:

            if not isinstance(
                transformer,
                BasePreprocessor,
            ):

                raise TypeError(

                    f"{transformer.__class__.__name__} "
                    "must inherit from BasePreprocessor."
                )

    @staticmethod
    def _validate_dataframe(
        X,
    ) -> None:

        if not isinstance(
            X,
            pd.DataFrame,
        ):

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
        """
        Fit every transformer sequentially.
        """

        self._validate_dataframe(X)

        X = X.copy()

        for transformer in self.transformers:

            transformer.fit(X, y)

            X = transformer.transform(X)

        return self

    # -------------------------------------------------
    # Transform
    # -------------------------------------------------

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply every fitted transformer sequentially.
        """

        self._validate_dataframe(X)

        X = X.copy()

        for transformer in self.transformers:

            X = transformer.transform(X)

        return X

    # -------------------------------------------------
    # Fit Transform
    # -------------------------------------------------

    def fit_transform(
        self,
        X: pd.DataFrame,
        y=None,
    ) -> pd.DataFrame:
        """
        Fit and transform the dataset.
        """

        self.fit(X, y)

        return self.transform(X)