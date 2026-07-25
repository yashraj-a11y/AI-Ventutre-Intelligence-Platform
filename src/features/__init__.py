"""Day-3 Stage-2 feature engineering package.

Row-preserving, sklearn-compatible transforms that turn the Day-2 cleaned
dataset into a leakage-free multi-modal feature matrix. See ``docs/day3_roadmap.md``.
"""

from src.features.feature_builder import FeatureBuilder
from src.features.feature_selection import FeatureSelector, SelectionResult
from src.features.fusion import FeatureFusion, FusedMatrix
from src.features.pipeline import FeaturePipeline
from src.features.profiling import FeatureProfile, FeatureProfiler, Recommendation
from src.features.splitting import DataSplitter, SplitResult
from src.features.text_preprocessing import TextPreprocessor
from src.features.tfidf_vectorizer import TfidfVectorizerHandler

__all__ = [
    "DataSplitter",
    "SplitResult",
    "FeatureProfiler",
    "FeatureProfile",
    "Recommendation",
    "FeatureBuilder",
    "TextPreprocessor",
    "TfidfVectorizerHandler",
    "FeatureFusion",
    "FusedMatrix",
    "FeatureSelector",
    "SelectionResult",
    "FeaturePipeline",
]
