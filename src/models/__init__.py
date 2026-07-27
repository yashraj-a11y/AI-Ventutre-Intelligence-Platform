"""Day-4 modeling package (Architecture Layer 4 — Prediction Engine).

Trains, tunes, calibrates, and serves models on the Day-3 feature matrix. See
``docs/day4_roadmap.md``.
"""

from src.models.benchmarking import BenchResult, ModelBenchmark, classifier_metrics
from src.models.calibration import ProbabilityCalibrator
from src.models.data_split import ModelData, ModelDataSplitter
from src.models.evaluation import ModelEvaluator
from src.models.logistic_regression import LogisticRegressionTrainer, ModelResult
from src.models.prediction_engine import PredictionEngine, PredictionOutput
from src.models.serialization import ModelSerializer
from src.models.tuning import HyperparameterTuner
from src.models.venture_scores import VentureScoreGenerator

__all__ = [
    "ModelData",
    "ModelDataSplitter",
    "LogisticRegressionTrainer",
    "ModelResult",
    "ModelBenchmark",
    "BenchResult",
    "classifier_metrics",
    "HyperparameterTuner",
    "ProbabilityCalibrator",
    "VentureScoreGenerator",
    "PredictionEngine",
    "PredictionOutput",
    "ModelSerializer",
    "ModelEvaluator",
]
