"""Day 16 — Training orchestration.

Reproducible, tracked wrappers around the EXISTING Day 2–4 training modules (``ModelDataSplitter``,
``LogisticRegressionTrainer``, ``HyperparameterTuner``, ``ModelBenchmark``, ``ProbabilityCalibrator``,
``ModelEvaluator``). Nothing here reimplements training — it composes those pieces into a
hyperparameter search, a one-command pipeline, and automated model selection. See
``docs/day16_roadmap.md``.
"""
