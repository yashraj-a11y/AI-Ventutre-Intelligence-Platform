"""Day 16 — MLOps layer.

Experiment tracking, model registry, and artifact versioning over the models the platform ALREADY
trained (the Logistic-Regression baseline + the calibrated LightGBM incumbent). Nothing here retrains
or overwrites those models — it records, versions, and registers them. See ``docs/day16_roadmap.md``.
"""
