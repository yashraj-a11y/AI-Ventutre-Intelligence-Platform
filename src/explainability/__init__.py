"""Layer 5 — Explainability Engine (Day 5).

Explains Day-4 funding predictions: which real features push the funding
probability up or down, globally and per-campaign, in technical and
founder-readable form. Two models, two tools:
  - LogisticRegression -> coefficient analysis (white-box, Week 7)
  - base tuned LightGBM -> SHAP + tree feature importance (black-box)
"""
