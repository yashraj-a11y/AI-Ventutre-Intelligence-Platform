\
# 🚀 Day 5 — Production-Grade Explainability Engine (Roadmap V1.1 — FROZEN)

> **Status: FROZEN engineering contract** (2026-07-27). Corrections from the draft flagged 🔧; changelog + improvements at the end. Completes mandatory **Week 7 (Feature Importance & Model Interpretation)** and Architecture **Layer 5**.

## Goal
Make every Day-4 prediction *explainable*: which real features pushed the funding probability up or down, globally and per-campaign, in both technical and founder-readable form. Foundation for Founder/Investor Reports and the Venture Copilot.

## Data / model reality (Source of Truth) 🔧
- **Explanations may only cite REAL features.** The dataset has: `goal`, `usd_goal_real`, `category`/`main_category`, `country`, `currency`, `campaign_duration`, `goal_per_day`, launch timing (year/month/weekday/quarter/weekend), title TF-IDF tokens (800), title stats (length, word_count, avg_word_length, uppercase_ratio, has_number). **There are NO founder/team/market/revenue/retention/competitor features.** Any explanation citing those would be fabricated — forbidden. 🔧
- **865 feature names** in `artifacts/feature_pipeline/feature_metadata.json` (text tokens + tabular).
- **Two models, two tools** 🔧:
  - **LogReg** (`artifacts/models/logistic_regression.joblib`) → coefficient analysis (Phase C, Week 7 mandatory).
  - **Base tuned LightGBM** (`artifacts/models/best_model.joblib`, refit uncalibrated) → SHAP + tree feature importance (B, D, E). SHAP explains the uncalibrated margin; calibration is a monotonic squash on top — attributions remain valid (document it).
- The shipped `calibrated_model.joblib` is for *probabilities*; it is **not** SHAP-able (no `feature_importances_`) — hence the base model.

## Environment (resolved) 🔧
`shap` 0.52.0 installed + functional (TreeExplainer verified on the saved base model), pinned in requirements. matplotlib/lightgbm/sklearn present. `best_model.joblib` saved.

---

## Phase A — Understanding Explainable AI
**Deliverable:** explainability notes (black-box vs white-box, global vs local, interpretability/accuracy trade-off, why VCs need evidence). In notebook/doc.

## Phase B — Global Feature Importance → `src/explainability/global_importance.py`
**Source** 🔧: base LightGBM `feature_importances_` (gain) + mean|SHAP| (subsampled). Permutation importance optional (on calibrated model, subsampled — slow).
**Tasks:** rank all 865 features; split **text vs tabular by NAME membership** (not fixed index — selection dropped `goal`, indices shifted) 🔧; top-50 table.
**Deliverables:** `global_importance.csv` + report.

## Phase C — Logistic Regression Coefficient Analysis (Mandatory Week 7) → `src/explainability/coefficient_analysis.py`
**Model:** `logistic_regression.joblib` (L2). Positive coef → ↑ funding prob; negative → ↓.
**Tasks:** extract 865 coefficients; separate TF-IDF tokens vs tabular; rank top-positive / top-negative.
**Deliverables:** `coefficients.csv`, ranked feature table, report.
**🔧 Improvement:** note where LogReg coefficients and LightGBM SHAP **agree** (cross-model agreement = stronger evidence).

## Phase D — SHAP Explainability Engine → `src/explainability/shap_explainer.py`
**Model:** base LightGBM via `shap.TreeExplainer`.
**Runtime** 🔧: SHAP wants dense → explain a **subsample** (~2,000 rows, densified ≈13 MB) for global/summary; per-instance local is instant. Never densify all 66k×865.
**Sanity** 🔧: assert `expected_value + Σshap ≈ model margin` (additivity).
**Deliverables:** `SHAPExplainer` + `shap_values.pkl` (subsample).

## Phase E — Local Startup Explanation → `src/explainability/local_explainer.py`
**Input:** one campaign. **Output:** funding probability + top positive/negative **real-feature** contributions (SHAP for one row).
**🔧 Every factor is a real feature** (e.g. "modest goal for the category", "launched in a weak month", "title contains 'canceled'") — never founder/market/revenue.
**Deliverables:** per-campaign explanation.

## Phase F — Feature Importance Visualizations (Mandatory Week 7) → `src/explainability/visualization.py`
**Generate PNGs:** feature-importance bar (top +/−), TF-IDF-token importance, tabular importance, SHAP summary (beeswarm), SHAP waterfall (one campaign), decision plot (optional).
**Deliverables:** `artifacts/visualizations/*.png`.

## Phase G — Human-readable Explanation Generator → `src/explainability/explanation_generator.py`
**Tasks:** template SHAP/coefficient contributions into sentences using a **readable-label map** 🔧 (`category_freq → "how common the project category is"`, `goal → "funding goal size"`, TF-IDF `'canceled' → "the word 'canceled' in the title"`, etc.).
**🔧 Honesty caption:** explanations reflect *what the model learned from campaign metadata + titles*, not ground-truth causal drivers.
**Deliverables:** readable explanations, `explanation_examples.json`.

## Phase H — Venture Copilot Explanation Layer → `src/copilot/venture_copilot.py` 🔧 REDESIGNED (honesty tiers)
- **Generate (from real SHAP contributions):** Strengths / Weaknesses / Risks derived from top positive/negative real-feature contributions; funding-probability, startup-score, risk-score explanations (reuse Day-4 scores).
- **DEFER (no data — do NOT fabricate):** Market Analysis, Competitor observations, Founder Intelligence, Opportunities-from-market. Return "pending Layers 2–3 / richer data", never invented text. 🔧
**Deliverables:** initial `VentureCopilot` (real-feature SWOT-lite + score explanations; market/competitor/founder deferred).

## Phase I — Explainability Pipeline → `src/explainability/pipeline.py`
Chain: `raw campaign → PredictionEngine (Day-4) → SHAP + coefficients → readable explanation → Copilot output`. Wraps the Day-4 `PredictionEngine` (train/serve consistency).
**Deliverables:** `ExplainabilityPipeline`.

---

## Repository Structure After Day 5
```
src/explainability/  global_importance.py  coefficient_analysis.py  shap_explainer.py
                     local_explainer.py    visualization.py  explanation_generator.py  pipeline.py
src/copilot/         venture_copilot.py
notebooks/           05_Explainability.ipynb
```

## Artifacts After Day 5
```
artifacts/explainability/  shap_values.pkl  global_importance.csv  coefficients.csv
                           feature_rankings.csv  explanation_examples.json
artifacts/visualizations/  feature_importance.png  coefficient_plot.png  shap_summary.png
                           shap_waterfall.png  startup_explanation.png
artifacts/models/          best_model.joblib   (base tuned LightGBM — SHAP-able)
```

## Deliverables Checklist
✅ global_importance.py · coefficient_analysis.py (Week 7) · shap_explainer.py · local_explainer.py · visualization.py (Week 7) · explanation_generator.py (real-feature labels) · pipeline.py · venture_copilot.py (honesty tiers) · 05_Explainability.ipynb · SHAP + coefficient + importance plots · human-readable explanations

## Curriculum
Week 7 — Feature Importance & Model Interpretation ✅ (completed today) · Week 8 — Evaluation ⏳

---

## 🔒 Changelog — corrections from the draft
1. **shap installed** (0.52.0, functional) + pinned. (blocker)
2. **base LightGBM saved** (`best_model.joblib`, uncalibrated) so SHAP/importance has a tree to attach to — the calibrated wrapper is not SHAP-able. (blocker)
3. **Explanations restricted to REAL features** — every example rewritten off founder/market/revenue/retention (which don't exist) onto goal/category/timing/title. (honesty)
4. **Phase H Copilot split into tiers** — real-feature SWOT-lite generated; market/competitor/founder narratives **deferred**, not fabricated. (honesty)
5. **Two models clarified:** LogReg→coefficients (Week 7), base LightGBM→SHAP. (design)
6. **SHAP on a subsample, densified** (~2k rows), never 66k×865 dense; additivity sanity check. (executability)
7. **Text/tabular split by name**, not fixed index (selection dropped `goal`). (correctness)

## 💡 Improvements (beyond the draft)
- **Readable-label map** so explanations say "funding goal size", not "goal=+0.31".
- **Cross-model agreement**: flag features both LogReg and SHAP rank highly (stronger evidence).
- **SHAP additivity assertion** (expected_value + Σshap ≈ margin) to catch wrong-model/link bugs.
- **Global honesty caption** on every report: explanations = learned associations from metadata + titles, not causal truth.

✅ **Spec frozen.** Say "build it" or "build Phase A/B/C…" to implement + teach phase-by-phase via Framework 1.
