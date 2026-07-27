





# 🚀 Day 4 — Production-Grade Prediction Engine & Model Benchmarking (Roadmap V1.1 — FROZEN)

> **Status: FROZEN engineering contract** (2026-07-27). Corrections from the draft are flagged 🔧; changelog + improvements at the end. Mandatory **Week 6 (Logistic Regression + regularization)** is fully satisfied here.

## Goal
Build the first **Prediction Engine** (Architecture Layer 4): learn from historical Kickstarter outcomes to predict **funding success** and derive venture-intelligence scores. Deliver a mandatory LogReg baseline, benchmarked ML models, tuning, calibrated probabilities, and a serialized, reusable engine.

## Data reality (Source of Truth) 🔧
```
input:   artifacts/fused_features/X_train_selected.npz   (265,171 × 865, sparse CSR)
         artifacts/fused_features/X_test_selected.npz    ( 66,293 × 865, sparse CSR)
         y_train.npy / y_test.npy                        (funding success: 1=successful, 0=failed)
balance: 40.38% positive (both partitions, stratified)
```
- Train on **`X_selected` (865 features)** — the final Day-3 output — **not** the raw "X_fused" (866). 🔧
- The matrix is already **StandardScaled** (Day-3 Phase E): correct for LogReg, harmless for trees (scale-invariant).
- **Only one real label exists: funding success.** There are NO labels for growth/risk/readiness/founder/market — this constrains Phase G (see below). 🔧

## Environment (resolved) 🔧
- `scikit-learn` 1.9, `xgboost` 3.3.0, `lightgbm` 4.7.0, `matplotlib`, `seaborn`, `joblib` — all installed & functionally tested on sparse CSR.
- macOS OpenMP fix applied: `brew install libomp` + symlink to `~/lib/libomp.dylib` (see requirements.txt).
- **CatBoost SKIPPED** (heavy, marked optional). **reportlab SKIPPED** — reports are **Markdown + PNG**, not PDF. 🔧

---

## Phase A — Prediction Problem Formulation
**Objective:** define the task precisely.
- **Primary target (this day):** funding success — **binary classification**.
- **Future targets (NOT today, no labels):** valuation, growth, failure risk, readiness, scalability, market fit.
**Deliverable:** short problem-formulation note (in the notebook / this doc).

## Phase B — Train / Validation / Test Strategy 🔧
**FROZEN: do NOT re-split.** Day 3 already produced a leakage-free, stratified 80/20 **train/test** with persisted indices, and the feature pipeline was fit on that train. The Day-3 **test set is the untouched final holdout** — touched exactly once, at Phase J.
**Tasks:**
- Carve a **validation set out of the training portion** (e.g. stratified 80/20 of train → ~212k fit / ~53k val), fixed `random_state=42`. Or use **stratified k-fold CV on train** (preferred for tuning/calibration).
- Never fit/tune/calibrate on the test set.
**Deliverable:** `src/models/data_split.py` — a validation-from-train helper that reuses the Day-3 split (does not re-partition test).

## Phase C — Logistic Regression Baseline (Mandatory Week 6) → `src/models/logistic_regression.py`
**Why LogReg:** strong, fast, interpretable baseline; excellent on high-dim sparse (TF-IDF) data; naturally probabilistic.
**FROZEN implementation notes:** 🔧
- L1 / ElasticNet require `solver="saga"` (lbfgs cannot do L1); raise `max_iter` (≈1000) for convergence on 865 sparse features.
- Expose `class_weight` (`None` | `"balanced"`) as a tunable given the 40/60 imbalance.
**Tasks:** implement L2 (mandatory), L1, ElasticNet; compare accuracy, precision, recall, F1, ROC-AUC, **PR-AUC** (primary for imbalance).
**Deliverable:** `LogisticRegressionTrainer`.
**🔧 Improvement:** also fit a `DummyClassifier(strategy="most_frequent")` as the true floor — the honest number LogReg must beat (≈0.596 accuracy by always predicting "failed").

## Phase D — Classical ML Benchmarking → `src/models/benchmarking.py`
**Models:** LogReg (mandatory) + RandomForest, XGBoost, LightGBM, GradientBoosting, ExtraTrees. (CatBoost skipped.)
**Tasks:** train each on `X_selected`; compare **training time**, validation accuracy, recall, F1, ROC-AUC, PR-AUC.
**Deliverable:** `ModelBenchmark` + a Markdown benchmark report (`docs/day4_benchmark_report.md`) + `artifacts/evaluations/benchmark.json`.
**🔧 Note:** evaluate models on the **validation set / CV**, not test.

## Phase E — Hyperparameter Optimization → `src/models/tuning.py`
**FROZEN:** use `RandomizedSearchCV` (not exhaustive grid) with `StratifiedKFold`, `n_jobs=-1`, fixed seed. 🔧 Optionally subsample train for the search to bound runtime.
**Tune:** regularization strength (C / reg_alpha / reg_lambda), tree depth, learning rate, n_estimators.
**Deliverable:** `HyperparameterTuner` + best-config JSON.

## Phase F — Probability Calibration → `src/models/calibration.py`
**Why:** output *"81% funding probability"*, not just a label.
**FROZEN:** calibrate with `CalibratedClassifierCV` (Platt/sigmoid and Isotonic) on **validation / CV — never test**. 🔧 Report Brier score + reliability curve before/after.
**Deliverable:** `ProbabilityCalibrator` + calibration curve PNG.

## Phase G — Venture Scores → `src/models/venture_scores.py` 🔧 REDESIGNED (honesty tiers)
Multi-output scores, split by what the data can honestly support:
- **✅ Derivable now (from the funding model + engineered features), labelled "heuristic, not independently validated":**
  - `funding_probability` = calibrated model output
  - `failure_risk` = 1 − funding_probability
  - `startup_score` / `investment_readiness` = transparent composite of calibrated probability + engineered signals (goal_per_day, campaign_duration, category_freq), documented formula.
- **⏳ DEFERRED (no data basis — do NOT emit fabricated numbers):** `founder_strength`, `market_fit`, `business_model`, `scalability`. The dataset has no founder/team/market/business-model columns. Mark pending real data / retrieval (Layer 2/3), not ✅.
**Deliverable:** `VentureScoreGenerator` returning only the honest scores, each with a stated derivation.

## Phase H — Prediction Engine Integration → `src/models/prediction_engine.py`
Chain: `X (raw campaign) → Day-3 FeaturePipeline.transform → best calibrated model → probability → venture scores → PredictionOutput`.
**🔧 Key:** the engine wraps the **Day-3 `feature_pipeline.joblib`** so a raw campaign flows end-to-end (train/serve consistency — one transform, one model).
**Deliverable:** `PredictionEngine` with `.predict(raw_df) -> {funding_probability, scores, label}`.

## Phase I — Model Serialization → `src/models/serialization.py`
**Save:** best model, calibrated model, hyperparameters, metrics, **and a pointer to the feature-pipeline version** (so model + preprocessing are versioned together). Use `joblib`. 🔧
**Deliverable:** `artifacts/models/{best_model.joblib, calibrated_model.joblib, model_metadata.json}`.

## Phase J — Model Validation Report (evaluate on the untouched test set ONCE) 🔧
**Deliverable:** `docs/day4_model_report.md` + PNGs in `artifacts/evaluations/`:
- model comparison table, best-model selection rationale
- confusion matrix, ROC curve, precision-recall curve, calibration curve
- stratified CV scores (mean ± std), threshold analysis (F1-optimal vs 0.5), brief error analysis
**🔧 Discipline:** the Day-3 test set is scored **exactly once**, here, after all model selection/tuning/calibration is done on train/validation — no test-set peeking during development.

---

## Repository Structure After Day 4
```
src/models/
  data_split.py          logistic_regression.py   benchmarking.py
  tuning.py              calibration.py            venture_scores.py
  prediction_engine.py    serialization.py
notebooks/04_Model_Training.ipynb
```

## Artifacts After Day 4
```
artifacts/
  models/       best_model.joblib  calibrated_model.joblib  logistic_regression.joblib  model_metadata.json
  evaluations/  benchmark.json  confusion_matrix.png  roc_curve.png
                precision_recall_curve.png  calibration_curve.png
docs/           day4_benchmark_report.md  day4_model_report.md
```

## Deliverables Checklist
✅ data_split.py (validation-from-train) · logistic_regression.py (L1/L2/EN + Dummy floor) · benchmarking.py · tuning.py · calibration.py · venture_scores.py (honest tiers) · prediction_engine.py · serialization.py · 04_Model_Training.ipynb · Markdown benchmark + model reports · serialized best/calibrated model + metadata

## Curriculum
Week 6 — Logistic Regression + L1/L2/ElasticNet ✅ (completed today) · Week 7 — Feature Importance ⏳ (Day 5) · Week 8 — Evaluation ⏳

---

## 🔒 Changelog — corrections from the draft
1. **Deps installed** (xgboost 3.3, lightgbm 4.7) with macOS libomp fix; **CatBoost + reportlab dropped**. (blocker)
2. **Reports are Markdown + PNG, not PDF** (no reportlab dependency). (blocker)
3. **No re-splitting** — validation is carved from the Day-3 *train* portion; the Day-3 test set is the untouched final holdout. (leakage)
4. **Train on `X_selected` (865)**, not the ambiguous "X_fused". (clarity)
5. **Calibration on validation/CV, never test.** (leakage)
6. **LogReg L1/EN require `solver="saga"` + higher `max_iter`**; `class_weight` exposed. (executability)
7. **Phase G redesigned into honesty tiers** — only funding_probability / failure_risk / startup+investment composites are emitted (labelled heuristic); founder/market/business-model/scalability **deferred** (no data). (honesty)
8. **Prediction Engine wraps the Day-3 feature pipeline** for true raw→prediction train/serve consistency. (architecture)
9. **Test set scored exactly once, at Phase J**, after all tuning/calibration. (evaluation discipline)

## 💡 Improvements (beyond the draft)
- **Dummy/majority baseline** as the true floor (≈0.596 acc) — the honest number every model must beat.
- **PR-AUC + F1 as primary metrics** (not accuracy) given 40/60 imbalance; report a majority-baseline delta.
- **Threshold analysis** — the F1/recall-optimal decision threshold usually isn't 0.5.
- **Brier score + reliability curve** to *quantify* calibration quality, not just plot it.
- **Free feature-importance preview** — LogReg coefficients cost nothing now and pre-seed Week-7 (Day 5).
- **Everything seeded** (`random_state=42`) and **runtime-logged** per model for reproducibility.
- **`metrics.json`** alongside every report for programmatic downstream use.

✅ **Spec frozen. This is the Day-4 engineering contract.** Say "build it" or "build Phase A/B/C…" to implement + teach phase-by-phase via Framework 1.
