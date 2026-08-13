# Day 17 · Phase K — Testing & Validation Reference

*Per-module reference for the test suite. Day 17 extended the original 48-test suite to **91 tests** —
a **fast lane (53)** that runs in CI on fixtures, and a **heavy lane (38)** that needs the real data /
models / 407 MB index and runs locally. Markers, not directories, draw the line. All numbers below are
real, measured on this suite.*

---

## 0. The suite at a glance

| Module | Tests | Lane | Validates |
|---|---|---|---|
| `test_validation.py` | 6 | fast | raw dataset schema (required cols, duplicate ids, state) |
| `test_cleaning.py` | 5 | fast | title/text cleaning |
| `test_missing.py` | 3 | fast | missing-value handling |
| `test_datatype.py` | 3 | fast | dtype coercion |
| `test_duplicates.py` | 4 | fast | duplicate detection |
| `test_integrity.py` | 4 | fast | referential/row integrity |
| `test_pipeline.py` | 5 | fast | preprocessing pipeline composition |
| `test_preprocessing_transforms.py` | 10 | fast | encoding / scaling / log-transform / outliers |
| `test_features.py` | 8 | fast | feature builder / text / TF-IDF / selection |
| `test_data_validation.py` | 6 | fast + 1 heavy | processed dataset + feature-matrix integrity |
| `test_model_validation.py` | 5 | heavy | LightGBM + LogReg quality / overfit / reproducibility |
| `test_retrieval_validation.py` | 3 | heavy | retrieval proxy metrics + self-retrieval |
| `test_explainability.py` | 6 | heavy | SHAP additivity + faithfulness |
| `test_api.py` | 19 | heavy | 10 routes: status / schema / 422 / 503 / latency |
| `test_e2e.py` | 3 | heavy | full `/analyze` pipeline + deferred-honesty survival |
| `test_benchmark.py` | 1 | heavy | latency percentiles + memory/CPU |

**Run:** `pytest -m "not heavy"` (fast, CI, ~4 s) · `pytest` (everything, local) · `pytest -m heavy`
(local artifact tests). Markers registered in `pytest.ini`.

---

## 1. Unit tests (fast lane) — `test_{preprocessing_transforms,features,cleaning,...}.py`

- **Purpose:** verify each Day-2/3 transform in isolation.
- **Scope / inputs:** the `conftest.py` `raw_df` fixture (a 10-row hand-built frame, every row targeting
  one behaviour) + tiny inline frames. **No real data.**
- **Expected outputs / assertions:** *properties*, not constants — StandardScaler → mean ≈ 0; log1p →
  `log1p(0)=0` and rank-preserving; IQR → an extreme value clipped to the fence; TF-IDF width ≤ **800**
  (the real cap, not the draft's fabricated 2,500); FeatureSelector drops a zero-variance column.
- **Edge cases:** unseen category (one-hot ignores), missing configured column (skipped), NaN title,
  empty string, `transform` before `fit`.
- **Failure scenario:** a transform silently changes shape, drops a column, or crashes on a NaN.
- **Example:** `ScalingHandler(cfg).fit_transform(df)["usd_goal_real"].mean()  # ≈ 0`.

## 2. `test_data_validation.py` — processed dataset + feature matrix (fast logic + heavy real-data)

- **Purpose:** validate the *processed* artifacts a model consumes (the raw schema is already covered).
- **Scope:** `build_data_validation_report(df, X, y)` with **severity** — `critical` failures invalidate
  the data; `warning` failures are downstream-handled quality issues.
- **Expected outputs:** `passed` == no *critical* failures + a JSON report → `artifacts/testing/data_validation.json`.
- **Edge cases / findings:** the real dataset has **4 / 331,464** rows with NaN `name_clean` — surfaced
  as a **warning** (TextPreprocessor `fillna("")`s them), not a failure.
- **Failure scenario:** non-binary target, goal ≤ 0, NaN/inf features, or X/y row misalignment (critical).
- **Marker:** fast logic tests (fixtures) + one `heavy` real-dataset test.

## 3. `test_model_validation.py` — model quality *(heavy)*

- **Purpose:** prove the calibrated LightGBM + LogReg baseline meet quality bars. NN **deferred** (none trained).
- **Scope / inputs:** the persisted `calibrated_model.joblib` + fused train/test matrices.
- **Expected outputs:** thresholds set below real actuals (margin for drift) — test **PR-AUC ≥ 0.68**
  (actual 0.6883), **ROC-AUC ≥ 0.77** (0.7763), Brier ≤ 0.20, F1 ≥ 0.63; beats baseline (0.6883 > 0.6264);
  no-overfit gap **< 0.10** (actual 0.040); deterministic inference; re-training reproduces the incumbent.
  → `artifacts/testing/model_validation.json`.
- **Failure scenario:** a retrain or dependency bump that drops PR-AUC below 0.68 or widens the gap past 0.10.
- **Note:** the reproduction test retrains (~145 s) in a repo-relative sandbox; never touches the incumbent.

## 4. `test_retrieval_validation.py` — retrieval quality *(heavy)*

- **Purpose:** validate semantic retrieval **without fabricating similarity labels**.
- **Scope:** reuse the Day-15 `RetrievalEvaluator` (300 held-out queries, one FAISS batch).
- **Expected outputs:** proxy thresholds — category-consistency **lift ≥ 3×** (actual ~8×), kNN-outcome
  **AUC > 0.5** (0.66/0.68), MRR > 0.5, similarity descending, kNN beats majority; plus a **self-retrieval**
  anchor (a stored campaign searched by its own text returns itself at rank 0, similarity ≥ 0.99).
  → `artifacts/testing/retrieval_validation.json`.
- **Edge case:** `similar_to_id` **excludes self** by design → self-retrieval uses `engine.search(text)`.
- **Failure scenario:** an index rebuild that breaks category clustering or the row→campaign mapping.

## 5. `test_explainability.py` — SHAP faithfulness *(heavy)*

- **Purpose:** ensure explanations reflect the real model.
- **Scope:** Day-5 `LocalExplainer` / `SHAPExplainer` on stored test rows.
- **Expected outputs:** **additivity** in log-odds space — `base + Σφ ≈ model_margin` (measured error
  **3.1e-08**); `margin` reconstructs from the *full* SHAP sum; the explanation reports the model's
  *actual* calibrated probability; contribution signs consistent; deterministic.
  → `artifacts/testing/explainability_validation.json`.
- **Failure scenario:** SHAP computed in the wrong space (probability vs margin) so additivity breaks —
  the exact bug the guard + this test catch.

## 6. `test_api.py` — API contract *(heavy)*

- **Purpose:** validate the 10 real routes end-to-end over `TestClient`.
- **Scope:** `/health`, `/predict`, `/explain`, `/retrieve`, `/recommend`, `/scenario`,
  `/counterfactual`, `/report/{founder,investor}`, `/analyze`.
- **Expected outputs:** 200 + schema shape; **422** on fabricated/missing/bad-date/negative-goal input
  (honesty enforced at the boundary); **503** with a typed body when an artifact is unavailable;
  determinism; response times within budget (health < 1 s … analyze < 30 s, ~3× measured warm).
  → `artifacts/testing/api_response_times.json`.
- **Failure scenario:** a schema drift, a missing 422 on a fabricated field, or a hung endpoint.

## 7. `test_e2e.py` — end-to-end integration *(heavy)*

- **Purpose:** one campaign through all 13 layers via `/analyze`.
- **Scope:** the HTTP path (request → normalize → orchestrate → 4 reports) **and** the `ReportBundle`
  (`vi` stages: prediction / drivers / counterfactual / scenario / evolution / cohort / copilot).
- **Expected outputs:** every stage non-empty; committee verdict + confidence ∈ [0, 1]; **the `deferred_*`
  honesty fields survive from the deepest pipeline stage to the report output**.
- **Edge case:** the bundle needs a **normalized** campaign (`CampaignRequest(**c).to_campaign()`), not a raw dict.
- **Failure scenario:** a silently-empty stage behind a 200 response, or aggregation dropping a deferred field.

## 8. `test_benchmark.py` — performance *(heavy)*

- **Purpose:** record production latency + footprint.
- **Expected outputs:** warm p50/p95 — predict **11 ms**, explain **32 ms**, retrieve **15 ms**, report
  **2.15 s**, analyze **3.4 s**; RSS **1.09 GB**; CPU snapshot. → `artifacts/testing/benchmark.json`.
- **Note:** generous sanity budgets (catch a hang / blow-up, don't flake); the 1 GB RSS shows the 407 MB
  embeddings `.npy` is **not** loaded at serve time (only the FAISS index is).

---

## 9. CI — the fast lane (`.github/workflows/ci.yml`)

- **On** push / PR to `main`: install **full** requirements (collection imports every test file, so the
  heavy modules' torch/faiss/mlflow imports must resolve even though those tests are deselected),
  download NLTK corpora, `OMP_NUM_THREADS=1`, run **`pytest -m "not heavy" --cov=src`**.
- **Coverage is reported, not gated** — the fast lane is ~17% because it only exercises fixture-testable
  code; a hard threshold would be a false signal. Heavy tests (which cover the ML pipeline) run locally.

---

## 10. What's deferred (never fabricated)

Neural-Network validation (none trained), Precision@K/Recall@K vs similarity ground truth (no labels),
live drift detection (no data stream), full-artifact CI + at-scale load testing (heavy → local). The
suite tests **properties and thresholds**, and is honest about what has a ground truth to test against.

> **Interview soundbite:** *"I have 91 tests split into a fast lane of 53 that runs in CI on tiny
> fixtures and a heavy lane of 38 that needs the real data, models, and a 407 MB index and runs locally
> — the split is by pytest marker, not directory. Every test asserts a property or a threshold, not a
> frozen constant: the model clears its baseline and reproduces exactly, SHAP is additive to 3e-8, the
> API rejects fabricated fields with a 422, and the deferred-honesty fields survive end to end. And I'm
> honest about coverage — CI reports 17% without gating on it, because the fast lane only covers the
> fixture-testable code and a threshold would go green for the wrong reason."*

---

*Phase K complete. Reference for the 91-test suite (53 fast / 38 heavy) + CI. Numbers measured 2026-08-12.*
