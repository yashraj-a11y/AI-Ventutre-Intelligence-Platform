# Day 17 — Production Testing, Validation & Quality Assurance (FROZEN SPEC V1.1)

**Roadmap V1.1 — Production Quality Engineering.** Build a testing + validation framework that verifies
every Architecture-V1.0 layer, **extending the test suite that already exists** (48 passing tests) —
not rebuilding it — and being honest about what can run in CI vs what needs the local artifacts.

> **🔴 READ THIS FIRST — a real test suite already exists.** `tests/` has **48 passing tests** across
> `test_{api,cleaning,datatype,duplicates,integrity,missing,pipeline,validation}.py` + a `conftest.py`
> (OMP guard + tiny hand-built fixtures). Day 17 **adds** the missing coverage (model/retrieval/
> explainability/e2e/benchmark validation + CI) and **does not reorganize or break** the existing files.

> **Status:** FROZEN 2026-08-11 (V1.1). Build phase-by-phase on `build Phase <X>`; each phase =
> implement **and** teach via Framework 1, grounded in real numbers.

---

## 0. What already exists vs. what's new

| Item | Status | Where |
|---|---|---|
| pytest suite (48 tests) + `conftest.py` (fixtures, OMP guard) | ✅ **DONE** | `tests/` |
| Unit tests: cleaning, datatype, duplicates, missing, validation, integrity, pipeline | ✅ **DONE** | `tests/test_*.py` |
| API tests (subset) | ✅ **DONE** | `tests/test_api.py` |
| Real preprocessing/feature modules to test | ✅ **EXIST** | `src/preprocessing/*`, `src/features/*` |
| Day-15 `RetrievalEvaluator` (honest proxy metrics) | ✅ **DONE** | reuse for Phase E |
| Day-16 `verify_reproduction` (deterministic re-train) | ✅ **DONE** | reuse for Phase D |
| `psutil` (memory/CPU), `httpx` (TestClient) | ✅ **INSTALLED** | Phase I / G |
| **Unit tests for untested modules** (encoding, scaling, transformations, outliers, feature_builder, feature_selection, tfidf, text_preprocessing) | ❌ **NEW** | Day 17 |
| **Data / model / retrieval / explainability validation** | ❌ **NEW** | Day 17 |
| **E2E integration + performance benchmark** | ❌ **NEW** | Day 17 |
| **GitHub Actions CI** + `pytest-cov` | ❌ **NEW** | Day 17 |
| **`docs/day17_testing_reference.md`, `notebooks/17_Testing_Validation.ipynb`** | ❌ **NEW** | Day 17 |

---

## 1. What changed from the draft (changelog + why)

| # | Draft said | 🔧 Frozen correction | Why |
|---|---|---|---|
| **C-1** | Validate **Neural Network** (Phase D) | **Validate the REAL persisted models:** calibrated **LightGBM** incumbent + **LogReg** baseline. Assert thresholds (test PR-AUC ≥ 0.68, ROC-AUC ≥ 0.77), calibration (Brier), no-overfit (train–test gap), reproducibility (reuse Day-16 `verify_reproduction`). **Neural Network → DEFER.** | No NN exists (torch is SBERT-only). Validating a model the platform never trained is fabrication. |
| **C-2** | Retrieval **Precision@K / Recall@K** (Phase E) | **Reuse Day-15 `RetrievalEvaluator` HONEST proxy metrics** — category-consistency@K (lift), kNN-outcome ROC-AUC vs the real `target`, MRR-category. Assert thresholds (cat-lift ≥ 3×, AUC > 0.5). **Reject Precision@K vs relevance labels.** | There are **no** ground-truth "campaign X ~ Y" similarity labels; Precision@K against them = fabrication (established Day 15). |
| **C-3** | Endpoints `/simulate`, `/semantic_search`, `/investor_questions`, `/report` | **Test the REAL routes:** `/health`, `/predict`, `/explain`, `/retrieve`, `/recommend`, `/scenario`, `/counterfactual`, `/report/founder`, `/report/investor`, `/analyze`. Extend `test_api.py`. | Those draft names don't exist. `/simulate`→`/scenario`, `/semantic_search`→`/retrieve`, `/report`→`/report/{founder,investor}`; investor objections live inside `/analyze`. |
| **C-4** | Build `tests/{unit,api,integration,...}/` subtree | **Extend the EXISTING flat `tests/`** — don't move/rename the 8 working files (regression risk). Add new `tests/test_*.py` modules + **pytest markers** (`heavy`) to separate artifact-dependent tests. | The suite works and is committed; reorganizing risks breaking it + CI. Markers, not directories, are what CI needs. |
| **C-5** | "TF-IDF (2,500 features)" | **TF-IDF = 800 features** (`configs`: `max_features: 800`); **fused-selected = 865**. Tests assert the REAL dims. | The config explicitly caps at 800 ("titles ~6 words → 800, NOT 2500"). 2,500 is fabricated. |
| **C-6** | Run all tests in CI (Phase J) | **🔑 CI runs only the FAST lane** (`pytest -m "not heavy"`) — the fixture-based unit + data-validation tests. **Data (`*.csv`), models (`*.joblib`), and the 407 MB FAISS index are all git-ignored → unavailable in CI.** Artifact-heavy tests (`model/retrieval/explainability/api/e2e/benchmark`) are marked `heavy` and run **locally**, documented as such. | GitHub Actions has no access to the gitignored real data/artifacts; rebuilding them (SBERT-encoding 265K campaigns) is impractical in CI. Honest CI ≠ pretending the heavy suite runs there. |
| **C-7** | Reports in a new top-level `reports/` | **Write validation reports as JSON to `artifacts/testing/`** (metrics-as-data), prose to `docs/`. No new top-level `reports/` dir. | Consistent with the platform's `artifacts/` + `docs/` convention. |

**IMPROVEMENTS:** IMP-1 **pytest markers** (`heavy` = needs real artifacts) so CI runs a fast lane and
local runs everything. IMP-2 **validation reports as JSON** in `artifacts/testing/`. IMP-3 **reuse**
`conftest` fixtures, Day-15 `RetrievalEvaluator`, Day-16 `verify_reproduction`, Day-14 `/analyze`
orchestrator — don't reimplement. IMP-4 deterministic (fixed seeds); assert **invariants/thresholds**,
not frozen decimals.

---

## 2. Honesty tiers

**Tier 1 — REAL:** the existing 48 tests, real preprocessing/feature modules, the 10 real API routes,
calibrated LightGBM + LogReg models and their real metrics/thresholds (PR-AUC 0.688, ROC-AUC 0.777,
threshold 0.329), TF-IDF 800 / fused 865, Day-15 retrieval proxy metrics, Day-16 reproducibility.
**Tier 2 — DERIVED (labeled):** new test assertions, benchmark measurements (latency, memory/CPU via
psutil), coverage %. **Tier 3 — DEFERRED, never fabricate:** Neural-Network validation (none exists),
Precision@K/Recall@K vs similarity ground truth (no labels), live **drift detection** (no data stream —
concept only), full-artifact CI + at-scale load/throughput testing (heavy — marked/local).

---

## 3. Phases (build order)

- **A** — **Testing notes** (`docs/day17_phaseA_testing_notes.md`): software vs ML testing, the test
  pyramid, data/model/retrieval/explainability validation, why proxy metrics + threshold assertions
  (not frozen decimals), the git-ignored-artifacts → fast-vs-heavy CI split. Framework 1.
- **B** — **Unit tests (EXTEND):** add `tests/test_*` for untested modules — encoding, scaling,
  transformations, outliers, feature_builder, feature_selection, tfidf_vectorizer, text_preprocessing —
  using `conftest` fixtures (expected output / invalid input / edge case / exception). Fast (no markers).
- **C** — **Data validation:** schema / dtype / range / missing / duplicate / unexpected-category checks
  on the processed dataset + feature matrix + target. Reuse `src/preprocessing/validation.py`. →
  `artifacts/testing/data_validation.json`. (Fixture-based parts fast; real-dataset parts `heavy`.)
- **D** — **Model validation** (`tests/test_model_validation.py`, `heavy`): assert calibrated LightGBM
  meets thresholds (test PR-AUC ≥ 0.68, ROC-AUC ≥ 0.77), Brier calibration, LogReg baseline floor,
  no-overfit (train–test gap bound), **reproducibility via Day-16 `verify_reproduction`**. NN deferred.
  → `artifacts/testing/model_validation.json`.
- **E** — **Retrieval validation** (`tests/test_retrieval_validation.py`, `heavy`): reuse Day-15
  `RetrievalEvaluator`; assert category-consistency lift ≥ 3×, kNN-AUC > 0.5, MRR sane, self-retrieval
  rank-1. No Precision@K vs fake labels. → `artifacts/testing/retrieval_validation.json`.
- **F** — **Explainability validation** (`tests/test_explainability.py`, `heavy`): reuse Day-5
  `LocalExplainer`; assert SHAP **additivity** (contributions + base ≈ model output), top-feature
  stability, sign sanity, explanation matches the prediction it explains.
- **G** — **API tests (EXTEND `test_api.py`, `heavy`):** the 10 real routes — status code, response
  schema (pydantic), error handling (422 on bad input, 503 when artifact absent), response time. FastAPI
  `TestClient`.
- **H** — **E2E integration** (`tests/test_e2e.py`, `heavy`): one campaign through the Day-14 `/analyze`
  orchestrator (predict → explain → retrieve → recommend → scenario → counterfactual → report); assert
  every stage yields output + the honesty `deferred_*` fields survive end to end.
- **I** — **Performance benchmark** (`tests/test_benchmark.py` + script, `heavy`): prediction latency,
  retrieval latency (cite Day-15 warm p50 14 ms + re-measure), SHAP time, report time, memory/CPU
  (psutil). → `artifacts/testing/benchmark.json`.
- **J** — **CI** (`.github/workflows/ci.yml`): GitHub Actions — install deps, `OMP_NUM_THREADS=1`, run
  **`pytest -m "not heavy"`** (fast lane) + coverage (`pytest-cov`). Document that `heavy` tests are
  local-only (no artifacts in CI). Add `pytest-cov` (dev dep); register the `heavy` marker.
- **K** — **Docs** `docs/day17_testing_reference.md`: per test module — purpose / scope / inputs /
  expected outputs / edge cases / failure scenarios / example / marker.
- **Notebook** — `notebooks/17_Testing_Validation.ipynb`: run each validation layer, show the reports +
  benchmark, the fast-vs-heavy split, honesty boundary.

---

## 4. Environment & repo

- **New dep: `pytest-cov` only** (dev, for CI coverage). `psutil` + `httpx` + `pytest` already installed.
  **No new runtime deps.**
- **🔴 `OMP_NUM_THREADS=1`** for any test loading faiss + torch (already set in `conftest.py`).
- **Markers:** register `heavy` (needs real data/models/index — not runnable in CI) in `pytest.ini` /
  `pyproject`. Fast lane = `pytest -m "not heavy"`.
- **New files:** `tests/test_{model_validation,retrieval_validation,explainability,e2e,benchmark,data_validation}.py`
  + new unit tests; extend `tests/test_api.py`; `.github/workflows/ci.yml`; `pytest.ini` (markers);
  `docs/day17_{phaseA_testing_notes,testing_reference}.md`; `notebooks/17_Testing_Validation.ipynb`;
  `artifacts/testing/*.json`. **No changes** to `src/` unless a test finds a real defect.
- **Reports:** JSON → `artifacts/testing/`; prose → `docs/`. **No top-level `reports/` dir** (C-7).

---

## 5. Non-negotiables (frozen)

1. **Extend, don't rebuild** the existing 48-test suite; only fix real defects a test uncovers (C-4).
2. **Validate real models only** — LightGBM incumbent + LogReg baseline; **Neural Network deferred** (C-1).
3. **Honest retrieval validation** — Day-15 proxy metrics, never Precision@K vs nonexistent labels (C-2).
4. **Real routes, real dims** — the 10 actual endpoints; TF-IDF 800 / fused 865 (C-3/C-5).
5. **Honest CI** — fast lane only (`-m "not heavy"`); heavy tests marked + local, because data/models/
   index are git-ignored (C-6).
6. **Assert thresholds/invariants, not frozen decimals**; deterministic; reports as JSON (IMP-2/4).
7. Teach every phase via Framework 1, grounded in real numbers; end with an interview soundbite.

---

*Frozen 2026-08-11. Supersedes the Day-17 draft. Day 17 = extend the existing test suite with model/
retrieval/explainability/e2e/benchmark validation + honest CI, over the real trained platform. Next:
Day 18 — Deployment, Dockerization & Cloud-Ready Architecture.*
