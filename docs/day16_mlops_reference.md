# Day 16 · Phase I — MLOps Component Reference

*Per-component reference for the MLOps layer (Layer 16). Everything here **wraps** the models the
platform already trained — the mandatory Logistic-Regression baseline and the calibrated LightGBM
incumbent — with tracking, versioning, a registry, a reproducible pipeline, and automated selection.
All numbers are **real, measured** during this build.*

---

## 0. System at a glance

| Component | File | Role |
|---|---|---|
| `MLflowSetup` | [src/mlops/mlflow_setup.py](../src/mlops/mlflow_setup.py) | local, serverless MLflow backend |
| `VersionManager` | [src/mlops/version_manager.py](../src/mlops/version_manager.py) | content hashing + reproducibility fingerprint |
| `ExperimentTracker` | [src/mlops/experiment_tracker.py](../src/mlops/experiment_tracker.py) | log one run as data |
| `ModelRegistry` | [src/mlops/model_registry.py](../src/mlops/model_registry.py) | source of truth for what's deployed |
| `ArtifactManager` | [src/mlops/artifact_manager.py](../src/mlops/artifact_manager.py) | version artifacts (copy small / reference large) |
| `SingleModelTrainer` | [src/training/trainer.py](../src/training/trainer.py) | train one config end-to-end |
| `TrainingPipeline` | [src/training/train_pipeline.py](../src/training/train_pipeline.py) | one reproducible command |
| `HyperparameterSearch` | [src/training/hyperparameter_search.py](../src/training/hyperparameter_search.py) | compare families, each tracked |
| `ModelSelector` | [src/training/model_selector.py](../src/training/model_selector.py) | rank + gated promotion |

**Real anchors from this build:** incumbent calibrated LightGBM test **PR-AUC 0.6883 / ROC-AUC 0.7763**,
threshold **0.329**; LogReg baseline F1 **0.6346** (floor); best challenger (XGBoost, fair threshold)
F1 **0.6576** → **keep incumbent**; **776.9 MB saved** by referencing the FAISS index + embeddings
instead of copying; deterministic run version **`lightgbm-7711b9fc22ae`**.

**🔴 New dep:** `mlflow==3.15.1` only. **Training path is faiss-free** (no OpenMP collision); the
artifact manager *references* the FAISS index by hash and never loads it.

---

## 1. `MLflowSetup` — local, serverless tracking backend

- **Purpose:** point MLflow at a **local** backend — no server to run.
- **Inputs:** config `day16.mlflow.{tracking_uri, artifact_location, experiment_name}`.
- **Outputs:** a configured tracking store + `run(...)` / `mlflow_run(...)` context managers.
- **Logging strategy:** SQLite backend `sqlite:///mlflow.db` + local `mlartifacts/`. *(The plain file
  store is deprecated in MLflow 3.15 — C-9; SQLite is still one local file, no service.)* `mlflow ui`
  is an optional local viewer.
- **Example:** `with mlflow_run("train-x") as r: mlflow.log_metric("pr_auc", 0.688)`.
- **Assumptions/limitations:** single-node, local; a remote server is deferred (Tier 3).

---

## 2. `VersionManager` — content hashing + reproducibility fingerprint

- **Purpose:** the canonical home for file hashing and the reproducibility fingerprint (consolidated in
  Phase F so the tracker and registry share one implementation).
- **Inputs:** config (dataset path, split dir, `day16.random_state`).
- **Outputs:** `sha256_file`, `file_reference→{path,sha256,size_bytes}`, `fingerprint()`,
  `version_string()`.
- **Artifact versioning:** streamed SHA-256 (a 407 MB index hashes without loading into memory).
- **Reproducibility fingerprint (the six dimensions):** `dataset` sha256 · `split` train/test sha256 ·
  `seed` · `git` {commit, dirty} · `libraries` (python/numpy/pandas/scipy/sklearn/lightgbm/xgboost/mlflow).
  `version_string` = a deterministic 12-char id over the stable parts (e.g. `7711b9fc22ae`).
- **Example:** `VersionManager().version_string()  # -> '7711b9fc22ae'`.
- **Limitations:** the `dirty` flag is recorded but excluded from the version id (uncommitted work is
  flagged, not silently versioned).

---

## 3. `ExperimentTracker` — log one run as data

- **Purpose:** turn a training run into reproducible evidence.
- **Inputs:** `track(run_name, params, metrics, model_type, artifacts, references, tags, extra)`.
- **Outputs:** an MLflow run + a **metrics-as-data mirror** at `artifacts/reports/experiment_<name>.json`
  and a `fingerprint_<name>.json`; returns the report dict (with `run_id`).
- **Logging strategy:** log only *real* params per model (no invented "batch size"); the six real
  metrics; the fingerprint as tags + an uploaded JSON. **Small artifacts uploaded; large artifacts
  logged by content-hash reference** (Day-14/16 C-4) — a 407 MB index adds ~0 bytes to the store.
- **Experiment workflow:** open run → log params/metrics → set fingerprint tags → upload small
  artifacts → mirror JSON.
- **Example:** `ExperimentTracker().track(run_name="lgbm", params={...}, metrics={"pr_auc":0.688})`.
- **Limitations:** references record hashes, not bytes — the referenced file must remain reachable.

---

## 4. `ModelRegistry` — source of truth for what's deployed

- **Purpose:** answer "which model is in production, trained on what, scoring how much?"
- **Inputs:** `register(...)`, `set_status/promote(name, version, status)`, `get/production/list/history`.
- **Outputs:** a versioned JSON registry at `artifacts/registry/model_registry.json`.
- **Model lifecycle:** `Development → Staging → Production → Archived`, with a **single-Production-per-slot
  invariant** — promoting a version **auto-archives** the incumbent, and every transition is appended to
  `status_history`.
- **Logging strategy:** each entry carries metrics, real params, content-hash provenance (artifact /
  feature-pipeline / dataset), owner, role, status. Registered here: `funding_success_classifier` slot
  = {`logreg-l2-baseline` (Archived, role=baseline), `lightgbm-1.0.0` (Production, role=incumbent)}.
- **Example:** `ModelRegistry().production("funding_success_classifier")["version"]  # -> 'lightgbm-1.0.0'`.
- **Note (real finding):** the incumbent entry carries a `pipeline_parity_flag` — the served
  feature-pipeline hash drifted from the recorded one (a re-serialization), and prediction parity was
  **verified byte-identical** over 66,293 rows, so the drift is serialization-only (`behavioral_parity_verified: true`).

---

## 5. `ArtifactManager` — version artifacts (copy small / reference large)

- **Purpose:** version every artifact that shapes a prediction, without duplicating giant files.
- **Inputs:** `register(path, category, name, force_reference)`; categories = models / pipelines /
  vectorizers / explainers / reports.
- **Outputs:** an index at `artifacts/registry/artifact_index.json`; `get/list/verify/describe`.
- **Artifact versioning (the rule):** ≤ `max_copy_mb` (50) → **copied** into `artifacts/store/<category>/`;
  larger → **referenced** by sha256 + path. Measured: model (20.5 MB) + pipeline copied; FAISS index +
  embeddings (388 MB each) referenced → **776.9 MB saved**; store on disk 21 MB with no FAISS in it.
- **Example:** `ArtifactManager().register("artifacts/vector_db/faiss.index", "vectorizers")  # referenced`.
- **`verify(name, category)`** re-hashes on demand — the same tripwire that surfaced the pipeline drift.

---

## 6. `SingleModelTrainer` — train one config end-to-end

- **Purpose:** the atom of the pipeline — fit (+ optionally calibrate) one config and score it on test.
- **Inputs:** `train(data, family, params, calibrate, out_dir) → TrainResult`.
- **Outputs:** `TrainResult{family, calibrated, threshold, val_metrics, test_metrics, model, ...}`.
- **Reuse + C-5 safety:** the incumbent path reuses `ProbabilityCalibrator` (reproduces the calibrated
  LightGBM); its `save()` is **never called** and its plot path is **redirected** to the run dir, so the
  incumbent's files are never touched.
- **Example:** `SingleModelTrainer().train(data, family="lightgbm", calibrate=True).test_metrics["pr_auc"]`.
- **Limitations:** calibration is LightGBM-specific (the incumbent path); other families train uncalibrated.

---

## 7. `TrainingPipeline` — one reproducible command

- **Purpose:** load → train (+calibrate) → evaluate → save **versioned** → track → register (Development).
- **Inputs:** `run(family, calibrate, register, track)`, `verify_reproduction(family)`.
- **Outputs:** a report (version, test metrics, `reproduces_incumbent`, `run_id`); versioned outputs at
  `artifacts/training_runs/<version>/`.
- **The two guarantees (C-5):** **never overwrites the incumbent** (verified: model/config/plot hashes
  unchanged after a run); **reproduction verified** — re-training from config gave test PR-AUC **0.6883**
  vs incumbent 0.6884, and `verify_reproduction` trained twice for **identical** metrics.
- **Model lifecycle:** registers the new model as **Development** — promotion is deliberate (Phase H),
  never automatic.
- **Example:** `TrainingPipeline().run()["reproduces_incumbent"]  # -> True`.

---

## 8. `HyperparameterSearch` — compare families, each tracked

- **Purpose:** answer "which model / config?" with evidence, each candidate a tracked run.
- **Inputs:** `run(data, track)`; config `day16.search.{candidates, search_family, n_iter, cv}`.
- **Outputs:** ranked candidates → `artifacts/search/hyperparameter_search.json`.
- **Reuse:** `ModelBenchmark` (tree families) + `LogisticRegressionTrainer` (baseline) +
  `HyperparameterTuner` (randomized search) — with the tuner's output **redirected** so the incumbent's
  `best_config.json` is never clobbered. Grid = trivial swap; **Bayesian/optuna deferred**.
- **Example:** `HyperparameterSearch().run()["best"]["name"]`.

---

## 9. `ModelSelector` — rank + gated promotion

- **Purpose:** pick the best model fairly and promote it **only if it beats the incumbent**.
- **Inputs:** `run(families)`, `evaluate_family`, `rank`, `recommend`, `apply(decision, version, registry)`.
- **Outputs:** a decision → `artifacts/selection/model_selection.json`.
- **Fair comparison:** each model scored at **its own F1-optimal threshold** (tuning lifted XGBoost F1
  from 0.607 to 0.6576); rank by F1 → PR-AUC → recall, then footprint (latency, size).
- **Model lifecycle:** promotion is **explicit and gated** — `apply()` promotes only when recommended,
  else a no-op. Measured decision: incumbent 0.6664 > XGBoost 0.6576 > RandomForest 0.6526 (**1023 MB**)
  > baseline 0.6346 → **keep incumbent**.
- **Example:** `ModelSelector().run(["xgboost","random_forest"])["recommendation"]  # -> 'keep_incumbent'`.

---

## 10. The end-to-end workflow

```
raw data (Day-3 features, fingerprinted)
   └─ TrainingPipeline.run()
        ├─ SingleModelTrainer.train()           # fit + calibrate (reuses Day-4 modules)
        ├─ save versioned  artifacts/training_runs/<version>/
        ├─ ExperimentTracker.track()            # MLflow + fingerprint + metrics-as-data
        ├─ ModelRegistry.register(status=Development)
        └─ ArtifactManager.register()           # copy small / reference large
   HyperparameterSearch.run()  → compare families (each tracked)
   ModelSelector.run()         → fair rank → recommend keep/promote → apply() (gated)
```

---

## 11. Honesty boundary (deferred, never fabricated)

Same discipline as every layer. **Deferred (Tier 3):** a **Neural Network** production model (none
trained; torch is SBERT-only), **Bayesian/optuna** tuning (no dependency), a **remote MLflow server** +
model-serving infra, **data-drift monitoring / continuous training**, and **DVC**. What's built wraps
the models that *exist* (LogReg baseline + calibrated LightGBM) plus honest XGBoost / RandomForest
comparison candidates on the same real split.

> **Interview soundbite:** *"Day 16 turns the platform into a reproducible ML system without touching
> the production model. Every run is tracked with a reproducibility fingerprint — dataset hash, split,
> seed, git SHA, library versions — so re-training lands back on 0.688 PR-AUC, verified not assumed.
> Models live in a registry with a Development-to-Production lifecycle and a single-Production
> invariant; artifacts are versioned by content hash, referencing a 407 MB index instead of copying it
> to save 777 MB; and selection is fair (each model at its own optimal threshold) and gated (promote
> only on a real margin) — which honestly recommended keeping the incumbent. The whole day is built on
> resisting scope creep: I wrapped what exists and deferred the neural network I never trained."*

---

*Phase I complete. Reference for the five `src/mlops/` + four `src/training/` modules. Numbers measured
2026-08-11.*
