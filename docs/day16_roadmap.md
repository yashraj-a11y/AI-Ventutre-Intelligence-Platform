# Day 16 — MLOps, Experiment Tracking & Model Management (FROZEN SPEC V1.1)

**Roadmap V1.1 — Production ML Engineering phase begins.** Wrap the ML the platform *already has*
(mandatory Logistic-Regression baseline + calibrated LightGBM incumbent) in a reproducible,
experiment-tracked, versioned MLOps layer — **without retraining-away or overwriting the production
model Days 4–15 depend on.**

> **🔴 READ THIS FIRST — Day 16 is an MLOps LAYER over an EXISTING, TRAINED system.** The training
> logic already exists as reusable modules (`ModelDataSplitter`, `LogisticRegressionTrainer`,
> `HyperparameterTuner`, `ProbabilityCalibrator`, `ModelBenchmark`, `ModelEvaluator`,
> `FeaturePipeline`). The production model is a **calibrated LightGBM** (`artifacts/models/
> calibrated_model.joblib`, threshold **0.329**, test PR-AUC **0.688** / ROC-AUC **0.777**) with the
> mandatory **Logistic-Regression** baseline beside it. Day 16 does **not** reimplement training or
> replace this model — it **orchestrates + tracks + versions + registers** what exists, and adds
> honest model-comparison candidates.

> **Status:** FROZEN 2026-08-05 (V1.1). Build phase-by-phase on `build Phase <X>`; each phase =
> implement/wrap **and** teach via Framework 1, grounded in real numbers.

---

## 0. What already exists vs. what's new (the whole point)

| Item | Status | Where |
|---|---|---|
| Logistic-Regression baseline (mandatory 8-week) | ✅ **TRAINED** | `logistic_regression.joblib`, `logreg_results.json` |
| Calibrated LightGBM (tuned) = **production model** | ✅ **TRAINED** | `calibrated_model.joblib` (21 MB), `model_metadata.json` |
| TF-IDF + sparse/dense fusion + feature pipeline | ✅ **DONE** | `src/features/*`, `artifacts/feature_pipeline/` |
| Train/test split indices | ✅ **DONE** | `artifacts/splits/{train,test}_index.npy` |
| Model comparison harness (`ModelBenchmark`) | ✅ **DONE** | `src/models/benchmarking.py` |
| Metrics (`classifier_metrics`: acc/prec/rec/F1/ROC-AUC/PR-AUC) | ✅ **DONE** | `src/models/benchmarking.py` |
| SHAP explainers, FAISS index (to be *referenced*, not copied) | ✅ **DONE** | Day 5 / Day 6 artifacts |
| **MLflow tracking (local file store `mlruns/`)** | ❌ **NEW** | Day 16 |
| **`experiment_tracker.py`, `model_registry.py`, `artifact_manager.py`, `version_manager.py`** | ❌ **NEW** | `src/mlops/` |
| **`train_pipeline.py`, `hyperparameter_search.py`, `model_selector.py`, `trainer.py`** | ❌ **NEW** | `src/training/` |
| **`docs/day16_mlops_reference.md`, `notebooks/16_MLOps.ipynb`** | ❌ **NEW** | Day 16 |

**So Day 16 = WRAP existing training in MLOps (tracking + registry + artifact versioning +
reproducible one-command pipeline + automated selection) — NOT retrain 5 new models from scratch.**

---

## 1. What changed from the draft (changelog + why)

| # | Draft said | 🔧 Frozen correction | Why |
|---|---|---|---|
| **C-1** | Train LogReg, XGBoost, LightGBM, RandomForest, **Neural Network** | **Baseline = LogReg (exists); incumbent = calibrated LightGBM (exists). ADD as honest comparison candidates: XGBoost (3.3.0 ✅) + RandomForest (sklearn ✅). Neural Network → DEFER (Tier 3).** | No trained NN exists; torch is SBERT-only. Registering a "production Neural Network" would be fabrication. XGBoost/RF are installed and trained on the *same real split* = legitimate comparison. |
| **C-2** | Build `train_pipeline.py`, `trainer.py`, `hyperparameter_search.py` | **ORCHESTRATE the existing Day 2–4 modules; do NOT reimplement training.** Reuse `ModelDataSplitter`/`LogisticRegressionTrainer`/`HyperparameterTuner`/`ProbabilityCalibrator`/`ModelBenchmark`/`ModelEvaluator`/`FeaturePipeline`. | Same "wrap, don't rebuild" discipline as Day 15. Reimplementing risks regressing a working, committed system. |
| **C-3** | "Working **MLflow server**" | **MLflow = LOCAL file-backed tracking store (`mlruns/`, `tracking_uri=file:./mlruns`).** `mlflow ui` is an OPTIONAL local viewer, not a deployed server. **No remote/hosted server.** | The platform is local; a file store is fully reproducible and needs no running service. Don't overclaim infra. |
| **C-4** | Version **every** artifact (incl. 407 MB FAISS index, embeddings) per run | **Content-hash REFERENCE, not copies.** Small artifacts (models, pipelines, metrics JSON) are copied into the run; large artifacts (FAISS index 407 MB, embeddings 407 MB, SHAP explainers) are **registered by `sha256` + path reference**. | Copying 407 MB per run is wasteful and pointless. Reuse the platform's existing content-hash serialization discipline (Day 6 / pattern repo). |
| **C-5** | (implied) retrain & save the model | **NEVER overwrite `artifacts/models/calibrated_model.joblib` in place.** `train_pipeline` writes to **versioned** run locations; the incumbent stays Production until `model_selector` + explicit promotion. **Verify a re-run reproduces the recorded metrics (PR-AUC ≈ 0.688), don't assume.** | Days 4–15 load this exact model. Silent overwrite would break the live platform + destroy reproducibility of the committed results. |
| **C-6** | Log "TF-IDF max_features, learning rate, batch size, ..." for all | **Log only params each model ACTUALLY has** (LogReg: C/penalty; LightGBM: n_estimators/num_leaves/learning_rate/… — the real set in `model_metadata.json`). No "batch size" for models without one. Metrics on the **real held-out test split** only. | Honesty: don't log fabricated/irrelevant hyperparameters or invented metrics. |
| **C-7** | Bayesian Optimization | **Discussion only — optuna is NOT installed (NO new dep for it).** Implement grid + random search over the existing tuner; Bayesian = notes. | Keep deps minimal; the draft itself marks Bayesian as "discussion". |
| **C-8** | `version_manager.py` listed but no phase | **Define it:** the reproducibility fingerprint utility — dataset hash + split hash + seed + git SHA + key library versions → a run "version" string; content-hash helpers. Folded into Phase F. | The tree lists it; give it an honest, concrete job (reproducibility), reusing `src/models/serialization.py` patterns. |
| **C-9** *(Phase B, forced)* | Backend = `file:./mlruns` (C-3) | **Local SQLite backend `sqlite:///mlflow.db` + local `mlartifacts/`.** MLflow **3.15.1** put the plain file store into *maintenance mode* — it raises unless you opt into a deprecated backend. Same frozen INTENT (local, serverless, reproducible, git-ignored), supported mechanism. | Shipping a "production-grade" MLOps day on a deprecated, "no further updates" backend is the wrong lesson. SQLite is still a single local file, no server. |

**IMPROVEMENTS:** IMP-1 **reproducibility fingerprint** (dataset+split+seed+git+env) logged with every
run. IMP-2 **metrics-as-data** (MLflow logs mirrored to `artifacts/reports/*.json`). IMP-3
**deterministic** (fixed seeds; re-run reproduces incumbent metrics within tolerance — asserted).
IMP-4 **content-hash artifact registry** (no giant copies). IMP-5 **file-backed model registry**
(JSON) with honest lifecycle statuses.

---

## 2. Honesty tiers

**Tier 1 — REAL:** LogReg baseline + calibrated LightGBM incumbent, their *actual* hyperparameters
(`model_metadata.json`) and *real* held-out test metrics (PR-AUC 0.688, ROC-AUC 0.777, threshold
0.329), the feature pipeline, train/test split indices + hashes, dataset hash, SHAP explainers &
FAISS index (**by reference**), git SHA, library versions.
**Tier 2 — DERIVED (labeled):** newly-trained XGBoost / RandomForest **comparison candidates**
(trained on the same real split, clearly marked Day-16 comparisons), model-selection scores,
inference-latency & model-size measurements, MLflow run ids.
**Tier 3 — DEFERRED, never fabricate:** Neural Network / deep model (none exists), Bayesian/optuna
tuning (no dep), remote MLflow server & model-serving infra, data-drift monitoring & continuous
training, data-version-control (DVC) — all **future**, named not built.

---

## 3. Phases (build order)

- **A** — **MLOps concept notes** (`docs/day16_phaseA_mlops_notes.md`): ML lifecycle, experiment
  tracking, reproducibility, model/data versioning, model registry & lifecycle — **grounded in the
  REAL platform** (LogReg baseline + calibrated LightGBM incumbent, existing artifacts), not a
  hypothetical "5 models". Framework 1.
- **B** — **`src/mlops/mlflow_setup.py`:** local file-backed MLflow (`tracking_uri=file:./mlruns`,
  experiment name from config), a `mlflow_run(...)` context-manager helper. **New dep: `mlflow`.** No
  server. Verify a run appears under `mlruns/`.
- **C** — **`src/mlops/experiment_tracker.py`:** wrap one training run — log the model's *real* params,
  the six *real* test metrics, artifacts (model + pipeline ref + report), and the **reproducibility
  fingerprint** (dataset/split hashes, seed, git SHA, lib versions). Mirror metrics to
  `artifacts/reports/experiment_*.json` (metrics-as-data).
- **D** — **`src/training/hyperparameter_search.py`:** reuse `ModelBenchmark` + `HyperparameterTuner`.
  Candidates: **LogReg (baseline) + LightGBM (incumbent) + XGBoost + RandomForest**; grid + random
  search; each config → a tracked run. Bayesian = discussion. NN deferred.
- **E** — **`src/mlops/model_registry.py`:** file-backed JSON registry. Register each model with
  metadata (version, training date, metrics, feature-pipeline hash, dataset version, owner, status).
  Statuses: **Development / Staging / Production / Archived**. Register the calibrated LightGBM as the
  current **Production** entry (real metrics), LogReg as the mandatory baseline.
- **F** — **`src/mlops/artifact_manager.py` + `version_manager.py`:** content-hash artifact registry —
  small artifacts copied into `artifacts/{models,pipelines,vectorizers,explainers,reports}/`; large
  artifacts (FAISS index, embeddings, SHAP explainers) **referenced by sha256 + path**.
  `version_manager` builds the reproducibility fingerprint.
- **G** — **`src/training/train_pipeline.py` + `trainer.py`:** ONE reproducible command,
  raw → validate → preprocess → features → TF-IDF → fusion → train → calibrate → evaluate → MLflow
  log → registry → artifacts. **Orchestrates existing modules; MUST NOT overwrite the incumbent;**
  writes versioned outputs; **verifies re-run reproduces PR-AUC ≈ 0.688**. `trainer.py` = single-config
  trainer reused by Phases D & G.
- **H** — **`src/training/model_selector.py`:** reuse the metrics. Primary **F1**, then **PR-AUC**,
  recall, inference speed, model size. **Keep LogReg as the mandatory baseline**; propose the best
  candidate; promotion to Production is **explicit** and only if it genuinely beats the incumbent.
- **I** — **Documentation** `docs/day16_mlops_reference.md`: per component — purpose / inputs /
  outputs / experiment workflow / logging strategy / model lifecycle / artifact versioning / example.
- **Notebook** — `notebooks/16_MLOps.ipynb`: one tracked training run → compare candidates → register
  → select best → show the reproducibility fingerprint & MLflow run; incumbent-vs-candidate honesty.

---

## 4. Environment & repo

- **New dep: `mlflow` only** (add to `requirements.txt`, pin). xgboost 3.3.0 + lightgbm 4.7.0 +
  sklearn 1.9.0 already installed. **NO optuna, NO DVC, NO ChromaDB.**
- **Training path is faiss-free** (lightgbm/xgboost/sklearn only) — like Days 11–13, so no OpenMP
  collision. `artifact_manager` **references** the FAISS index by hash and does **not** load it. Set
  `OMP_NUM_THREADS=1` only in a process that co-loads faiss (none in the training path).
- **New files:** `src/mlops/{mlflow_setup,experiment_tracker,model_registry,artifact_manager,version_manager}.py`,
  `src/training/{train_pipeline,hyperparameter_search,model_selector,trainer}.py`,
  `docs/day16_{phaseA_mlops_notes,mlops_reference}.md`, `notebooks/16_MLOps.ipynb`, `mlruns/`
  (git-ignored), config `day16` block. **No changes** to the working Day 2–15 modules unless a real
  defect is found.
- **Config `day16`:** `mlflow.tracking_uri` (`file:./mlruns`) + `experiment_name`; `registry.path`
  (`artifacts/registry/model_registry.json`); search candidates + grids; selection criteria; seeds.

---

## 5. Non-negotiables (frozen)

1. **Don't retrain-away or overwrite the incumbent** — the calibrated LightGBM (threshold 0.329,
   PR-AUC 0.688) stays Production; `train_pipeline` writes versioned outputs and **verifies
   reproduction**, never overwrites in place (C-5).
2. **Wrap, don't rebuild** — orchestrate the existing training modules (C-2).
3. **Honest model roster** — LogReg baseline + LightGBM incumbent + XGBoost/RandomForest candidates;
   **Neural Network + Bayesian/optuna DEFERRED** (C-1/C-7).
4. **MLflow local file store, not a server; artifacts by content-hash reference, not 407 MB copies**
   (C-3/C-4).
5. **Log only real params/metrics on the real test split; metrics-as-data JSON mirror** (C-6, IMP-2).
6. **Reproducibility fingerprint** (dataset+split+seed+git+env) on every run; deterministic (IMP-1/3).
7. Teach every phase via Framework 1, grounded in real numbers; end with an interview soundbite.

---

*Frozen 2026-08-05. Supersedes the Day-16 draft. Day 16 = MLOps layer (tracking + registry + artifact
versioning + reproducible one-command pipeline + automated selection) over the EXISTING trained
LogReg baseline + calibrated LightGBM incumbent. Next: Day 17 — Testing & Production QA.*
