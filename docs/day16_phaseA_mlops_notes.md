# Day 16 · Phase A — MLOps, Experiment Tracking & Model Management (conceptual notes)

*The Production ML Engineering phase begins. These notes recap the concepts and ground them in the
**real, already-trained** models this platform ships — before Day 16 adds the MLOps layer (tracking,
registry, artifact versioning, a reproducible pipeline, automated selection).*

---

## 0. Context — what already exists

Days 2–5 already trained and shipped the models. Day 16 does **not** retrain them; it makes them
**reproducible, tracked, and versioned.** The real, on-disk system:

| Model | Role | Real held-out metrics |
|---|---|---|
| **Logistic Regression** (`logreg_l2`, C=1.0) | mandatory 8-week **baseline** | F1 **0.549**, ROC-AUC **0.727**, PR-AUC **0.626**, acc 0.677 |
| **Calibrated LightGBM** (tuned, sigmoid) | **production incumbent** | PR-AUC **0.688**, ROC-AUC **0.777**, Brier **0.187**, threshold **0.329** |

The LightGBM lifts PR-AUC **+0.062** over the baseline (0.688 vs 0.626) — a real, measured
improvement, which is *exactly* the kind of comparison MLOps exists to record and defend.

And crucially, `artifacts/models/model_metadata.json` **already** pins:

- dataset hash — `sha256 90bf5829…` (the exact `kickstarter_processed.csv` trained on),
- feature-pipeline hash — `sha256 d27e8ae5…` ("model MUST be served with this exact pipeline"),
- environment — `python 3.14.2 / sklearn 1.9.0 / lightgbm 4.7.0`.

So the platform is *already doing reproducibility informally*. Day 16 **formalizes** that instinct
into standard MLOps machinery.

---

## 1. The problem: scripts don't remember

Two weeks after a training sprint, a bare script leaves you unable to answer:

- Which LogReg won — `logreg_l1`, `logreg_l2`, or `logreg_elasticnet`? *(They tie at F1 0.549 but
  differ: L1 took **247 s** and kept **824** non-zero coefficients; L2 took **30 s** with **865**.
  Which did we ship, and why?)*
- Which LightGBM config produced PR-AUC 0.688 — the `num_leaves=127, n_estimators=300, lr=0.05` set?
- Which feature pipeline generated the features? Which decision threshold — 0.329 or something else?
- Which SHAP explainer matches which model? Which model is actually deployed behind `/predict`?

A folder of `.joblib` files can't answer these. **MLOps** does — by recording every run's inputs,
outputs, and lineage as first-class data.

---

## 2. The ML lifecycle vs. traditional software

Traditional software is `code → compile → deploy`: the artifact is a deterministic function of the
code. **ML is not** — the artifact is a function of *code **and** data **and** hyperparameters **and**
randomness*:

```
data → preprocessing → feature engineering → training → evaluation
      → experiment tracking → model registry → deployment
```

Two extra stages (**tracking**, **registry**) exist because the same code on different data/seeds/
configs yields different models. You must record *which* combination produced *which* model — the
compiler never had to.

---

## 3. Experiment tracking — record every run as data

An **experiment** is a training run; **tracking** logs its params, metrics, and artifacts so runs are
comparable and reproducible. What we log is **real, per-model**:

- **params** — LogReg: `penalty`, `C`; LightGBM: `num_leaves=127, n_estimators=300, learning_rate=
  0.05, …` (the actual set, not invented "batch size").
- **metrics** — the six we already compute on the **held-out test split**: accuracy, precision,
  recall, F1, ROC-AUC, PR-AUC.
- **artifacts** — the model, a *reference* to the feature pipeline, the metrics report.

MLflow stores these in a **local file store** (`mlruns/`) — no server required; `mlflow ui` is an
optional local viewer. Every run becomes a queryable row: *"show me the run where PR-AUC = 0.688."*

---

## 4. Reproducibility — the fingerprint

A result is reproducible only if you can pin **everything** that shaped it:

| Dimension | This platform's real anchor |
|---|---|
| **Data** | dataset `sha256 90bf5829…` |
| **Split** | `artifacts/splits/{train,test}_index.npy` (the exact held-out rows) |
| **Feature transform** | feature-pipeline `sha256 d27e8ae5…` |
| **Randomness** | fixed seed |
| **Code** | git commit SHA |
| **Environment** | `python 3.14.2 / sklearn 1.9.0 / lightgbm 4.7.0` |

Together these are a run's **reproducibility fingerprint**. The honest test of it: re-run training and
confirm you land back on **PR-AUC ≈ 0.688** — *verified, not assumed*. (Day 16 asserts this.)

---

## 5. Model & data versioning

Models and datasets get **versions**, like code releases. A model version bundles: the artifact, its
metrics, the dataset + pipeline hashes it depends on, the training date, an owner, and a **status**.
Data versioning here is lightweight and honest — the dataset **hash** *is* its version (full
dataset-version-control / DVC is deferred, §8). If the hash changes, it's a different dataset, and any
model trained on it is a different version.

---

## 6. Model registry & lifecycle

A **registry** is the source of truth for "which model is where." Each entry carries a **status** that
moves through a lifecycle:

```
Development → Staging → Production → Archived
```

- The **calibrated LightGBM** (PR-AUC 0.688, threshold 0.329) is the current **Production** entry —
  it's what `/predict` serves.
- The **Logistic Regression** is the retained mandatory **baseline** — every future candidate must
  beat it to justify itself.

Promotion is **explicit and evidence-based**: a challenger is registered in Development, compared on
real metrics, and only promoted if it genuinely beats the incumbent. The registry makes "why is *this*
model in production?" answerable with numbers, not memory.

---

## 7. Artifact versioning — reference, don't duplicate

Every artifact that shaped a prediction is versioned: preprocessing pipeline, TF-IDF vectorizer,
encoders/scalers, feature pipeline, **vector DB**, prediction models, SHAP explainers. But versioning
≠ blindly copying: the FAISS index is **407 MB** and the embeddings another **407 MB**. Copying those
per run is wasteful and pointless. The honest design is **content-hash reference**: small artifacts
(models, pipelines, metrics JSON) are copied into the run; large ones are registered by **`sha256` +
path** — the same content-hash discipline the platform already uses (Day 6, pattern repository). Two
runs on the same index share one hash, not two 407 MB copies.

---

## 8. The honesty boundary (what Day 16 defers)

MLOps is where scope-creep is most tempting. What we **build** is grounded in real trained models
(LogReg + LightGBM) plus honest new **comparison candidates** (XGBoost, RandomForest — libraries
already installed, trained on the *same* real split). What we **defer, never fabricate**:

- a **Neural Network** "production model" — none exists; torch is for SBERT retrieval, not a trained
  deep classifier. Registering one would be a lie.
- **Bayesian / optuna** hyperparameter optimization — optuna isn't installed; it stays *discussion*.
- a **remote MLflow server** and **model-serving** infra — the store is local files.
- **data-drift monitoring, continuous training, DVC** — real future work, named not built.

Same discipline as every prior layer: track what's real, defer what isn't.

---

## 9. MLOps — the pattern

The Production ML phase turns "a folder of scripts and `.joblib` files" into a **system with memory**:
every run recorded as data, every model versioned with its lineage, every artifact hash-referenced,
one command to reproduce, and a registry that answers "what's in production and why?" with metrics.
The engineering discipline is **honest reproducibility** (pin and *verify*, don't assume) and an
**honest boundary** (wrap the models that exist; defer the ones that don't).

> **Interview soundbite:** *"Day 16 turns the platform from scripts into a reproducible ML system. My
> production model is a calibrated LightGBM at 0.688 PR-AUC, lifting +0.06 over a logistic-regression
> baseline I keep as the mandatory reference. MLOps records that comparison as data: I track every run
> in MLflow's local file store — real params and six test metrics — and pin a reproducibility
> fingerprint of dataset hash, split indices, seed, git SHA, and library versions, so re-running lands
> back on 0.688, verified not assumed. Models live in a registry with a Development-to-Production
> lifecycle, and artifacts are versioned by content hash — I reference the 407 MB FAISS index by
> sha256 rather than copying it per run. The honest part is scope: I wrap the models that exist and
> add XGBoost and RandomForest as real comparison candidates, but I defer a 'neural network' the
> platform never trained and Bayesian tuning I don't have the dependency for — MLOps is exactly where
> you must resist registering things you didn't actually build."*
