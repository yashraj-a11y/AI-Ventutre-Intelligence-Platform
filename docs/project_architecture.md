# Project Architecture

**AI Venture Intelligence Platform** — an ML system that analyzes crowdfunding
campaigns (Kickstarter) to predict, explain, and reason about venture success.

The codebase follows a layered, modular data-science layout: raw data flows through a
composable preprocessing pipeline into models, whose predictions are then made
explainable, counterfactual, and queryable via retrieval and agent layers.

---

## 1. Repository Layout

```text
AI-Venture-Intelligence-Platform/
├── data/
│   ├── raw/          # Immutable source data (ks-projects-201801.csv)
│   ├── interim/      # Cleaned / featured intermediate CSVs
│   └── processed/    # Model-ready train/test splits
├── notebooks/        # Ordered analysis notebooks (01_EDA … 06_XGBoost)
├── src/              # Reusable Python modules (the platform itself)
├── configs/          # YAML/JSON run + pipeline configuration
├── artifacts/        # Serialized fitted objects (pipelines, encoders)
├── outputs/
│   ├── figures/      # Generated plots
│   └── reports/      # Generated summary CSVs
├── tests/            # Unit / integration tests
├── docs/             # This documentation
├── requirements.txt
└── README.md
```

### Data flow

```text
data/raw ──▶ preprocessing pipeline ──▶ data/interim ──▶ feature engineering
   │                                                          │
   └──────────────── notebooks / EDA ────────────────────────┘
                                                              ▼
                          data/processed (train/test) ──▶ models ──▶ artifacts
                                                              │
                                     ┌────────────────────────┼────────────────────────┐
                                     ▼                        ▼                        ▼
                              explainability            counterfactual            retrieval / KB
                                     └────────────── agents / reports ─────────────────┘
```

---

## 2. The `src/` Package

The source tree names the platform's intended capability areas. `preprocessing` is
implemented; the remaining modules are scaffolded stages of the roadmap (§6).

| Module | Responsibility | Status |
| ------ | -------------- | ------ |
| `src/preprocessing/` | Data cleaning, feature engineering, encoding, scaling — the composable pipeline (§3). | **Implemented** |
| `src/features/` | Higher-level feature store / selection on top of preprocessing. | Scaffold |
| `src/models/` | Model training and inference (baseline, XGBoost, …). | Scaffold |
| `src/explainability/` | Global & local model explanations (e.g. SHAP). | Scaffold |
| `src/counterfactual/` | "What would flip this campaign to successful?" analysis. | Scaffold |
| `src/pattern_mining/` | Recurring success/failure patterns across campaigns. | Scaffold |
| `src/simulation/` | Scenario / what-if simulation of campaign parameters. | Scaffold |
| `src/recommendation/` | Actionable recommendations for a campaign. | Scaffold |
| `src/retrieval/` | Retrieval layer (RAG) over campaigns and knowledge. | Scaffold |
| `src/knowledge_base/` | Structured knowledge / entity store. | Scaffold |
| `src/agents/` | LLM agents orchestrating the layers above. | Scaffold |
| `src/reports/` | Report generation for end users. | Scaffold |
| `src/utils/` | Shared helpers. | Scaffold |

---

## 3. Preprocessing Pipeline (implemented)

Located in [`src/preprocessing/`](../src/preprocessing/). Every handler derives from a
common `BasePreprocessor` contract and is composed by `PreprocessingPipeline`.

### Design: a scikit-learn-style contract

`BasePreprocessor` (ABC) defines the shared interface:

- `fit(X, y=None)` — learn any state from the data.
- `transform(X)` — apply the transformation, returning a new DataFrame.
- `fit_transform(X, y=None)` — convenience combination.
- `save(filepath)` / `load(filepath)` — persist a fitted handler (to `artifacts/`).

This mirrors the scikit-learn transformer API, so handlers are interchangeable and
chainable, and `PreprocessingPipeline` runs them in sequence with DataFrame validation.

### Handlers

| Handler | File | Role |
| ------- | ---- | ---- |
| `DataTypeHandler` | `datatype.py` | Coerce columns to correct dtypes (dates, categories). |
| `DuplicateHandler` | `duplicates.py` | Drop duplicate rows (configurable `keep`). |
| `MissingValueHandler` | `missing.py` | Impute / drop missing values per-column strategy. |
| `OutlierHandler` | `outliers.py` | Clip or remove outliers before scaling. |
| `TransformationHandler` | `transformations.py` | Column-wise math transforms (e.g. log). |
| `FeatureEngineeringHandler` | `feature_engineering.py` | Derive temporal, numeric, text, interaction & funding features (see [data_dictionary.md](data_dictionary.md)). |
| `EncodingHandler` | `encoding.py` | Encode categoricals via a column→strategy mapping. |
| `ScalingHandler` | `scaling.py` | Scale numerics (creates the scaler per configured strategy). |
| `PreprocessingPipeline` | `pipeline.py` | Validate and run an ordered list of handlers. |
| — | `serialization.py` | Save/load helpers for fitted pipeline artifacts. |

### Typical order

```text
DataType ▶ Duplicates ▶ Missing ▶ FeatureEngineering ▶ Outliers
        ▶ Transformations ▶ Encoding ▶ Scaling
```

Fitted pipelines are serialized to `artifacts/` (e.g. `preprocessing_pipeline.joblib`)
so training and inference apply identical transformations.

---

## 4. Notebook Workflow

Numbered notebooks in [`notebooks/`](../notebooks/) drive the end-to-end workflow and
double as the reference usage of the `src/` modules:

| Notebook | Purpose |
| -------- | ------- |
| `01_EDA.ipynb` | Exploratory analysis + data-quality audit; exports to `outputs/`. See [eda_observations.md](eda_observations.md) and [data_quality_report.md](data_quality_report.md). |
| `02_Preprocessing.ipynb` | Build & fit the preprocessing pipeline. |
| `03_Feature_Engineering.ipynb` | Apply and inspect engineered features. |
| `04_Baseline.ipynb` | Baseline model. |
| `05_SHAP.ipynb` | Explainability with SHAP. |
| `06_XGBoost.ipynb` | Gradient-boosted model. |

---

## 5. Conventions

- **Immutable raw data.** Never write to `data/raw/`; derived data goes to
  `data/interim/` then `data/processed/`.
- **`*.csv`, `*.pkl`, `*.joblib` are git-ignored** (see `.gitignore`) — data and
  fitted artifacts are reproducible from code, not versioned.
- **Handlers are stateless until `fit`.** Any learned parameters live on the fitted
  instance and are persisted via `save`/`load`, keeping train and inference consistent.
- **Environment:** dependencies pinned in `requirements.txt`; use the project `.venv`
  (Python 3.14). The Jupyter kernel must point at `.venv/bin/python` (absolute path).

---

## 6. Roadmap

1. **Now:** preprocessing pipeline + EDA / data-quality assessment on the Kickstarter dataset.
2. **Next:** baseline and XGBoost models (`src/models/`), explainability (`src/explainability/`).
3. **Then:** counterfactual, simulation, pattern mining, and recommendation layers.
4. **Later:** retrieval + knowledge base + agents to expose the platform conversationally,
   with `src/reports/` generating user-facing venture-intelligence reports.

---

*Related: [data_dictionary.md](data_dictionary.md) · [eda_observations.md](eda_observations.md)
· [data_quality_report.md](data_quality_report.md) · [statistical_summary.md](statistical_summary.md)
· [business_insights.md](business_insights.md).*
