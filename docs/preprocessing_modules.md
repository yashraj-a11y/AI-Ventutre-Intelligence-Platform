# Preprocessing Modules — Reference & Contract

Documentation for the Day-2 **Stage-1** preprocessing package (`src/preprocessing/`).
Each module is a single-responsibility handler; the `PreprocessingPipeline` orchestrates
them in the frozen order. This is the **official contract** for the AVIP data-ingestion layer.

## Pipeline order

```text
DatasetValidator          (validate; raise on failure)
  → ColumnCleaner         (snake_case names)
  → TextCleaner           (name_clean; raw name preserved)
  → CategoryNormalizer    (trim category labels)
  → ColumnRemover         (drop id + leakage)
  → TargetBuilder         (filter decided; build binary target)
  → MissingValueHandler   (type-aware fills)
  → DuplicateRemover      (exact-duplicate rows)
  → DataTypeHandler       (datetime / category / float)
  → DataIntegrityValidator(V1–V8; quarantine)
→ processed dataset
```

Row-preserving handlers (Cleaning, Missing, Datatype) use the sklearn `fit`/`transform`
interface. Row-changing handlers (Target, Duplicates, Integrity) are **Stage-1 only** —
they remove rows, so they are not `sklearn.Pipeline` steps.

## Feature contract

- **Approved (pre-launch):** `name`, `name_clean`, `category`, `main_category`,
  `currency`, `goal`, `usd_goal_real`, `country`, `launched`, `deadline`, `state` (label), `target`
- **Rejected (dropped):** `ID` (identifier); `pledged`, `usd pledged`,
  `usd_pledged_real`, `backers` (post-event leakage)
- ⚠️ `usd_goal_real` is kept — it is the *goal* in USD (known at launch), not leakage.

## Integrity rules (V1–V8)

| Rule | Check | Action |
| ---- | ----- | ------ |
| V1 | goal > 0 | drop |
| V2 | duration (days) > 0 | flag (sub-1-day) |
| V3 | duration ≤ 90 | flag |
| V4 | launched < deadline | drop |
| V5 | launched ≠ 1970 epoch | drop |
| V6 | state in valid set | drop |
| V7 | country ≠ `N,0"` | drop (quarantine) |
| V8 | required business fields present | flag |

---

## Module reference

### `validation.py` — `DatasetValidator`
- **Purpose:** verify structural validity against the 15-column schema before any cleaning.
- **Inputs:** raw dataframe.
- **Outputs:** `ValidationReport` (JSON-serializable); optionally raises `DataValidationError`.
- **Example:** `DatasetValidator().validate(df)` / `.assert_valid(df)`
- **Assumptions:** the 15-column Kickstarter schema; `state` is the target column.
- **Limitations:** structural checks only — no business rules (see integrity). Read-only.

### `cleaning.py` — `ColumnCleaner`, `TextCleaner`, `CategoryNormalizer`, `ColumnRemover`
- **Purpose:** canonicalize names, clean the title (preserving raw), trim labels, drop id + leakage.
- **Inputs:** raw/validated dataframe.
- **Outputs:** cleaned dataframe; `name_clean` added; leakage columns removed.
- **Example:** `ColumnRemover().fit_transform(df)`
- **Assumptions:** leakage/identifier lists come from `configs/preprocessing.yaml`.
- **Limitations:** `TextCleaner` does not tokenize/stopword/TF-IDF (Day 3). Row-preserving.

### `missing.py` — `MissingValueHandler`
- **Purpose:** type-aware imputation; `fit` learns fills on train, `transform` applies them.
- **Inputs:** dataframe; strategies from config (numeric median/mean, categorical mode/Unknown, text "").
- **Outputs:** dataframe with no missing values in configured columns; `fill_values_` learned state.
- **Example:** `h.fit(train); h.transform(test)`
- **Assumptions:** `fit` before `transform`; `N,0"` is not missing (handled by integrity).
- **Limitations:** no KNN/iterative imputation. Row-preserving.

### `duplicates.py` — `DuplicateDetector`, `DuplicateRemover`, `DuplicateReporter`
- **Purpose:** detect / remove / report exact duplicate rows.
- **Inputs:** dataframe. **Outputs:** stats dict; de-duplicated dataframe (`keep="first"`).
- **Example:** `DuplicateRemover().transform(df)`
- **Assumptions:** ID uniqueness already checked in Phase A (id dropped by Phase B).
- **Limitations:** exact duplicates only — no fuzzy/near-duplicate detection. Row-changing → Stage 1.

### `datatype.py` — `DataTypeHandler`
- **Purpose:** coerce columns to datetime / category / float / string.
- **Inputs:** cleaned dataframe. **Outputs:** correctly-typed dataframe; `memory_report()`.
- **Example:** `DataTypeHandler().fit_transform(df)`
- **Assumptions:** column→type mapping from config; runs **before** integrity (needs real dates).
- **Limitations:** `errors="coerce"` turns bad values into NaN/NaT (caught by integrity). Row-preserving.

### `integrity.py` — `DataIntegrityValidator`
- **Purpose:** business-rule validation (V1–V8); quarantine/flag invalid rows.
- **Inputs:** typed dataframe (datetimes required). **Outputs:** valid rows; `report()`; quarantine CSV.
- **Example:** `DataIntegrityValidator(mode="train").transform(df)`
- **Assumptions:** `train` mode drops+quarantines; `inference` mode flags via `integrity_valid`, drops nothing.
- **Limitations:** duration computed via `.dt.days` (V2 is a flag, not a drop). Row-changing → Stage 1.

### `pipeline.py` — `TargetBuilder`, `PreprocessingPipeline`
- **Purpose:** build the binary target; orchestrate all Stage-1 handlers with reporting.
- **Inputs:** raw dataframe. **Outputs:** processed dataframe; `get_report()` (per-step rows/timing + sub-reports).
- **Example:** `PreprocessingPipeline(mode="train").fit_transform(df)`
- **Assumptions:** `TargetBuilder` filters to `{successful, failed}`; keeps `state` for integrity V6.
- **Limitations:** not an `sklearn.Pipeline` (row-changing stages). `transform` reuses fitted state.

### `serialization.py` — `PipelineSerializer`
- **Purpose:** persist/restore the fitted pipeline + schema + metadata for reproducibility.
- **Inputs:** fitted pipeline, processed dataframe, raw path.
- **Outputs:** `preprocessing_pipeline.joblib`, `feature_schema.json`, `metadata.json`.
- **Example:** `PipelineSerializer().save(pipe, processed, raw_path=...)` / `.load()`
- **Assumptions:** dataset version = SHA-256 of the raw file.
- **Limitations:** joblib artifacts are Python/version-sensitive; pin the environment for inference.

---

*Utilities: `src/utils/logger.py` (`get_logger`) and `src/utils/config.py` (`load_config`).
Config contract: `configs/preprocessing.yaml`. Tests: `tests/test_*.py`.*
