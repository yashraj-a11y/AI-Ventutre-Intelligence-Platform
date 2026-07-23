# Preprocessing Blueprint

**Project:** AI Venture Intelligence Platform · **Phase H** — the Day 2 / Day 3
implementation plan. Every decision below is derived from the Day 1 EDA and data-quality
findings and maps to a handler in [`src/preprocessing/`](../src/preprocessing/).

**Inputs:** `data/raw/ks-projects-201801.csv` (378,661 × 15)
**Output:** model-ready train/test splits in `data/processed/`, plus a fitted,
serialized pipeline in `artifacts/`.

> Guiding principle: build a **pre-launch** predictor. Any feature unknown at campaign
> launch (see §10 leakage) is rejected regardless of predictive power.

---

## 0. Pipeline Order

```text
DataType ▶ Duplicates ▶ Missing ▶ Integrity/Filter ▶ FeatureEngineering
        ▶ Outliers ▶ Transformations ▶ Encoding ▶ Scaling
```

Composed by `PreprocessingPipeline`; each stage is a `BasePreprocessor` with
`fit`/`transform`/`save`/`load`. Fit on **train only**, then apply to test.

---

## 1. Missing Value Strategy → `MissingValueHandler`

| Column | Missing | Decision | Rationale |
| ------ | ------- | -------- | --------- |
| `usd pledged` | 3,797 (1.0%) | **Drop the column** | MNAR (= `country='N,0"'` cohort); `usd_pledged_real` is a complete replacement. Also a leakage column (§10). |
| `name` | 4 | **Fill `""`** | Trivial; empty title → length/word features = 0. |
| MNAR cohort rows | 3,797 | **Drop rows** | `country='N,0"'` / `state='undefined'` are corrupt; remove from supervised data. |

**No imputation of monetary values** — imputing skewed, leakage-prone columns adds noise.

## 2. Datatype Strategy → `DataTypeHandler`

- `launched`, `deadline` → **datetime** (`pd.to_datetime`).
- `category`, `main_category`, `currency`, `country` → **`category` dtype** (reclaims ~49% memory).
- `state` → mapped to binary target (§9).
- `ID` → dropped (identifier, not a feature).

## 3. Integrity Filtering → (pipeline filter step)

Apply the validation rules from the DQ report before feature building:

- Keep only decided outcomes: `state ∈ {successful, failed}`.
- Drop `campaign_duration ≤ 0` (102 rows) and `> 90` days (29 rows, out-of-policy).
- Null / drop `launched == 1970` epoch sentinels (7 rows).
- Drop `goal < $1` (4 rows).

## 4. Outlier Treatment → `OutlierHandler` + `TransformationHandler`

- **Do not delete** business outliers (viral campaigns are real signal).
- `log1p` transform on `goal`, `usd_goal_real` (skew 70–78).
- **IQR-clip / winsorize** extreme tails at Q3 + 1.5·IQR before scaling.

## 5. Scaling Strategy → `ScalingHandler`

- **RobustScaler (median / IQR)** — mean/std scaling is corrupted by the heavy tails
  (kurtosis up to ~19,000).
- Fit on train; persist scaler params in the pipeline artifact.

## 6. Encoding Strategy → `EncodingHandler`

| Feature | Levels | Encoding | Why |
| ------- | ------ | -------- | --- |
| `main_category` | 15 | **One-hot** | Low cardinality, interpretable. |
| `category` | 159 | **Frequency or target encoding** | One-hot too wide; 8 levels also map to multiple parents. |
| `country` | 23 | Group rare (<20) → "Other", then one-hot | Highly imbalanced (77% US). |
| `currency` | 14 | Group rare → one-hot, or drop | ~redundant with `country`. |
| `state` | — | Binary target map | Target, not a feature. |

## 7. Text Cleaning Strategy → `FeatureEngineeringHandler` (text group) + future `src/features/`

Applied to `name` (the only text column — no `desc` exists):

1. `fillna("")` the 4 blank titles.
2. **Strip outcome markers** — remove `"(canceled)"`/`"(suspended)"` and similar
   (⚠️ #7 most common token, 23,105×; pure leakage).
3. **Unicode-normalize** (NFKC) — 9,612 titles contain non-ASCII characters.
4. Lowercase; strip surrounding punctuation; collapse whitespace.
5. Keep a raw copy for descriptive features (uppercase ratio) before lowercasing.

## 8. Stopword Strategy

**Evidence** (Phase D word-frequency): the top title tokens are dominated by stopwords —
`the` (101,702), `a` (49,769), `of`, `and`, `for`, `to` — mixed with domain words
(`album`, `film`, `project`, `book`, `game`, `music`).

**Decision:**
- Use the **NLTK English stopword list** as the base.
- **Extend** it with corpus-specific noise: the `-` and `&` tokens, and the outcome
  marker `canceled` (belt-and-suspenders with the §7 strip).
- **Do NOT remove** domain words — they carry the real TF-IDF signal.
- Remove stopwords **only for the TF-IDF / bag-of-words path**; keep raw titles for the
  length/word-count/casing features (those need the full string).

## 9. Tokenization Strategy

**Decision:**
- Baseline (Week 2): **regex word tokenizer** — lowercase, split on non-alphanumeric,
  drop tokens < 2 chars. Deterministic and fast on 378k rows.
- Upgrade path: **spaCy** (`en_core_web_sm`) for lemmatization + linguistic features
  when the text sub-model is built (Weeks 3–5). *(Requires downloading the model —
  spaCy library is installed, the model is not yet.)*
- Handle special tokens seen in EDA: `&` → "and", strip `()`, split hyphenated words.
- **Target encoding of the label** (`state`) → `is_successful` (1 = successful, 0 = failed).

## 10. TF-IDF Readiness

**Status: READY after cleaning (§7–9).** Checklist:

- [x] Single text field identified (`name`); no `desc`.
- [x] Missing/blank handled (4 rows).
- [x] Leakage token identified for removal (`canceled`).
- [x] Stopword + tokenization strategy defined (§8–9).
- [ ] Apply `TfidfVectorizer(max_features, ngram_range=(1,2), stop_words=custom)` — Week 2.
- [ ] Sparse TF-IDF matrix fused with dense tabular features — Week 4–5.

## 11. Leakage Governance (feature approval)

**Approved (pre-launch):** `goal`, `usd_goal_real`, `main_category`, `category`,
`country`, `currency`, temporal features (`campaign_duration`, `launch_month/weekday/quarter`),
cleaned `name` text features.

**Rejected (post-event leakage):** `pledged`, `usd pledged`, `usd_pledged_real`,
`backers`, and every derived feature (`pledge_ratio`, `is_overfunded`, `pledged_log`,
`log_goal_pledge_gap`). `ID` dropped as identifier.

## 12. Feature Engineering Roadmap → `FeatureEngineeringHandler`

| Group | Features | Leakage-safe? |
| ----- | -------- | ------------- |
| Temporal | `campaign_duration`, `launch_year/month/weekday/quarter`, `is_weekend_launch` | ✅ |
| Numeric | `goal_log`, `is_high_goal` | ✅ |
| Text | `name_length`, `word_count`, `avg_word_length`, `uppercase_ratio`, `has_number/!/?` | ✅ (after cleaning) |
| Interaction | `goal_per_day`, `goal × duration`, `goal × month` | ✅ |
| Funding | `pledge_ratio`, `is_overfunded`, `pledged_log`, `log_goal_pledge_gap` | ❌ **leakage — descriptive only** |

---

## Day 2 / Day 3 Execution Plan

**Day 2 — Production preprocessing pipeline**
1. Implement validation rules + integrity filter (§3).
2. Wire `DataType ▶ Duplicates ▶ Missing ▶ Filter` into `PreprocessingPipeline`.
3. Add `Outliers ▶ Transformations ▶ Encoding ▶ Scaling`; fit on train, persist to `artifacts/`.
4. Emit `data/interim/kickstarter_cleaned.csv`.

**Day 3 — Feature engineering + text**
1. Run `FeatureEngineeringHandler` (temporal/numeric/interaction/text).
2. Apply text cleaning (§7) + tokenization/stopwords (§8–9); build TF-IDF (Week 2).
3. Assemble the **Approved Feature** matrix (§11); write `data/processed/` train/test splits.
4. Establish a baseline classifier; report F1 / PR-AUC (not accuracy).

---

*Grounded in [data_quality_report.md](data_quality_report.md), [eda_report.md](eda_report.md),
[statistical_summary.md](statistical_summary.md), and [business_insights.md](business_insights.md).
Handlers live in [`src/preprocessing/`](../src/preprocessing/).*
