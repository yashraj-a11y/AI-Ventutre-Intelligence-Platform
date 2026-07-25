# 🚀 Day 3 — Feature Engineering & Multi-Modal Representation (Roadmap V1.2 — FROZEN)

> **Status: FROZEN engineering contract** (2026-07-25). Changes are flagged 🔧; changelog at the end.

## Goal
Transform the Day-2 cleaned dataset (`kickstarter_processed.csv`, **331,464 × 12**) into a unified, leakage-free, ML-ready multi-modal feature matrix — engineering, encoding, scaling, vectorizing (titles), fusing, and selecting features. Output feeds the Multi-Modal Understanding Layer and completes Weeks 1–5.

## This is Stage 2 (FROZEN)
Day 2 was Stage 1 (row-changing cleaning). Day 3 is **Stage 2** — row-preserving feature transforms, so it is a real sklearn-compatible pipeline. **No stage here may remove rows** (outliers are capped, not dropped). 🔧

## Dataset reality (Source of Truth) 🔧
```
numeric (2):   goal, usd_goal_real            (correlated → selection drops one)
categorical:   category(153), main_category(15), currency(13), country(21)
text:          name_clean  → ~5.6 words (TITLES, not descriptions)
dates:         launched, deadline             (→ engineered temporal features)
target/label:  target (1/0), state (drop for modeling)
```
There is **no description/blurb column**. All text work is on the short title. Titles are weak signal — set expectations accordingly.

## Curriculum Coverage
Week 1 (text normalization) · Week 2 (TF-IDF) · Week 3 (encode + scale) · Week 4 (sparse vs dense) · Week 5 (fusion). Weeks 6–8 = Day 4+.

---

## Phase 0 — Train/Test Split 🔧 NEW (the leakage centerpiece)
**Objective:** split before fitting anything that learns.
**Why:** every Day-3 component learns statistics — OHE categories, scaler mean/std, TF-IDF idf+vocabulary, selection thresholds. Fitting on the full set leaks test info.
**Tasks:**
- Stratified split on `target` (e.g. 80/20), fixed `random_state`.
- Fit every transformer on **train only**; transform the test set.
- Persist the split indices for reproducibility.
**Deliverables:** `X_train/X_test/y_train/y_test`; split policy documented.

## Phase A — Exploratory Feature Analysis
**Objective:** decide which features need engineering.
**Tasks:** distributions/skew/kurtosis of the 2 numerics; cardinality of the 4 categoricals; `goal ↔ usd_goal_real` correlation; title length/word-count distribution.
**Deliverables:** feature profiling report + candidate-transformation list.

## Phase B — Categorical Encoding → `src/preprocessing/encoding.py`
**FROZEN encoding policy 🔧:**
- One-hot `main_category`, `currency`, `country` (~49 sparse cols).
- Frequency-encode `category` (153 levels; one-hot too wide, and 8 map to multiple parents). Optional target-encoding discussed but not default.
- Output **sparse** (feeds sparse fusion).
- Persist the fitted encoder.
**Deliverables:** `EncodingHandler`.

## Phase C — Numerical Transformation → `src/preprocessing/transformations.py`
**Tasks:** `log1p` on `goal`, `usd_goal_real`, and any skewed engineered numeric (`goal_per_day`); store transform metadata. (Yeo-Johnson optional.)
**Deliverables:** `TransformationHandler`.

## Phase D — Outlier Treatment → `src/preprocessing/outliers.py`
**FROZEN: cap/winsorize only — NEVER remove rows 🔧** (Stage 2 is row-preserving).
**Tasks:** IQR/percentile detection → clip to fences. (Integrity already removed impossible values in Day 2.)
**Deliverables:** `OutlierHandler` (capping).

## Phase E — Feature Scaling → `src/preprocessing/scaling.py`
**Tasks:** `StandardScaler` (Week 3 mandatory) applied **after** log-transform. Also expose RobustScaler/MinMax/MaxAbs.
**Note 🔧:** tree models (Day 4) won't need scaling — scaling is for logistic regression; keep it a toggle. Fit on train, save scaler.
**Deliverables:** `ScalingHandler`.

## Phase F — Domain Feature Engineering → `src/features/feature_builder.py`
**FROZEN: pre-launch, leakage-free features only 🔧.** Approved list:
- `campaign_duration = (deadline − launched).days`
- `goal_per_day = goal / campaign_duration`
- `launch_year/month/weekday/quarter`, `is_weekend_launch`
- `title_length`, `word_count`, `avg_word_length`, `uppercase_ratio`, `has_number`
- `goal_log`
- `category_frequency` (map from train frequencies)

❌ **Excluded (leakage/absent):** funding velocity, funding efficiency, reward tiers, founder activity, pledged/backers anything.
**Deliverables:** `FeatureBuilder`. (Runs FIRST in the pipeline — see Phase L.)

## Phase G — Text Preprocessing (Week 1) → `src/features/text_preprocessing.py`
**Input:** `name_clean` (titles). **Steps:** lowercase → remove punctuation/non-alpha → tokenize → stopword removal (NLTK English + keep domain words) → lemmatize.
**Dependency ✅ RESOLVED (2026-07-25):** `nltk` 3.10.0 and `spacy` 3.8.13 are installed in `.venv`. NLTK data (`stopwords`, `wordnet`, `omw-1.4`, `punkt`, `punkt_tab`) downloaded and verified. Reproducible via `.venv/bin/python scripts/download_nltk_data.py` (handles the macOS framework-Python SSL cert issue via certifi + the wordnet-unzip quirk). spaCy `en_core_web_sm` model is NOT installed — only needed if switching Phase G from the NLTK path to spaCy.
**Deliverables:** `TextPreprocessor`.

## Phase H — TF-IDF Vectorization (Week 2) → `src/features/tfidf_vectorizer.py`
**FROZEN 🔧:** `TfidfVectorizer(max_features≈800, ngram_range=(1,2))` — **not 2500** (titles are ~6 words; 2500 is over-sized). Fit on train, transform test, save vectorizer.
**Deliverables:** `TFIDFVectorizer` + sparse title matrix (N × ~800).

## Phase I — Sparse vs Dense Analysis (Week 4)
**Objective:** understand storage/memory; why you can't `np.concatenate` a sparse TF-IDF with a dense tabular block.
**Deliverables:** sparse-vs-dense comparison report + fusion-strategy note.

## Phase J — Multi-Modal Fusion (Week 5) → `src/features/fusion.py`
**Tasks:** `X_fused = scipy.sparse.hstack([X_tfidf, X_categorical_ohe, X_numeric_scaled])` → one sparse matrix.
**Honest framing 🔧:** "multi-modal" = weak title text + tabular; the tabular block (goal, duration, category, timing) carries most signal.
**Deliverables:** `X_fused` + fusion pipeline.

## Phase K — Feature Selection → `src/features/feature_selection.py`
**Tasks:** `VarianceThreshold`; drop one of `goal`/`usd_goal_real` (correlated) 🔧; mutual-information / L1 ranking. Fit on train.
**Deliverables:** `FeatureSelector`.

## Phase L — Feature Engineering Pipeline → `src/features/pipeline.py`
**FROZEN order 🔧** (feature building first; fit-on-train throughout):
```
X_train (from Phase 0)
  → FeatureBuilder            (domain + temporal + title stats)   [build first]
  → TextPreprocessor          (clean titles)
  ├─ numeric branch:  Transform(log) → OutlierCap → Scale
  ├─ categorical branch: Encode (OHE low-card + freq category)
  └─ text branch:     TF-IDF
  → Fusion (sparse hstack)    → X_fused
  → FeatureSelection
Final: X_fused_selected  (fit on train, applied to test)
```
**Deliverables:** `FeaturePipeline` (sklearn-compatible, row-preserving).

---

## Repository Structure After Day 3
```
src/preprocessing/   encoding.py  transformations.py  outliers.py  scaling.py
src/features/        feature_builder.py  text_preprocessing.py  tfidf_vectorizer.py
                     fusion.py  feature_selection.py  pipeline.py
notebooks/           03_Feature_Engineering.ipynb
```

## Artifacts After Day 3
```
artifacts/
  encoders/         encoder.joblib
  scalers/          scaler.joblib
  vectorizers/      tfidf_vectorizer.joblib
  fused_features/   X_train_fused.npz   X_test_fused.npz
  feature_pipeline/ feature_pipeline.joblib   feature_metadata.json
```

## Deliverables Checklist
✅ Phase-0 stratified split 🔧 · encoding.py · transformations.py · outliers.py (cap) · scaling.py · feature_builder.py (pre-launch only) · text_preprocessing.py · tfidf_vectorizer.py (~800) · fusion.py · feature_selection.py · pipeline.py · 03_Feature_Engineering.ipynb · serialized encoder/scaler/vectorizer/pipeline · sparse TF-IDF (titles) · scaled tabular · X_fused (train + test) · feature_metadata.json

## Curriculum Progress
Week 1 ✅ · Week 2 ✅ · Week 3 ✅ · Week 4 ✅ · Week 5 ✅ · Weeks 6–8 ⏳ (Day 4+)

---

## 🔒 Changelog — what changed on freeze
1. **Phase 0 (train/test split) added** — fit-on-train/transform-test is now the explicit foundation (was missing). (Fix #4)
2. **Text = titles, not descriptions**; TF-IDF max_features 2500 → ~800; weak-signal expectation set. (Fix #1)
3. **Phase F restricted to pre-launch features**; removed funding-velocity/efficiency/reward/founder leakage items. (Fix #2)
4. **Phase L reordered** — FeatureBuilder runs first, then transform/scale/encode/TF-IDF → fuse → select. (Fix #3)
5. **Phase D outliers = cap/winsorize only** (row-preserving Stage 2; no row removal). (Fix #5)
6. **category(153) → frequency-encoded**; one-hot reserved for low-cardinality. (Fix #6)
7. **Feature selection drops one of goal/usd_goal_real** (correlated). (Fix #7)
8. **NLP dependency flagged** — download nltk data / spaCy model before Phase G. (Fix #8)
9. **StandardScaler after log**; noted tree models won't need scaling. (Fix #9)
10. **Stage-2 = row-preserving, sklearn-compatible** made explicit throughout.

✅ **Spec frozen. This is the Day-3 engineering contract.**
