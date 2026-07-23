# Production Data Quality Report

**Project:** AI Venture Intelligence Platform · **Day 1 — Phase G**
**Deliverable:** Final engineering data-quality document (Parts A–F consolidated)

Severity legend: 🔴 High · 🟠 Medium · 🟢 Low / informational.
All metrics computed on the raw file via `notebooks/01_EDA.ipynb`.

---

## 1. Executive Summary

The dataset is **structurally sound** — zero duplicate rows, a valid primary key, no
constant columns, and only 1.0% missingness. It is **fit for modeling after targeted
preprocessing**. The issues that matter are not missing data; they are:

- 🔴 **Target leakage** — `pledged`, `backers`, and both USD-pledged columns are
  post-event outcomes; ~23k titles literally contain "cancel". These must be quarantined.
- 🔴 **Extreme outliers / skew** — 12–13% of rows flagged per numeric column; skew 70–106.
- 🟠 **MNAR missingness** — the 3,797 `usd pledged` nulls map *exactly* to a corrupt
  `country = 'N,0"'` cohort (3,562 of them `undefined` state).
- 🟠 **Integrity anomalies** — 102 non-positive campaign durations, 3,082 pledge/backer
  mismatches, two USD columns disagreeing in 26% of rows.
- 🟠 **Schema/type gaps** — two date columns stored as strings; 49% memory reclaimable.

**Verdict:** Proceed to Day 2 feature engineering using the **Approved Feature List (§11)**
and the **Preprocessing Strategy (§12)**. No blockers.

## 2. Dataset Overview

| Field | Value |
| ----- | ----- |
| Dataset | Kickstarter Projects (`ks-projects-201801.csv`) |
| Source | Kaggle — Kickstarter crowdfunding platform (2009–2018) |
| Rows | 378,661 |
| Columns | 15 |
| Target | `state` (binary: `successful` = 1 vs `failed` = 0) |
| Deep memory | 195.7 MB |
| Date of Analysis | 2026-07-23 |
| Analyst | Yash (yashraj-a11y) |

## 3. Missing Value Summary  *(Part A)*  🟠

| Column | Missing | % | Severity | Pattern |
| ------ | ------- | - | -------- | ------- |
| `usd pledged` | 3,797 | 1.00% | 🟠 Medium | **MNAR** — see below |
| `name` | 4 | 0.00% | 🟢 Low | Random, trivial |

**Missingness correlation (severity driver):** the 3,797 `usd pledged` nulls are **not
random**. They coincide *exactly* (100% overlap) with the 3,797 rows where
`country = 'N,0"'` — an unknown/garbage country sentinel — and 3,562 of them carry
`state = 'undefined'`. This is **Missing-Not-At-Random**: these rows are a single corrupt
cohort, not scattered gaps.

**Recommendation:** do not impute `usd pledged`. Use the complete `usd_pledged_real`
instead, and treat the `N,0"` / `undefined` cohort as removable noise for supervised
modeling. Fill the 4 `name` nulls with `""`.

## 4. Duplicate Analysis  *(Part B)*  🟢

| Check | Count | Assessment |
| ----- | ----- | ---------- |
| Duplicate rows | 0 | Clean |
| Duplicate `ID` | 0 | Valid primary key (378,661 unique) |
| Reused campaign **names** | 5,121 rows / 2,228 names | Not true duplicates — distinct IDs/dates; one title reused up to 41× |

**Recommendation:** no row de-duplication needed. Reused names are legitimate re-launches;
keep, but note `name` is not a unique identifier.

## 5. Integrity Assessment & Suspicious Records  *(Part B)*  🟠

| Rule | Violations | Interpretation |
| ---- | ---------- | -------------- |
| `duration > 0` (deadline after launch) | 102 | 🔴 Impossible — bad timestamps |
| `duration ≤ 90 days` (platform max) | 29 | 🟠 Out-of-policy window |
| `launched ≠ 1970-01-01` (epoch sentinel) | 7 | 🟠 Placeholder date |
| `goal ≥ $1` | 4 (incl. 2 at $0.01) | 🟠 Implausible target |
| `pledged > 0 ⇒ backers > 0` | 3,082 | 🟠 Money with zero backers |
| `successful ⇒ pledged ≥ goal` (native) | 5 | 🟢 FX-conversion edge cases |
| `usd_pledged_real ≈ usd pledged` (±$1) | 98,423 | 🟠 Two USD columns disagree (26%) |
| Currency↔country: US with non-USD | 0 | ✅ Consistent |
| Currency↔country: USD with non-US | 2,738 | 🟢 Expected (US-based creators abroad) |
| Category↔main_category: 1-to-1 mapping | 8 categories break it | 🟠 Ambiguous taxonomy |

**Category taxonomy issue:** 8 sub-categories map to multiple parents — e.g. `Comedy`
appears under Film & Video / Theater / Publishing / Music; `Web` under Technology /
Journalism. Encode on `main_category` or the (`main_category`,`category`) pair, not
`category` alone.

**Recommendation:** drop/flag the 102 non-positive-duration rows and 7 epoch dates;
standardize on `usd_pledged_real`; treat `usd pledged` as unreliable.

## 6. Datatype Validation  *(Part C)*  🟠

| Column | Stored | Correct? | Fix |
| ------ | ------ | -------- | --- |
| `launched` | `str` | ❌ | → datetime |
| `deadline` | `str` | ❌ | → datetime |
| `state` | `str` | ⚠️ | → encoded target |
| `ID` | `int64` | ⚠️ | identifier — exclude from math |
| `goal`,`pledged`,`usd*` | `float64` | ✅ | — |
| `backers` | `int64` | ✅ | — |
| `category`,`main_category`,`currency`,`country` | `str` | ⚠️ | → `category` dtype |

## 7. Schema Validation  *(Part C)*  🟠

**Constant columns:** none. **Near-constant (>90% one value):** none (`currency`=USD is
78%, `country`=US is 77% — imbalanced but informative).

**Cardinality:**

| Tier | Columns |
| ---- | ------- |
| Identifier / near-unique | `ID` (378,661), `launched` (378,089), `name` (375,764) |
| Continuous | `usd_pledged_real` (106k) … `goal` (8k) |
| High-card categorical | **`category` (159)** — manage before encoding |
| Low-card categorical | `country` (23), `main_category` (15), `currency` (14), `state` (6) |

**Memory optimization:** converting the 5 string categoricals to `category` dtype
reclaims **~96.8 MB (49%)** of the 195.7 MB footprint.

**Proposed schema:**

| Column | Target dtype | Role |
| ------ | ------------ | ---- |
| ID | int64 | drop (identifier) |
| name | string | text feature source |
| category, main_category, currency, country | category | categorical feature |
| launched, deadline | datetime64 | temporal source |
| goal, usd_goal_real | float64 | **approved** numeric |
| pledged, usd pledged, usd_pledged_real, backers | float64/int | **rejected** (leakage) |
| state | category | **target** |

## 8. Outlier Assessment  *(Part D)*  🔴

IQR method (beyond Q3 + 1.5·IQR / below Q1 − 1.5·IQR):

| Column | Outliers | % | Upper Fence | Max | Fence→Max |
| ------ | -------- | - | ----------- | --- | --------- |
| `goal` | 46,613 | 12.31% | 37,000 | 100,000,000 | 2,703× |
| `pledged` | 50,981 | 13.46% | 10,145 | 20,338,986 | 2,005× |
| `backers` | 45,343 | 11.97% | 137 | 219,382 | 1,601× |
| `usd pledged` | 50,919 | 13.45% | 7,560 | 20,338,986 | 2,690× |
| `usd_pledged_real` | 50,578 | 13.36% | 10,078 | 20,338,986 | 2,018× |
| `usd_goal_real` | 45,508 | 12.02% | 35,750 | 166,361,391 | 4,653× |

**Business vs data-error distinction:** these are overwhelmingly **genuine business
outliers** (viral mega-campaigns), not entry errors — the values are internally
consistent. The exceptions flagged as *errors* are the `goal < $1` cases (§5).

**Treatment:** `log1p` transform + winsorize/clip at IQR fences; robust (median/IQR)
scaling. Do **not** delete business outliers.

## 9. Text Quality Assessment  *(Part E)*  🟠

Only one text column exists — `name` (no `desc` column in this dataset).

| Check | Count | Note |
| ----- | ----- | ---- |
| Null | 4 | fill `""` |
| Blank / whitespace-only | 4 | same rows |
| Very short (≤3 chars) | 342 | low information |
| All-caps titles | 13,808 | style signal (`uppercase_ratio`) |
| Non-ASCII titles | 9,612 | multilingual — normalize for NLP |
| **Contains "cancel"** | **23,145** | ⚠️ **outcome leakage in text** |

**NLP readiness:** GOOD after cleaning. Required steps: strip outcome markers
("(Canceled)"), Unicode-normalize, lowercase, remove the 4 blanks. Title-length and
word-count features are already validated (successful titles avg 36 vs 33 chars).

## 10. Leakage Assessment  *(Part F)*  🔴

A feature is **leakage** if its value is unknown at campaign *launch*.

| Feature | Corr w/ target | Known at launch? | Verdict |
| ------- | -------------- | ---------------- | ------- |
| `goal` | −0.025 | ✅ Yes | **Approve** |
| `usd_goal_real` | −0.024 | ✅ Yes | **Approve** |
| `pledged` | +0.110 | ❌ Post-event | **Reject** |
| `usd pledged` | +0.096 | ❌ Post-event | **Reject** |
| `usd_pledged_real` | +0.108 | ❌ Post-event | **Reject** |
| `backers` | +0.126 | ❌ Post-event | **Reject** |
| `state` | (target) | ❌ | **Target only** |

- **Target leakage:** `pledged` / `backers` / `usd_*pledged*` are the *outcome* of a campaign.
- **Temporal leakage:** none among approved features (goal, category, timing are set at launch).
- **Business/text leakage:** the 23,145 "cancel" titles (§9).
- Every engineered feature from rejected columns (`pledge_ratio`, `is_overfunded`,
  `pledged_log`, `log_goal_pledge_gap`) inherits the rejection.

## 11. Feature Risk Matrix

| Feature | Category | Leakage | Risk | Decision |
| ------- | -------- | ------- | ---- | -------- |
| `goal`, `usd_goal_real` | Numeric | None | 🟢 | ✅ Approve (log-transform) |
| `main_category` | Categorical | None | 🟢 | ✅ Approve |
| `category` | Categorical | None | 🟠 (159 levels, 8 ambiguous) | ✅ Approve w/ grouping |
| `country`, `currency` | Categorical | None | 🟢 (imbalanced) | ✅ Approve w/ rare-grouping |
| `launched`,`deadline` → duration/month/weekday | Temporal | None | 🟢 | ✅ Approve |
| `name` → length/word/signals | Text | Conditional | 🟠 (strip "cancel") | ⚠️ Approve after cleaning |
| `pledged`,`usd pledged`,`usd_pledged_real` | Numeric | **Target** | 🔴 | ❌ Reject |
| `backers` | Numeric | **Target** | 🔴 | ❌ Reject |
| `ID` | Identifier | — | 🟢 | ❌ Drop |
| `state` | Target | — | — | 🎯 Target |

**Approved Feature List:** `goal`, `usd_goal_real`, `main_category`, `category`,
`country`, `currency`, temporal features (from `launched`/`deadline`), cleaned `name`
text features.

## 12. Recommended Preprocessing Strategy

1. **Filter** to decided outcomes (`successful`/`failed`); drop the `N,0"`/`undefined` cohort.
2. **Datatypes:** parse `launched`/`deadline`; cast categoricals to `category` dtype.
3. **Integrity:** drop rows with `duration ≤ 0` (102) and epoch dates (7); null `goal < $1`.
4. **Leakage:** remove `pledged`, `usd pledged`, `usd_pledged_real`, `backers`, `ID`.
5. **Transform:** `log1p` on `goal`/`usd_goal_real`; robust scaling.
6. **Encode:** `main_category` one-hot; `category`/`country`/`currency` frequency or grouped.
7. **Text:** strip outcome markers, normalize, derive length/word/signal features.
8. **Target:** binary encode `state`.

## 13. Data Validation Rules

Rules to enforce in the pipeline (fail-fast on ingest):

| # | Rule |
| - | ---- |
| V1 | `ID` unique and non-null |
| V2 | `goal > 0` and `usd_goal_real > 0` |
| V3 | `0 < campaign_duration ≤ 90` |
| V4 | `launched` between 2009 and 2018; not epoch 1970 |
| V5 | `state ∈ {successful, failed, canceled, live, undefined, suspended}` |
| V6 | `country ≠ 'N,0"'` for supervised rows |
| V7 | `name` non-blank after strip |
| V8 | Approved-feature set contains **no** post-event column |

## 14. Engineering Decisions

| Decision | Rationale |
| -------- | --------- |
| Binary target `successful` vs `failed` | Covers 87.6% of rows; ambiguous states excluded |
| Standardize on `usd_pledged_real`, drop `usd pledged` | Latter has 3,797 MNAR nulls & disagrees in 26% of rows |
| Reject `pledged`/`backers` | Post-event target leakage (§10) |
| Encode on `main_category` | `category` has 159 levels & 8 ambiguous mappings |
| `category` dtype for strings | Reclaims ~49% memory (96.8 MB) |
| Keep business outliers, log-transform | Genuine signal, not errors |

## 15. Day 2 Implementation Blueprint

1. Implement validation rules V1–V8 (`src/preprocessing/` + a schema check).
2. Build the cleaning pipeline in the documented order (§12) via existing handlers
   (`missing`, `duplicates`, `datatype`, `outliers`, `transformations`, `encoding`, `scaling`).
3. Assemble the **Approved Feature** matrix; fit `FeatureEngineeringHandler`.
4. Train/test split on decided subset; establish a baseline classifier.
5. Persist the fitted pipeline to `artifacts/`.

## 16. Architecture Mapping

| DQ Finding | Owning Module |
| ---------- | ------------- |
| Missing values (§3) | `src/preprocessing/missing.py` |
| Duplicates (§4) | `src/preprocessing/duplicates.py` |
| Datatype/schema (§6–7) | `src/preprocessing/datatype.py` |
| Outliers (§8) | `src/preprocessing/outliers.py` + `transformations.py` |
| Encoding/cardinality (§7,§11) | `src/preprocessing/encoding.py` |
| Scaling (§8,§12) | `src/preprocessing/scaling.py` |
| Text features (§9) | `FeatureEngineeringHandler` + future `src/features/` |
| Leakage governance (§10–11) | pipeline feature-selection step |
| Reporting | `src/reports/` (this document) |

## 17. Deliverables Checklist

- [x] Part A — Missing Value Report (§3)
- [x] Part B — Integrity Report (§4–5)
- [x] Part C — Dataset Schema Report (§6–7)
- [x] Part D — Outlier Report (§8)
- [x] Part E — NLP Readiness Report (§9)
- [x] Part F — Leakage Report + Approved Feature List (§10–11)
- [x] Part G — Production Data Quality Report (`docs/data_quality_report.md`)

## 18. Conclusion & Bridge to Day 2

The Kickstarter dataset is **clean in structure but requires disciplined governance**:
its greatest risk is **leakage**, not missingness. With the post-event columns rejected,
outliers log-transformed, the MNAR cohort removed, and the schema tightened, the data is
ready for feature engineering and baseline modeling.

**Day 2 begins** with implementing validation rules V1–V8 and the §12 preprocessing
strategy against the Approved Feature List, producing the first model-ready dataset in
`data/processed/`.

---

*Related: [data_dictionary.md](data_dictionary.md) · [statistical_summary.md](statistical_summary.md)
· [eda_observations.md](eda_observations.md) · [business_insights.md](business_insights.md).*
