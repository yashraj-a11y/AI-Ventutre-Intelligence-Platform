# Data Quality Report

Source: **Kickstarter Projects** (`ks-projects-201801.csv`). Figures below come from
`notebooks/01_EDA.ipynb` and the exports in `outputs/reports/`.

## Dataset Overview

| Field | Value |
| ----- | ----- |
| Dataset Name | Kickstarter Projects |
| Source | Kaggle — `ks-projects-201801.csv` (Kickstarter crowdfunding platform, 2009–2018) |
| Number of Rows | 378,661 |
| Number of Columns | 15 |
| Target Variable | `state` (binary framing: `successful` = 1 vs `failed` = 0) |
| Date of Analysis | 2026-07-23 |
| Analyst | Yash (yashraj-a11y) |

## Dataset Shape
- Rows: **378,661**
- Columns: **15**
- Memory: ~155.6 MB in-memory

## Data Types
- Numeric (7): `ID`, `goal`, `pledged`, `backers`, `usd pledged`, `usd_pledged_real`, `usd_goal_real`
- Categorical (5): `category`, `main_category`, `currency`, `country`, `state`
- Text (1): `name`
- Dates (2): `deadline`, `launched` _(stored as strings — need parsing)_

## Missing Values
| Column | Count | Percentage |
|--------|-------|------------|
| usd pledged | 3,797 | 1.0% |
| name | 4 | 0.0% |
| _all other columns_ | 0 | 0.0% |

## Duplicate Records
- Number of duplicate rows: **0**
- Duplicate IDs: **0** (`ID` is fully unique — 378,661 distinct of 378,661 rows)

## Constant Columns
- **None.** Every column has >1 distinct value; the lowest-cardinality columns are
  `main_category` (15) and `currency` (14).

## High Cardinality Features
- `ID` — 378,661 unique (identifier; drop for modeling)
- `launched` — 378,089 unique (near-unique timestamp)
- `name` — 375,764 unique (effectively an identifier; use as **text**, not categorical)
- `usd_pledged_real` — 106,065 unique
- `usd pledged` — 95,455 unique
- `pledged` — 62,130 unique
- `usd_goal_real` — 50,339 unique
- `goal` — 8,353 unique
- **`category` — 159 unique** ← the one high-cardinality *categorical* to manage before encoding (nested within `main_category`'s 15 levels)

## Potential Data Leakage
These are known only **after** a campaign ends, so they leak the outcome into a
pre-launch predictor:
- `pledged` / `usd pledged` / `usd_pledged_real` — final amount raised
- `backers` — final backer count
- Any engineered feature derived from them (`pledge_ratio`, `is_overfunded`, `log_goal_pledge_gap`, `pledged_log`)

Keep these for descriptive analysis; **exclude from a forward-looking model.**

## Integrity Issues
- **Epoch sentinel in `launched`:** minimum `launched` is `1970-01-01 01:00:00` — an
  epoch-zero placeholder, not a real launch. Treat as missing.
- **`usd pledged` vs `usd_pledged_real`:** the space-named legacy column has 3,797
  nulls; `usd_pledged_real` is the complete Fixer.io conversion — prefer it.
- **Currency not comparable:** `goal` / `pledged` are in native currency (14 currencies);
  cross-campaign math must use the `usd_*_real` columns.
- **Extreme skew / outliers:** `goal` ranges 0.01 → 100,000,000; `backers` → 219,382;
  skewness 70–87 on monetary/count features. Log-transform + outlier handling needed.
- **Ambiguous target classes:** `state` has 6 values; `canceled`, `undefined`, `live`,
  `suspended` (~12%) are not clean outcomes.

## Recommended Actions
- **Convert date columns** — parse `launched` and `deadline` to datetime; null out the 1970 sentinel.
- **Handle missing values** — drop/prefer `usd_pledged_real` over `usd pledged`; fill 4 missing `name`s with empty string.
- **Remove duplicates** — none present; keep a guard in the pipeline anyway.
- **Drop constant columns** — none to drop currently.
- **Investigate leakage features** — quarantine `pledged`, `backers`, `usd_*pledged*` from any pre-launch model.
- **Reduce cardinality** — group rare levels in `category`, `country`, `currency` before encoding.
- **Frame target** — filter to `{successful, failed}` for a binary task (~87.6% of rows, ~40/60 split).
