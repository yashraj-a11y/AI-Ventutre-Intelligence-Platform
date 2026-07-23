# Exploratory Data Analysis Report

**Project:** AI Venture Intelligence Platform
**Dataset:** Kickstarter Projects (`ks-projects-201801.csv`) · 2009–2018
**Objective:** Predict crowdfunding campaign success from information available *at launch*.
**Date:** 2026-07-23 · **Analyst:** Yash

> A single-page technical narrative of the dataset: what it contains, how clean it is,
> what it reveals about venture success, and how it will be prepared for modeling.
> All figures are computed from the raw data in `notebooks/01_EDA.ipynb`.

---

## 1. Dataset Overview

| Property | Value |
| -------- | ----- |
| Rows | **378,661** campaigns |
| Columns | **15** |
| Time span | 2009 → 2018 |
| Memory | 195.7 MB (deep) |
| Grain | One row = one campaign |
| Primary key | `ID` (100% unique) |

**Feature summary (by role):**

| Role | Columns |
| ---- | ------- |
| Identifier | `ID` |
| Text | `name` (campaign title) |
| Temporal | `launched`, `deadline` |
| Numeric | `goal`, `pledged`, `backers`, `usd pledged`, `usd_pledged_real`, `usd_goal_real` |
| Categorical | `category` (159), `main_category` (15), `currency` (14), `country` (23) |
| Target | `state` |

**Target definition.** `state` has 6 classes; we frame a **binary classifier** on the
decided outcomes — `successful` (1) vs `failed` (0) — which cover **87.6%** of rows
(331,675 campaigns) with a **40.4% success rate** (~40/60 split). The ambiguous states
(`canceled` 10.2%, `undefined`/`live`/`suspended` ~2.2%) are excluded.

---

## 2. Data Quality

**Overall: structurally clean.** No duplicate rows, a valid key, no constant columns.

| Dimension | Finding | Severity |
| --------- | ------- | -------- |
| **Missing values** | `usd pledged` 3,797 (1.0%), `name` 4. The nulls are **MNAR** — they map exactly to a `country='N,0"'` corrupt cohort (3,562 `undefined`). | 🟠 |
| **Duplicates** | 0 duplicate rows, 0 duplicate IDs. (Campaign *titles* repeat — 2,228 names reused — but rows are distinct.) | 🟢 |
| **Datatypes** | `launched`/`deadline` stored as **strings** (need datetime); `state` needs encoding; string categoricals should be `category` dtype (reclaims ~49% memory). | 🟠 |
| **Outliers** | 12–13% of rows flagged per numeric column (IQR). Maxima are 2,700–4,600× the upper fence — genuine viral mega-campaigns, not errors. | 🔴 |

Full audit in [data_quality_report.md](data_quality_report.md).

---

## 3. Statistical Summary

**Key statistics** (native units):

| Feature | Median | Mean | Max | Skew |
| ------- | ------ | ---- | --- | ---- |
| `goal` | 5,200 | 49,081 | 100,000,000 | 70.8 |
| `pledged` | 620 | 9,683 | 20,338,986 | 75.2 |
| `backers` | 12 | 106 | 219,382 | 86.8 |
| `usd_goal_real` | 5,500 | 45,454 | 166,361,391 | 78.2 |

**Distribution observations.** Every monetary/count feature is **severely right-skewed**
(skew 70–106, kurtosis up to ~19,000). Mean ≫ median everywhere — a thin tail of
mega-campaigns dominates. The *typical* campaign targets **$5,200**, raises **$620** from
**12 backers**, i.e. ~12% of its goal. Log transforms are mandatory.

**Correlations.**

- *Feature ↔ feature (multicollinearity):* the USD conversions are near-duplicates —
  `pledged ↔ usd_pledged_real` **0.95**, `goal ↔ usd_goal_real` **0.94**,
  `backers ↔ pledged` **0.72**. Keep one column per group.
- *Feature ↔ target:* linear correlations are **weak** — `backers` +0.13, `pledged`
  +0.11, `goal` −0.03. The (monotonic) relationships are masked by extreme skew;
  Pearson understates them. Full detail in [statistical_summary.md](statistical_summary.md).

---

## 4. Venture Intelligence Insights

**Startup / category trends.** Success rate varies **3×** across categories — a strong
structural signal:

| Best categories | Rate | Worst categories | Rate |
| --------------- | ---- | ---------------- | ---- |
| Dance | 65.4% | Technology | 23.8% |
| Theater | 63.8% | Journalism | 24.4% |
| Comics | 59.1% | Crafts | 27.1% |

Creative/performance ventures over-perform; capital-intensive/product ventures under-perform.

**Funding observations.** Lower ambition correlates with success — successful campaigns
have a **median goal of $3,837** vs **$7,500** for failed. Volume peaked in **2015
(77,300 launches)** then declined through 2017.

**Success indicators (pre-launch):**
- Modest funding goal (log-goal is one of the few honest predictors).
- High-performing category (Dance, Theater, Comics, Music).
- **Short campaigns:** 0–15 day windows succeed **51%** of the time vs **27%** for 45–60 days.
- Launch timing: **Tuesday** (42.5%) and **spring months** (March 42.7%) edge out
  Saturday (37.5%) and December (37.1%).
- Slightly longer, richer titles (avg 36 vs 33 chars; titles with a number succeed 50% vs 39%).

**Failure indicators:**
- Large funding goals; capital-intensive categories (Technology, Food, Fashion).
- Long campaigns (**90+ days → 23%** success) — over-long windows signal weak planning.
- Weekend/December launches; very short, low-information titles.

Category deep-dive in [business_insights.md](business_insights.md).

---

## 5. ML Challenges

| Challenge | Detail | Mitigation |
| --------- | ------ | ---------- |
| **Class imbalance** | 40/60 success/fail (mild). Full `state` is worse (52% failed). | Stratified split; F1/PR-AUC over accuracy. |
| **Missing data** | 1.0% but **MNAR** (corrupt `N,0"` cohort). | Drop cohort; standardize on `usd_pledged_real`. |
| **High-cardinality** | `category` (159 levels) + 8 categories map to multiple parents; `name`/`ID` near-unique. | Encode on `main_category`; frequency/target-encode `category`. |
| **Text quality** | Only `name`; 23,145 titles contain "cancel" (**outcome leakage**); 9,612 non-ASCII. | Strip outcome markers; normalize; derive length/word/signal features. |
| **Feature leakage** 🔴 | `pledged`, `backers`, `usd pledged`, `usd_pledged_real` are **post-event** — known only after a campaign closes. | **Reject** them and all derivatives from any pre-launch model. |

The leakage risk is the single most important modeling constraint: only `goal`,
`usd_goal_real`, category, country/currency, timing, and cleaned title features are
legitimately available at launch.

---

## 6. Recommended Preprocessing

**Cleaning plan**
- Parse `launched`/`deadline` → datetime; null the 7 `1970` epoch sentinels.
- Filter to decided outcomes; drop the `N,0"` / `undefined` cohort.
- Drop 102 non-positive-duration rows; null `goal < $1` (4 rows).
- Standardize on `usd_goal_real`/`usd_pledged_real`; drop the legacy `usd pledged`.

**Encoding plan**
- `main_category` → one-hot (15 levels).
- `category` → frequency or target encoding (159 levels; avoid one-hot).
- `country`, `currency` → group rare levels, then encode.
- `state` → binary target.

**Scaling plan**
- `log1p` transform on `goal`/`usd_goal_real`; IQR-clip extreme outliers.
- **Robust scaling** (median/IQR), not standard, given heavy tails.

**NLP preprocessing plan (`name`)**
- Fill 4 blanks; Unicode-normalize; lowercase; strip "(Canceled)"/outcome markers.
- Derive `name_length`, `word_count`, `uppercase_ratio`, `has_number/!/?`.
- (Future) TF-IDF / embeddings for a text sub-model.

**Feature-engineering roadmap** (via `FeatureEngineeringHandler`)
- *Temporal:* `campaign_duration`, `launch_month/weekday/quarter`, `is_weekend_launch`.
- *Numeric:* `goal_log`, `is_high_goal`.
- *Interaction:* `goal_per_day`, `goal × duration`, `goal × month`.
- *Text:* the title features above.
- ⚠️ Funding features (`pledge_ratio`, `is_overfunded`) are **leakage** — descriptive use only.

---

## Conclusion

The Kickstarter dataset is **large, clean in structure, and rich in signal**, but its
predictive value hinges on **discipline around leakage**: the obvious "strong" features
are post-event outcomes. With those removed, log-transformed inputs, a decided-only
target, and category/timing/goal features, the data is ready for baseline and
gradient-boosted classifiers. **Expected honest performance ceiling is modest** —
pre-launch success is genuinely hard to predict — making category, goal size, campaign
duration, and timing the features to watch.

---

*Companion docs: [data_dictionary.md](data_dictionary.md) ·
[eda_observations.md](eda_observations.md) · [statistical_summary.md](statistical_summary.md) ·
[data_quality_report.md](data_quality_report.md) · [business_insights.md](business_insights.md) ·
[project_architecture.md](project_architecture.md).*
