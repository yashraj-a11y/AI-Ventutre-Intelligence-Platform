# Statistical Summary

Descriptive statistics for the **Kickstarter Projects** dataset (378,661 rows).
Numbers are computed on the 7 numeric columns; `ID` is included only where noted
(it's a random identifier, not a feature). Source: `notebooks/01_EDA.ipynb` and
`outputs/reports/numerical_summary.csv`.

Reference table (native units; USD columns already converted):

| Feature | Mean | Median | Std Dev | Skewness | Kurtosis | Min | Max |
| ------- | ---- | ------ | ------- | -------- | -------- | --- | --- |
| goal | 49,080.79 | 5,200 | 1,183,391 | 70.80 | 5,574.95 | 0.01 | 100,000,000 |
| pledged | 9,682.98 | 620 | 95,636 | 75.15 | 10,010.26 | 0 | 20,338,986 |
| backers | 105.62 | 12 | 907 | 86.76 | 13,954.94 | 0 | 219,382 |
| usd pledged | 7,036.73 | 394.72 | 78,640 | 105.90 | 18,960.92 | 0 | 20,338,986 |
| usd_pledged_real | 9,058.92 | 624.33 | 90,973 | 82.19 | 11,796.48 | 0 | 20,338,986 |
| usd_goal_real | 45,454.40 | 5,500 | 1,152,950 | 78.22 | 7,082.89 | 0.01 | 166,361,391 |

## Central Tendency

- **Mean observations:** every monetary/count feature has a **mean far above its
  median** — e.g. `goal` mean 49,081 vs median 5,200 (≈9×), `pledged` mean 9,683 vs
  median 620 (≈15×). The means are pulled up by a thin tail of mega-campaigns and are
  not representative of a typical project.
- **Median observations:** medians describe the "typical" campaign much better — a
  median project targets **$5,200**, raises **$620**, from **12 backers**. Median goal
  (5,200) far exceeds median pledged (620): the typical campaign raises ~12% of its goal,
  consistent with the majority-fail base rate.
- **Mode observations (categoricals):** most common values are `main_category` =
  **Film & Video**, `category` = **Product Design**, `currency` = **USD**,
  `country` = **US**, and target `state` = **failed** (the modal outcome).

## Dispersion

- **Variance:** enormous and dominated by outliers — `goal` ≈ 1.40 × 10¹², `usd_goal_real`
  ≈ 1.33 × 10¹², `pledged` ≈ 9.15 × 10⁹. Variance is nearly meaningless on the raw scale.
- **Standard deviation:** exceeds the mean on every feature (e.g. `goal` std 1,183,391 ≫
  mean 49,081; `backers` std 907 ≫ mean 106). Coefficient of variation > 1 everywhere →
  extreme spread. **Log-transform before any variance-based method** (scaling, PCA, linear models).

## Distribution Shape

- **Skewness:** severe right skew across the board — `goal` 70.8, `pledged` 75.2,
  `usd_goal_real` 78.2, `usd_pledged_real` 82.2, `backers` 86.8, `usd pledged` 105.9.
  (For reference, |skew| > 1 is already "highly skewed"; these are 70–106.) `ID` is
  ≈ 0, confirming it's a uniform identifier.
- **Kurtosis:** extreme heavy tails — from 5,575 (`goal`) up to 18,961 (`usd pledged`).
  A normal distribution has kurtosis 0; these signal a large number of extreme outliers.
- **Implication:** `log1p` transforms are essential; the engineered `goal_log` /
  `pledged_log` features exist precisely for this.

## Quantiles

| Feature | Min | Q1 (25%) | Median | Q3 (75%) | Max | IQR |
| ------- | --- | -------- | ------ | -------- | --- | --- |
| goal | 0.01 | 2,000 | 5,200 | 16,000 | 100,000,000 | 14,000 |
| pledged | 0 | 30 | 620 | 4,076 | 20,338,986 | 4,046 |
| backers | 0 | 2 | 12 | 56 | 219,382 | 54 |
| usd pledged | 0 | 16.98 | 394.72 | 3,034.09 | 20,338,986 | 3,017.11 |
| usd_pledged_real | 0 | 31 | 624.33 | 4,050 | 20,338,986 | 4,019 |
| usd_goal_real | 0.01 | 2,000 | 5,500 | 15,500 | 166,361,391 | 13,500 |

- **Q1 / Median / Q3:** the interquartile ranges are tight relative to the maxima — 50%
  of campaigns target between $2,000 and $16,000, yet the max goal is $100M.
- **Extreme percentiles:** the `max` values are 1,000–7,000× the Q3 in every column
  (e.g. `backers` Q3 = 56, max = 219,382). These are genuine outliers, not errors —
  a handful of viral campaigns. Min goals of `$0.01` are also suspicious and worth a
  data-integrity check.

## Correlation Analysis

Computed on the **decided subset** (successful/failed only, 331,675 rows; 40.4% success)
with `is_successful` = 1 for successful.

**Correlation of numeric features with `is_successful`:**

| Feature | r |
| ------- | --- |
| backers | +0.126 |
| pledged | +0.110 |
| usd_pledged_real | +0.108 |
| usd pledged | +0.096 |
| usd_goal_real | −0.024 |
| goal | −0.025 |

- **Strong positive relationships (features ↔ features):** the USD conversions of the
  same underlying quantity are near-duplicates —
  `pledged ↔ usd_pledged_real` **0.953**, `usd pledged ↔ usd_pledged_real` **0.908**,
  `pledged ↔ usd pledged` **0.857**; and `goal ↔ usd_goal_real` **0.943**.
  `backers ↔ pledged` is also strong (**0.72**) and `backers ↔ usd_pledged_real` (**0.75**).
- **Strong negative relationships:** none. Correlations with the target are only weakly
  negative (`goal`/`usd_goal_real` ≈ −0.025) — a larger goal is *slightly* associated
  with failure, as expected, but the raw linear signal is faint.
- **Potential multicollinearity:** severe among the money columns. `pledged`,
  `usd pledged`, `usd_pledged_real` are effectively one variable (r > 0.85); `goal` and
  `usd_goal_real` are another (r = 0.94). **Keep one USD column from each group** and drop
  the rest before any linear/regularized model.
- **Note on weak target correlation:** even `pledged`/`backers` show only r ≈ 0.11–0.13
  despite being definitionally tied to success. This is an artifact of extreme skew —
  Pearson underestimates the (monotonic) relationship. Re-check on `log1p`-transformed
  features or with a rank/mutual-information measure.

## Potential Data Leakage

- `pledged`, `backers`, `usd pledged`, `usd_pledged_real` are **outcome variables** —
  known only *after* a campaign closes. Their (even weak) positive correlation with
  `is_successful` is leakage, not predictive signal.
- Any engineered feature derived from them (`pledge_ratio`, `is_overfunded`,
  `pledged_log`, `log_goal_pledge_gap`) inherits the leakage.
- **Legitimate pre-launch numerics** are limited to `goal` / `usd_goal_real` (plus
  temporal, text, and categorical features engineered elsewhere).

## Engineering Recommendations

- **Log-transform** all monetary/count features (`log1p`) before scaling or modeling —
  addresses skew (70–106) and heavy tails (kurtosis up to ~19,000).
- **Handle outliers** — clip or winsorize using the IQR bounds above; the maxima are
  1,000×+ the Q3.
- **Drop redundant columns** — keep one of {`pledged`, `usd pledged`, `usd_pledged_real`}
  and one of {`goal`, `usd_goal_real`} to kill multicollinearity; prefer the `*_real`
  USD versions for cross-campaign comparability.
- **Quarantine leakage features** from any forward-looking model; retain them only for
  descriptive/post-hoc analysis.
- **Use robust scaling** (median/IQR) rather than standard (mean/std) given the outliers.
- **Investigate integrity edge cases** — `goal`/`usd_goal_real` minimums of `$0.01`.
- **Re-measure feature–target association** on transformed features (Spearman / mutual
  information) rather than raw Pearson.

---

*Correlations recomputed from `data/raw/ks-projects-201801.csv`; descriptive stats from
`outputs/reports/numerical_summary.csv`. See also [data_dictionary.md](data_dictionary.md)
and [eda_observations.md](eda_observations.md).*
