# Day 3 — Phase A: Feature Profile (train-only)

_Generated 2026-07-25T14:10:55.318014+00:00 · train rows: 265,171_

## Numerics

| feature | min | median | max | skew | kurtosis | log skew |
|---|--:|--:|--:|--:|--:|--:|
| `goal` | 0 | 5000 | 100000000 | 73.12 | 5918.2 | -0.12 |
| `usd_goal_real` | 0 | 5000 | 166361391 | 80.89 | 7589.1 | -0.14 |

## Categoricals

| feature | cardinality | unseen in test | top levels |
|---|--:|--:|---|
| `category` | 159 | 0 | Product Design (14,916), Documentary (11,619), Music (9,994) |
| `main_category` | 15 | 0 | Film & Video (45,162), Music (36,633), Publishing (28,333) |
| `currency` | 14 | 0 | USD (208,906), GBP (23,652), EUR (11,554) |
| `country` | 22 | 0 | US (208,906), GB (23,652), CA (9,873) |

## Numeric correlations

- `goal` ↔ `usd_goal_real`: pearson **0.957**, spearman **0.981**

## Title text

- `name_clean` word count: mean **5.63**, median 5, p95 10, max 27 · empty: 4

## Dates → campaign_duration

- duration (days): min **0**, median 29, mean 32.9, p95 59, max 91, skew 1.25
- rows with duration ≤ 0 (divide-by-zero risk): **74**

## Candidate transformations

| feature | transform | reason |
|---|---|---|
| `goal` | **log1p** | heavy right skew (73.1) collapses to -0.12 under log1p |
| `usd_goal_real` | **log1p** | heavy right skew (80.9) collapses to -0.14 under log1p |
| `category` | **frequency_encode** | high cardinality (159 > 25); one-hot would explode width |
| `main_category` | **one_hot** | low cardinality (15 <= 25) |
| `currency` | **one_hot** | low cardinality (14 <= 25) |
| `country` | **one_hot** | low cardinality (22 <= 25) |
| `goal|usd_goal_real` | **drop_one_of_pair** | near-collinear (pearson 0.957); keep one |
| `name_clean` | **tfidf_small_vocab** | short titles (~5.6 words); cap TF-IDF features low (~800) with (1,2)-grams, not 2500 |
| `campaign_duration` | **engineer_from_dates** | derive (deadline - launched).days + temporal parts (year/month/weekday) |
| `goal_per_day` | **guard_divide_by_zero** | 74 row(s) have duration <= 0; goal/duration would divide by zero — clip denominator to >= 1 |
