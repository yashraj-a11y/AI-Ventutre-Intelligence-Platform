# Day 5 · Phase B — Global Feature Importance

> Model: **LGBMClassifier** (base tuned LightGBM, uncalibrated, SHAP-able).
> 865 features = **800 title TF-IDF tokens + 65 tabular**.
> mean\|SHAP\| over **2000** densified rows via **interventional** TreeSHAP
> (the default fast path is numerically broken for this model).
> Additivity guard: max\|expected+Σshap−margin\| = **7.39e-08** ✅.

## What this answers
The **global** question from Phase A: which real features drive funding success across
the whole model? Three independent rankings — **gain** (loss reduction, the honest tree
importance), **split** (usage count), and **mean\|SHAP\|** (average absolute
contribution, model-agnostic + additive) — cross-checked against each other. Primary
ordering below is by mean\|SHAP\|, gain as tiebreak.

## Top 50 features (by mean\|SHAP\|)
| rank | feature | group | gain_share | split_share | mean\|SHAP\| |
|---:|---|---|---:|---:|---:|
| 1 | `usd_goal_real` | tabular | 0.1473 | 0.0505 | 0.3203 |
| 2 | `campaign_duration` | tabular | 0.1067 | 0.0494 | 0.2626 |
| 3 | `category_freq` | tabular | 0.1254 | 0.0844 | 0.2216 |
| 4 | `launch_year` | tabular | 0.0523 | 0.0393 | 0.2014 |
| 5 | `goal_per_day` | tabular | 0.0751 | 0.0555 | 0.1499 |
| 6 | `word_count` | tabular | 0.0416 | 0.0159 | 0.1341 |
| 7 | `title_length` | tabular | 0.0257 | 0.0365 | 0.1114 |
| 8 | `uppercase_ratio` | tabular | 0.0400 | 0.0598 | 0.1055 |
| 9 | `launch_weekday` | tabular | 0.0112 | 0.0165 | 0.0638 |
| 10 | `launch_month` | tabular | 0.0135 | 0.0286 | 0.0588 |
| 11 | `has_number` | tabular | 0.0078 | 0.0049 | 0.0428 |
| 12 | `main_category_Music` | tabular | 0.0171 | 0.0053 | 0.0404 |
| 13 | `main_category_Publishing` | tabular | 0.0082 | 0.0061 | 0.0366 |
| 14 | `currency_EUR` | tabular | 0.0025 | 0.0019 | 0.0316 |
| 15 | `currency_USD` | tabular | 0.0038 | 0.0048 | 0.0315 |
| 16 | `avg_word_length` | tabular | 0.0136 | 0.0382 | 0.0286 |
| 17 | `main_category_Crafts` | tabular | 0.0041 | 0.0028 | 0.0245 |
| 18 | `main_category_Technology` | tabular | 0.0110 | 0.0063 | 0.0239 |
| 19 | `main_category_Comics` | tabular | 0.0126 | 0.0047 | 0.0225 |
| 20 | `main_category_Fashion` | tabular | 0.0078 | 0.0042 | 0.0215 |
| 21 | `main_category_Theater` | tabular | 0.0179 | 0.0056 | 0.0210 |
| 22 | `main_category_Film & Video` | tabular | 0.0066 | 0.0057 | 0.0195 |
| 23 | `main_category_Games` | tabular | 0.0109 | 0.0067 | 0.0193 |
| 24 | `main_category_Food` | tabular | 0.0033 | 0.0038 | 0.0174 |
| 25 | `thesis` | text | 0.0013 | 0.0012 | 0.0139 |
| 26 | `album` | text | 0.0059 | 0.0046 | 0.0137 |
| 27 | `game` | text | 0.0023 | 0.0033 | 0.0126 |
| 28 | `film` | text | 0.0055 | 0.0052 | 0.0125 |
| 29 | `main_category_Art` | tabular | 0.0033 | 0.0028 | 0.0107 |
| 30 | `miniature` | text | 0.0011 | 0.0012 | 0.0105 |
| 31 | `clothing` | text | 0.0018 | 0.0019 | 0.0102 |
| 32 | `exhibition` | text | 0.0011 | 0.0017 | 0.0101 |
| 33 | `custom` | text | 0.0009 | 0.0016 | 0.0100 |
| 34 | `funding` | text | 0.0006 | 0.0011 | 0.0088 |
| 35 | `pet` | text | 0.0004 | 0.0010 | 0.0085 |
| 36 | `playing card` | text | 0.0019 | 0.0016 | 0.0082 |
| 37 | `event` | text | 0.0005 | 0.0009 | 0.0077 |
| 38 | `music video` | text | 0.0006 | 0.0013 | 0.0076 |
| 39 | `watch` | text | 0.0022 | 0.0024 | 0.0076 |
| 40 | `war` | text | 0.0008 | 0.0015 | 0.0072 |
| 41 | `app` | text | 0.0030 | 0.0018 | 0.0072 |
| 42 | `country_US` | tabular | 0.0008 | 0.0009 | 0.0070 |
| 43 | `new` | text | 0.0037 | 0.0043 | 0.0069 |
| 44 | `man` | text | 0.0006 | 0.0013 | 0.0068 |
| 45 | `series` | text | 0.0010 | 0.0024 | 0.0067 |
| 46 | `service` | text | 0.0004 | 0.0009 | 0.0066 |
| 47 | `book` | text | 0.0026 | 0.0041 | 0.0064 |
| 48 | `currency_GBP` | tabular | 0.0020 | 0.0025 | 0.0063 |
| 49 | `help` | text | 0.0011 | 0.0026 | 0.0063 |
| 50 | `main_category_Dance` | tabular | 0.0072 | 0.0040 | 0.0063 |

## Top 15 tabular features
| rank | feature | group | gain_share | split_share | mean\|SHAP\| |
|---:|---|---|---:|---:|---:|
| 1 | `usd_goal_real` | tabular | 0.1473 | 0.0505 | 0.3203 |
| 2 | `campaign_duration` | tabular | 0.1067 | 0.0494 | 0.2626 |
| 3 | `category_freq` | tabular | 0.1254 | 0.0844 | 0.2216 |
| 4 | `launch_year` | tabular | 0.0523 | 0.0393 | 0.2014 |
| 5 | `goal_per_day` | tabular | 0.0751 | 0.0555 | 0.1499 |
| 6 | `word_count` | tabular | 0.0416 | 0.0159 | 0.1341 |
| 7 | `title_length` | tabular | 0.0257 | 0.0365 | 0.1114 |
| 8 | `uppercase_ratio` | tabular | 0.0400 | 0.0598 | 0.1055 |
| 9 | `launch_weekday` | tabular | 0.0112 | 0.0165 | 0.0638 |
| 10 | `launch_month` | tabular | 0.0135 | 0.0286 | 0.0588 |
| 11 | `has_number` | tabular | 0.0078 | 0.0049 | 0.0428 |
| 12 | `main_category_Music` | tabular | 0.0171 | 0.0053 | 0.0404 |
| 13 | `main_category_Publishing` | tabular | 0.0082 | 0.0061 | 0.0366 |
| 14 | `currency_EUR` | tabular | 0.0025 | 0.0019 | 0.0316 |
| 15 | `currency_USD` | tabular | 0.0038 | 0.0048 | 0.0315 |

## Top 15 title TF-IDF tokens
| rank | feature | group | gain_share | split_share | mean\|SHAP\| |
|---:|---|---|---:|---:|---:|
| 1 | `thesis` | text | 0.0013 | 0.0012 | 0.0139 |
| 2 | `album` | text | 0.0059 | 0.0046 | 0.0137 |
| 3 | `game` | text | 0.0023 | 0.0033 | 0.0126 |
| 4 | `film` | text | 0.0055 | 0.0052 | 0.0125 |
| 5 | `miniature` | text | 0.0011 | 0.0012 | 0.0105 |
| 6 | `clothing` | text | 0.0018 | 0.0019 | 0.0102 |
| 7 | `exhibition` | text | 0.0011 | 0.0017 | 0.0101 |
| 8 | `custom` | text | 0.0009 | 0.0016 | 0.0100 |
| 9 | `funding` | text | 0.0006 | 0.0011 | 0.0088 |
| 10 | `pet` | text | 0.0004 | 0.0010 | 0.0085 |
| 11 | `playing card` | text | 0.0019 | 0.0016 | 0.0082 |
| 12 | `event` | text | 0.0005 | 0.0009 | 0.0077 |
| 13 | `music video` | text | 0.0006 | 0.0013 | 0.0076 |
| 14 | `watch` | text | 0.0022 | 0.0024 | 0.0076 |
| 15 | `war` | text | 0.0008 | 0.0015 | 0.0072 |

## Honesty caption
These are **associations the model learned from campaign metadata and title text**, not
causal drivers. A high mean\|SHAP\| means "this feature moved the model's funding
probability a lot on average," not "changing it will cause funding." Tabular features
(goal size, category commonness, timing) and specific title words are all the model saw —
there are no founder/team/market/revenue signals in this dataset.
