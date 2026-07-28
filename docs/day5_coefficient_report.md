# Day 5 · Phase C — Logistic Regression Coefficient Analysis (Week 7)

> Model: **LogisticRegression(L2)**, intercept β₀ = -0.0338.
> 865 coefficients = **800 title TF-IDF tokens + 65 tabular**.
> Features are StandardScaled ⇒ each coef = Δlog-odds per **+1 SD**; exp(coef) = odds multiplier.

## What this answers
The **white-box** view: the logistic model's coefficient *is* the explanation.
Positive → pushes funding probability up; negative → down. Ranked by |coef| because,
on standardized features, magnitude is directly comparable across features.

## Top 25 features that INCREASE funding probability (+coef)
| feature | group | coef | odds_ratio (per +1 SD) |
|---|---|---:|---:|
| `category_freq` | tabular | +5.956 | 386.195 |
| `new record` | text | +2.117 | 8.304 |
| `miniature` | text | +2.050 | 7.768 |
| `burning man` | text | +1.956 | 7.074 |
| `thesis` | text | +1.909 | 6.744 |
| `titanium` | text | +1.788 | 5.975 |
| `art book` | text | +1.770 | 5.868 |
| `post production` | text | +1.729 | 5.634 |
| `residency` | text | +1.727 | 5.625 |
| `brewing` | text | +1.692 | 5.432 |
| `anthology` | text | +1.644 | 5.173 |
| `relaunch` | text | +1.604 | 4.973 |
| `world first` | text | +1.393 | 4.027 |
| `pen` | text | +1.331 | 3.784 |
| `wireless` | text | +1.321 | 3.749 |
| `watch` | text | +1.300 | 3.669 |
| `playing card` | text | +1.298 | 3.661 |
| `new album` | text | +1.262 | 3.533 |
| `camera` | text | +1.220 | 3.386 |
| `modular` | text | +1.209 | 3.351 |
| `backpack` | text | +1.197 | 3.310 |
| `dice` | text | +1.148 | 3.151 |
| `edition` | text | +1.147 | 3.150 |
| `music video` | text | +1.127 | 3.086 |
| `studio album` | text | +1.110 | 3.035 |

## Top 25 features that DECREASE funding probability (−coef)
| feature | group | coef | odds_ratio (per +1 SD) |
|---|---|---:|---:|
| `youtube` | text | -2.902 | 0.055 |
| `hip hop` | text | -1.965 | 0.140 |
| `com` | text | -1.636 | 0.195 |
| `website` | text | -1.598 | 0.202 |
| `app` | text | -1.574 | 0.207 |
| `network` | text | -1.559 | 0.210 |
| `sport` | text | -1.446 | 0.236 |
| `entertainment` | text | -1.412 | 0.244 |
| `christian` | text | -1.282 | 0.278 |
| `io` | text | -1.229 | 0.293 |
| `freedom` | text | -1.205 | 0.300 |
| `clothing` | text | -1.195 | 0.303 |
| `event` | text | -1.190 | 0.304 |
| `veteran` | text | -1.165 | 0.312 |
| `funding` | text | -1.153 | 0.316 |
| `golf` | text | -1.125 | 0.325 |
| `book series` | text | -1.119 | 0.327 |
| `start` | text | -1.118 | 0.327 |
| `service` | text | -1.105 | 0.331 |
| `beginning` | text | -1.105 | 0.331 |
| `truth` | text | -1.097 | 0.334 |
| `texas` | text | -1.069 | 0.343 |
| `money` | text | -1.057 | 0.348 |
| `product` | text | -1.036 | 0.355 |
| `online` | text | -1.030 | 0.357 |

## Cross-model agreement (vs Phase-B LightGBM SHAP)
An independent tree model and this linear model agreeing on which features matter is
stronger evidence than either alone:

- **Spearman(|coef|, mean\|SHAP\|) = 0.426** over all 865 features.
- **1 / 20** features appear in the top-20 of *both* rankings: `category_freq`.

## Honesty caption
Coefficients are **associations learned from campaign metadata and title text**, not
causal levers. A negative coefficient on `youtube`/`website`/`app` means campaigns whose
titles lean on external-link words *tended* to fail in the training data — not that
removing the word causes funding. Only real features appear here; the dataset has no
founder/team/market/revenue signals.
