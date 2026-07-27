# Day 4 — Phase D: Model Benchmark

_Validation set (53,035 rows). Ranked by PR-AUC. Baseline: LogReg L2._

| model | acc | precision | recall | F1 | ROC-AUC | PR-AUC | train time |
|---|--:|--:|--:|--:|--:|--:|--:|
| hist_gradient_boosting | 0.7067 | 0.6607 | 0.5627 | 0.6077 | 0.7703 | **0.6805** | 30.2s |
| lightgbm | 0.7066 | 0.6608 | 0.5614 | 0.6071 | 0.7697 | **0.6798** | 2.9s |
| xgboost | 0.7046 | 0.6599 | 0.5538 | 0.6022 | 0.7675 | **0.6754** | 3.8s |
| random_forest | 0.7015 | 0.6676 | 0.5191 | 0.5841 | 0.7592 | **0.6643** | 115.4s |
| extra_trees | 0.6982 | 0.6617 | 0.5168 | 0.5803 | 0.7535 | **0.6589** | 186.9s |
| logreg_l2 (baseline) | 0.6768 | 0.6291 | 0.4864 | 0.5486 | 0.7269 | **0.6264** | 30.2s |

**Best by PR-AUC: `hist_gradient_boosting` (PR-AUC 0.6805).**
