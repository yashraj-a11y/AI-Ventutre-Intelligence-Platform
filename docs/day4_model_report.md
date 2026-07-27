# Day 4 — Model Validation Report

_Final model scored **once** on the untouched test set (66,293 rows) at threshold 0.329._

## Best model

Tuned **LightGBM** + sigmoid calibration (selected by validation PR-AUC over LogReg baseline, Random Forest, ExtraTrees, XGBoost, HistGradientBoosting).

## Test metrics (honest generalization)

| metric | value |
|---|--:|
| accuracy | 0.6713 |
| precision | 0.5646 |
| recall | 0.8131 |
| f1 | 0.6664 |
| roc_auc | 0.7763 |
| pr_auc | 0.6883 |
| brier | 0.1873 |

_No-Information Rate (majority baseline) accuracy = 0.5962; PR-AUC base rate = 0.4038._

## Generalization check (validation vs test PR-AUC)

- validation PR-AUC: **0.6884**  ·  test PR-AUC: **0.6883**  → gap 0.0001 (small ⇒ selection did not overfit).

## Confusion matrix (thr = 0.329)

| | pred failed | pred successful |
|---|--:|--:|
| **actual failed** | 22,735 (TN) | 16,788 (FP) |
| **actual successful** | 5,004 (FN) | 21,766 (TP) |

## Plots

- `artifacts/evaluations/confusion_matrix.png`
- `artifacts/evaluations/roc_curve.png`
- `artifacts/evaluations/precision_recall_curve.png`
- `artifacts/evaluations/calibration_curve_test.png`

## Error analysis

- At threshold 0.329, recall 0.8131 / precision 0.5646: the operating point favours catching successes (low FN) at the cost of more FP.
- 5,004 successful campaigns missed (FN); 16,788 failures wrongly flagged successful (FP).
- Signal ceiling is set by weak pre-launch features (no founder/market data); richer data (Layers 2–3) is the path to higher PR-AUC.
