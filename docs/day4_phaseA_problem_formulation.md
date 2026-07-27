# Day 4 — Phase A: Prediction Problem Formulation

## The task in one line
**Given the pre-launch features of a Kickstarter campaign, predict the probability that it will reach its funding goal.**

## Precise specification

| Dimension | Decision |
|---|---|
| **Task type** | Binary classification (probabilistic) |
| **Unit of prediction** | One campaign |
| **Target `y`** | `1` = `successful` (met goal), `0` = `failed` (did not) |
| **Features `X`** | 865 pre-launch features (Day-3 `X_selected`): goal, duration, timing, category, title text |
| **Output** | Calibrated probability `P(success | X) ∈ [0,1]`, plus a thresholded label |
| **Primary metrics** | PR-AUC and F1 (imbalance-aware); ROC-AUC secondary; accuracy only vs baseline |

## Population & scope (the subtle part)
The model is trained and applied **only on "decided" campaigns** — those that reached a natural conclusion (`successful` or `failed`).

| State | Count | In scope? |
|---|--:|---|
| failed | 197,719 | ✅ modeled (`y=0`) |
| successful | 133,956 | ✅ modeled (`y=1`) |
| canceled | 38,779 | ❌ excluded |
| undefined | 3,562 | ❌ excluded |
| live | 2,799 | ❌ excluded |
| suspended | 1,846 | ❌ excluded |

**Modeled population: 331,464 campaigns (87.6% of all).** The 46,986 excluded campaigns (canceled/live/suspended/undefined) never reached a clean success/fail outcome, so they carry no usable label.

## Base rate (the number every model must beat)
- **Success rate: 40.38%** (133,850 of 331,464).
- **Majority-class baseline: 59.62% accuracy** — achieved by *always* predicting "failed". Any model reporting < ~60% accuracy is worse than a constant guess; accuracy alone is therefore misleading, which is why PR-AUC/F1 are primary.

## Assumptions & limitations (stated up front)
1. **Selection scope:** predictions are valid for campaigns that will reach a decision. A brand-new campaign *might* later be canceled — a case the model was not trained on. The output is best read as *P(success | the campaign runs to completion)*.
2. **Pre-launch only:** every feature is knowable at launch (Day-3 Phase F). No outcome-derived signal (pledged/backers) — so the prediction is honestly *ex ante*.
3. **Single label:** only funding success has ground truth. Growth/risk/readiness/founder scores are **not learnable here** and are handled as documented heuristics (Phase G) or deferred.
4. **Temporal drift:** the data is a 2018 snapshot; a deployed model would need monitoring for concept drift.

## Deliverable
This document. Downstream phases (B–J) train, tune, calibrate, and evaluate against this frozen formulation.
