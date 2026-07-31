# 🚀 Day 8 — Counterfactual Engine & Intelligent What-If Analysis (Roadmap V1.1 — FROZEN)

> **Status: FROZEN engineering contract** (2026-07-30). Corrections from the draft flagged 🔧;
> changelog + honesty tiers at the end. Implements Architecture **Layer 7 — Counterfactual
> Engine**. Turns predictions into *actionable* what-ifs by perturbing **real, controllable**
> campaign features through the actual Day-4 model and validating against Day-6/7 evidence.

## Goal
Answer "what should this campaign change to raise its funding probability?" — but only via
features a founder can actually control, simulated through the **real calibrated model**, and
backed by historical evidence. Not "add a technical co-founder" (no such data) — but "lower
your goal from $57k to ~$14k: the model estimates your probability rises 0.02 → 0.05, and
campaigns like yours with small goals funded ~1.8× more often."

---

## Data / model reality (Source of Truth) 🔧
The campaign has **12 columns**; a founder controls only a few of them. The counterfactual
engine perturbs **RAW inputs** and lets the Day-3 pipeline + Day-4 calibrated model recompute
everything (train/serve-consistent), so feature coupling is handled automatically.

**MUTABLE (actionable) features — the only things we recommend changing** 🔧:
- **`goal` / `usd_goal_real`** — the dominant, proven lever (Days 4–7 all agree). Changing
  `goal` propagates to `usd_goal_real`, `goal_per_day`, `goal_band`.
- **`campaign_duration`** (via `deadline`) — shorter campaigns fund better (Day-7: Theater
  ≤15d lift 1.80). Bounded to Kickstarter's [1, 60] days.
- **launch timing** (via `launched`) — month / quarter / weekday / weekend (Day-7: Tuesday >
  weekend, small effect).
- **title features** (via `name`) — word count, `has_number` (Day-7: Comics + number).

**IMMUTABLE context — NEVER recommend changing (would be gaming, not advice)** 🔧:
`country`, `currency` (you can't relocate to game a model), and **`category` is "soft"** —
surfaced only as an optional *repositioning* note, never a trick.

**FABRICATED — the draft's adjustable features DO NOT EXIST → DEFER, never simulate** 🔧
(same discipline as Days 4–7): Monthly Revenue / MRR, revenue growth, **customer retention**,
burn rate, team size, **founder / CTO experience**, market size / TAM, product traction,
business model, CAC/LTV, funding stage. There is no founder/team/market/financial data. The
draft's headline example ("add an experienced technical co-founder → +11%") is pure invention.

### Non-obvious FROZEN decisions (easy to get wrong)
- **Perturb RAW campaign inputs, not engineered features** 🔧. Change `goal` / `deadline` /
  `launched` / `name` on the raw campaign, then run the Day-4 `PredictionEngine` (which wraps
  the Day-3 pipeline) — so `usd_goal_real`, `goal_per_day`, duration, timing, title-stats all
  recompute *consistently*. Never hand-edit a single engineered column (would desync coupled
  features). **Verified**: lowering a $57k goal to $14k moved calibrated p 0.023 → 0.048.
- **Simulate on the CALIBRATED model** (via `PredictionEngine`), so Δ probability is a
  meaningful calibrated change, not a raw margin.
- **Candidate values seeded by Day-7 patterns** 🔧: the target goal = the category's
  *funded-median* goal (Day-7 `funding_patterns.json`); target duration/timing = the
  best-lift Day-7 timing values. So recommendations are grounded, not arbitrary grid points.
- **Cumulative plans re-simulated, not summed** 🔧: the combined effect of several changes is
  measured by running the model on the *jointly-modified* campaign, never by adding individual
  Δs (features interact; summing over-counts). The 58→69→82→89 cascade is honest only if each
  step is a fresh model call on the cumulative campaign.
- **Bounded, feasible search** 🔧: goal ≥ a floor (e.g. $100) and within a sane multiple of the
  original (don't recommend $1); duration ∈ [1, 60]; only real weekday/month values.
- **Model-simulated ≠ causal guarantee** 🔧: a counterfactual shows how the *model's* estimate
  (trained on 265k historical campaigns) responds — a decision-support what-if, not a promise.
  Goal has a plausible *mechanism* (lower all-or-nothing target = easier to hit) → strongest
  tier; timing/title are weaker (the model may lean on confounds) → "suggestive" tier.
- **Repository/serialization = JSON, not pickle** 🔧 (`counterfactual_engine.pkl` →
  `counterfactual_engine.json`): configs/templates/reports are data (Day-6/7 lesson).
- **OpenMP** 🔧: the simulator needs `PredictionEngine` (lightgbm); if evidence uses Day-6
  retrieval (faiss+torch), reuse the Day-6 discipline — **load the predictor before faiss** and
  keep `OMP_NUM_THREADS=1`. Primary evidence is the Day-7 **pattern repository (pure JSON, no
  faiss)** → the core engine avoids the OpenMP mix entirely; Day-6 retrieval is optional.

## Environment (resolved) 🔧
No new dependencies — reuses Day-4 `PredictionEngine` (calibrated LightGBM), Day-7
`PatternRepository` (JSON), and optionally Day-6 retrieval + Day-5 SHAP. All installed.
Run with `PYTHONPATH=. ./.venv/bin/python`. Perturbation-through-model **verified** in preflight.

---

## Phase A — Understanding Counterfactual AI
**Deliverable:** notes (`docs/day8_phaseA_counterfactual_notes.md`): prediction vs
counterfactual reasoning; what-if / sensitivity analysis; feature perturbation; **model-simulated
vs causal** (the central honesty point); actionable-AI / decision intelligence; why only
controllable features count. Grounded in the real goal-perturbation numbers.

## Phase B — Counterfactual Candidate Generator → `src/counterfactual/candidate_generator.py`
Generate **feasible** changes to **mutable** features only: goal → category funded-median (and
a few multiples), duration → {15, 30, 45}d, launch weekday → Tuesday / quarter → best, title →
add number / adjust length. Respect bounds + immutability. **🔧 Real features only; deferred set
documented.** **Deliverables:** `CandidateGenerator`.

## Phase C — Funding Probability Simulator → `src/counterfactual/probability_simulator.py`
For each candidate, build the modified **raw** campaign, run `PredictionEngine.predict_proba`,
record Δp vs baseline. Provide single-feature Δ and **honest cumulative** (re-simulate the
jointly-modified campaign). **Deliverables:** `ProbabilitySimulator`.

## Phase D — Historical Evidence Validation → `src/counterfactual/evidence_validator.py`
Back each recommendation with real evidence: Day-7 **pattern repository** (the lift/confidence
for the target feature value — e.g. "small-goal campaigns in your category funded at 71%, lift
1.76, n=1,425") + optionally Day-6 retrieval (similar campaigns with that value). **🔧 No
fabricated "214 startups with retention >55%"** — real pattern/retrieval counts only.
**Deliverables:** `EvidenceValidator`.

## Phase E — Improvement Ranking Engine → `src/counterfactual/ranking_engine.py`
Rank recommendations by **predicted probability gain** (primary) × **evidence strength** (tier:
mechanism+pattern+model > model+pattern > model-only) × **feasibility** (mutable ease). Drop
changes with negligible or negative Δp. **🔧 Cost/time are heuristic labels, flagged as such**
(no real cost data). **Deliverables:** `RankingEngine`.

## Phase F — Counterfactual Explanation Generator → `src/counterfactual/explanation_generator.py`
Turn a ranked recommendation into a founder-readable sentence (reuse Day-5/7 readable-feature
map) + **mandatory caption**: "the model estimates … based on historical association, not a
guarantee." **🔧 Examples on real features** ("Lower your goal to ~$5,000: the model's estimate
rises from 23% to 41%; similar small-goal campaigns funded ~1.8× more often"). **Deliverables:**
`CounterfactualExplanationGenerator`.

## Phase G — Counterfactual Pipeline → `src/counterfactual/pipeline.py`
Chain: raw campaign → Day-4 prediction (baseline) → generate candidates → simulate → validate
(Day-7 patterns, optional Day-6 retrieval) → rank → explain → actionable plan. One warm
`PredictionEngine`; load-order-safe if retrieval is enabled. **Deliverables:**
`CounterfactualPipeline`.

## Phase H — Startup Optimization Report → `src/counterfactual/report_generator.py`
Assemble the ranked plan into a report: baseline probability, top-N improvements (each: change,
new probability, evidence, tier), honest **cumulative** optimized probability, and an overall
recommendation. **🔧 Optimized profile uses only real, mutable features.** **Deliverables:**
`ReportGenerator`, `optimization_report.json`.

## Phase I — Serialization → `src/counterfactual/serialization.py`
Persist engine config, recommendation templates, scenario results, and reports as **JSON**;
content-hash for version identity (Day-4/6 pattern). **Deliverables:** reusable engine +
`counterfactual_engine.json`.

---

## Artifacts After Day 8 🔧
```
artifacts/counterfactual/  recommendations.json  ranked_actions.csv  scenario_results.json
                           optimization_report.json  counterfactual_engine.json   (NOT .pkl)
```

## Deliverables Checklist
✅ candidate_generator.py · probability_simulator.py · evidence_validator.py · ranking_engine.py
· explanation_generator.py · report_generator.py · pipeline.py · serialization.py ·
08_Counterfactual_Engine.ipynb · counterfactual recommendation engine (real mutable features) ·
funding-probability simulator (real model) · startup optimization report

## Curriculum / Architecture
Layer 7 (Counterfactual Engine) ✅ · New techniques: counterfactual explanations, what-if /
sensitivity analysis, feature perturbation, decision intelligence. (Mandatory Week 8 —
Evaluation — still ⏳.)

---

## 🔒 Changelog — corrections from the draft
1. **Mutable-features-only.** Recommend changes to goal / duration / launch-timing / title;
   country & currency **immutable**; category "soft" (repositioning note, not a trick).
2. **All fabricated levers deferred** — MRR, retention, burn, team, founder/CTO, market,
   traction, CAC/LTV, stage (don't exist).
3. **Perturb RAW inputs through the real model** (Day-3 pipeline + Day-4 calibrated), so coupled
   features recompute consistently. Verified in preflight.
4. **Candidates seeded by Day-7 patterns** (category funded-median goal, best timing), not
   arbitrary grids.
5. **Cumulative = re-simulated jointly**, never summed (features interact).
6. **Evidence from real Day-7 patterns / Day-6 retrieval** — no invented startup counts.
7. **Model-simulated, not causal** — mandatory caption; tier by evidence strength (goal
   strongest: mechanism+pattern+model).
8. **JSON not pickle**; OpenMP discipline if retrieval enabled (predictor before faiss, OMP=1).

## 💡 Improvements (beyond the draft)
- **Interaction-aware cumulative** probability (re-simulate combined change).
- **Evidence-strength tiers** so the report distinguishes a mechanism-backed lever (goal) from a
  model-only nudge (title word).
- **Immutable-feature guard** — the engine structurally cannot recommend changing country/currency.
- **SHAP (Day-5) optionally explains** *why* each perturbation moved the probability.
- **Feasibility-bounded search** (goal floor, duration ∈ [1,60]) so no absurd advice.

✅ **Spec frozen.** Say "build it" or "build Phase A/B/C…" to implement + teach phase-by-phase
via Framework 1.
