# 🚀 Day 10 — Startup Evolution Planner & Scenario Simulation (Roadmap V1.1 — FROZEN)

> **Status: FROZEN engineering contract** (2026-08-03). Corrections from the draft flagged 🔧;
> changelog + honesty tiers at the end. Implements Architecture **Layer 9 (Evolution Planner)**
> + **Layer 10 (Scenario Simulator)**. Introduces readiness-stage classification, a pre-launch
> preparation roadmap, and multi-variable **scenario comparison** over real controllable features.

## Goal
Answer "what should I do to become fundable, and which path is best?" — by (1) classifying the
campaign's **funding-readiness stage**, (2) giving a **pre-launch preparation roadmap** of
controllable changes, and (3) generating, simulating, and **comparing multiple named scenarios**
(combinations of controllable changes) through the real Day-4 model. The new value over Day 8
(one greedy optimum) is *comparing discrete strategies* and their marginal value.

---

## Data / model reality (Source of Truth) 🔧
The dataset is **Kickstarter pre-launch campaigns** — a single artifact per campaign, with **no
post-launch, growth, or lifecycle observation**. So the draft's startup lifecycle
(Idea → MVP → PMF → Series A) and its 12-month growth journey **cannot be modeled** — we never
observe a campaign "acquiring users" or "reaching PMF". What we *can* do is reason about the
**controllable pre-launch levers** and the campaign's readiness.

**MUTABLE (controllable) variables — the only scenario dimensions** (same as Day 8): `goal` /
`usd_goal_real`, `campaign_duration` (via deadline), launch timing (via launched), title.

**FABRICATED — the draft's stages/milestones/scenarios DO NOT EXIST → DEFER, never generate** 🔧:
lifecycle stages (MVP/Validation/PMF/Series A), **customer interviews / MVP / first 100 users /
retention / MRR**, experienced founder, pre-seed/Series A rounds, unit economics, strategic
partnerships, a **12-month growth timeline**. The draft's scenarios ("Revenue $20K, Retention
60%, Experienced Founder") are invention — there is no revenue/retention/founder data.

### Non-obvious FROZEN decisions (easy to get wrong)
- **Startup "stage" → funding-READINESS stage** 🔧 (model-derived, not lifecycle). Classify from
  the calibrated probability, the threshold, and the **optimized** probability (Day-8): e.g.
  *Not ready* (even optimized < bar) · *Fixable* (as-is < bar, optimized ≥ bar) · *Ready*
  (as-is ≥ bar) · *Strong* (well above bar). Real, honest, no MVP/PMF.
- **"12-month roadmap" → pre-launch PREPARATION roadmap** 🔧: the controllable changes (Day-8
  levers) ordered as a prep checklist — set the right goal, choose duration, pick a launch
  window, refine the title. Days/weeks of *preparation*, **not** a 12-month growth journey.
- **Scenarios = multi-variable combinations of REAL controllable changes** 🔧, composed with a
  **canonical apply-order that recomputes derived fields** — NOT a naive merge of Day-8
  `raw_changes` (which double-edits `deadline`). **Verified bug**: naive merge of {goal, duration,
  weekday} gave a *regression* (0.375 → 0.059) because duration and weekday both set `deadline`
  from the original; canonical order (goal → shift `launched` to target weekday → set `deadline`
  = shifted-launched + duration → title) fixes it (0.384, monotone). This is mandatory.
- **Simulate through the real Day-4 calibrated model** (reuse Day-8 `ProbabilitySimulator`);
  each scenario is a fresh model call, never summed. **Model-simulated, not a causal guarantee.**
- **Scenario comparison is the new capability** 🔧: Day 8 gave one greedy optimum; Day 10 shows
  *multiple discrete strategies* side by side (goal-only vs goal+duration vs all), ranked, with
  the best recommended and each's readiness verdict — so a founder sees tradeoffs, not just the max.
- **Reuse Day-8 heavily**: `CandidateGenerator` for target values, `ProbabilitySimulator` for
  scoring. Roadmap/milestones/timeline are three framings of the same controllable levers — note
  the overlap; don't pretend they're independent analyses.
- **No LLM, deterministic** (Day-9 discipline). **No faiss → no OpenMP** (lightgbm/pandas only) —
  preflight-verified. **JSON, not pickle** (`evolution_engine.pkl` / `scenario_engine.pkl` → `.json`).

## Environment (resolved) 🔧
No new dependencies — reuses Day-8 `CounterfactualPipeline` (model + candidates + simulator) and
Day-9 copilot (optional context). Preflight verified: canonical multi-variable scenario
composition + simulation works (Reel Kids Film $38k: 0.035 → 0.38 "READY" via goal+duration).
Run with `PYTHONPATH=. ./.venv/bin/python`.

---

## Phase A — Understanding Startup Evolution
**Deliverable:** notes (`docs/day10_phaseA_evolution_notes.md`): the lifecycle concept **and why
Kickstarter data can't observe it** (single pre-launch artifact); the honest reframe to
funding-readiness stages + pre-launch preparation; scenario analysis + dynamic prediction.

## Phase B — Startup Stage Detection → `src/evolution/stage_detector.py`
Classify the **funding-readiness stage** from (calibrated prob, threshold, Day-8 optimized prob):
*Not ready / Fixable / Ready / Strong*, with a confidence and a real reason (prob vs bar, gap
closeable by controllable changes). **🔧 Not the MVP/PMF lifecycle.** **Deliverables:**
`StageDetector`, `startup_stage.json`.

## Phase C — Evolution Planner (pre-launch prep roadmap) → `src/evolution/roadmap_generator.py`
Order the controllable changes (Day-8 greedy plan) into a preparation roadmap from current →
readiness. **🔧 Real levers only** (goal/duration/timing/title), not customer-interviews/MVP.
**Deliverables:** `RoadmapGenerator`, `roadmap.json`.

## Phase D — Milestone Recommendation Engine → `src/evolution/milestone_engine.py`
Rank the controllable improvements as milestones by predicted impact × evidence × feasibility
(reuse Day-8 ranking). **🔧 Real actions** ("set your goal to the category norm", "shorten to
30 days"), not "expand founding team / improve unit economics". **Deliverables:**
`MilestoneEngine`, `milestones.json`.

## Phase E — Scenario Generator → `src/scenario/scenario_generator.py`
Generate named multi-variable scenarios = subsets/combinations of the controllable changes
(current, goal-only, goal+duration, goal+duration+timing, all), each composed via the **canonical
apply-order** (recompute derived fields). **🔧 Real controllable variables only** — no
revenue/retention/founder scenarios. **Deliverables:** `ScenarioGenerator`, `scenarios.json`.

## Phase F — Scenario Simulation Engine → `src/scenario/simulator.py`
Run the real Day-4 model on each composed scenario → funding probability + readiness verdict
(reuse Day-8 `ProbabilitySimulator`). Optionally attach Day-5 SHAP / Day-9 risk per scenario.
**Deliverables:** `ScenarioSimulator`, `simulation_results.json`.

## Phase G — Scenario Comparison Engine → `src/scenario/comparison.py`
Rank scenarios by probability, show gain vs current + marginal value of each added change, flag
which cross the readiness bar, recommend the best (highest prob, or fewest changes to reach
"Ready"). **Deliverables:** `ScenarioComparison`, `comparison_report.json`.

## Phase H — Funding Readiness Timeline → `src/evolution/timeline_generator.py`
Sequence the recommended controllable changes into an ordered **pre-launch preparation** plan
(finalize goal → set duration → choose launch date → refine title → launch). **🔧 Pre-launch
prep steps, NOT a 12-month post-launch growth timeline** (no growth data). **Deliverables:**
`TimelineGenerator`, `funding_timeline.json`.

## Phase I — Evolution & Scenario Pipeline → `src/evolution/pipeline.py`
Integrate: campaign → stage detection → roadmap + milestones → scenario generation + simulation
+ comparison → timeline → best growth strategy. One warm Day-8 stack shared; no faiss.
**Deliverables:** `EvolutionPipeline` (+ a thin `src/scenario/pipeline.py` for the scenario-only flow).

## Phase J — Serialization → `src/evolution/serialization.py`
Persist config + templates + dependency hashes (Day-4/8 artifacts) as **JSON**
(`evolution_engine.json`) with content hash + `verify()` (Day-4/6/8 pattern). **Deliverables:**
reusable evolution engine.

---

## Artifacts After Day 10 🔧
```
artifacts/evolution/  startup_stage.json  roadmap.json  milestones.json  funding_timeline.json
                      evolution_engine.json   (NOT .pkl)
artifacts/scenario/   scenarios.json  simulation_results.json  comparison_report.json
```

## Deliverables Checklist
✅ stage_detector.py · roadmap_generator.py · milestone_engine.py · timeline_generator.py ·
scenario_generator.py · simulator.py · comparison.py · pipeline.py · serialization.py ·
10_Evolution_Planner_And_Scenario_Simulator.ipynb · readiness stage · prep roadmap · milestones ·
multi-variable scenario comparison · pre-launch timeline — **all real controllable features,
model-simulated, honesty-tiered**

## Curriculum / Architecture
Layers 9 + 10 ✅ · New techniques: readiness-stage classification, scenario generation &
comparison, dynamic prediction, decision optimization. (Mandatory Week 8 — Evaluation — still ⏳.)

---

## 🔒 Changelog — corrections from the draft
1. **Scoped to real controllable variables** — lifecycle/MVP/revenue/retention/founder/rounds
   deferred (no data).
2. **Lifecycle stages → funding-readiness stages** (model-derived); **12-month roadmap → pre-launch
   preparation** (no growth data).
3. **Scenarios = real controllable combinations**, composed via a **canonical apply-order** that
   recomputes derived fields — fixing a verified naive-merge regression (deadline double-edit).
4. **Simulate through the real model**; scenario comparison (multiple strategies) is the new value
   over Day-8's single optimum.
5. **Reuse Day-8** (candidates + simulator); roadmap/milestones/timeline are framings of one lever set.
6. **Model-simulated, not causal**; **no LLM**; **no faiss → no OpenMP**; **JSON not pickle**.

## 💡 Improvements (beyond the draft)
- **Canonical scenario composition** (order-aware, derived-field-consistent) — prevents the
  silent regression naive merging causes.
- **Readiness-stage classification** grounded in optimized-prob-vs-threshold — a real maturity
  signal from the model, not a fabricated lifecycle.
- **Marginal-value comparison** — each scenario shows the incremental gain of its added change,
  so a founder sees which changes actually matter (and which don't help).
- **Honesty tiers as data**: deferred lifecycle/growth/founder dimensions explicit in every report.

✅ **Spec frozen.** Say "build it" or "build Phase A/B/C…" to implement + teach phase-by-phase
via Framework 1.
