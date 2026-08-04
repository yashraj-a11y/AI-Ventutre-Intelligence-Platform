# Day 11 — Unified Recommendation Engine & Decision Intelligence (FROZEN SPEC V1.1)

**Architecture V1.0 — Layer 11 (Recommendation & Strategy Engine).** The orchestration layer:
aggregate every prior intelligence layer into ONE evidence-backed, prioritized, audience-specific
**venture decision object**. This is the first point the whole architecture acts as a single AI
system.

> **Status:** FROZEN 2026-08-03 (V1.1). Read this before building any phase. Build phase-by-phase
> on the trigger `build Phase <X>` / `build the notebook`; each phase = implement **and** teach via
> Framework 1, grounded in real project numbers.

---

## 0. What changed from the draft (changelog + why)

The draft is architecturally sound — Day 11 genuinely IS the orchestrator, and it legitimately
reuses Days 4–10. But the *worked examples* in the draft describe a SaaS startup, not a Kickstarter
campaign, and several executability assumptions are wrong. Corrections:

| # | Draft said | 🔧 Frozen correction | Why |
|---|---|---|---|
| **C-1** | Examples: "Improve Customer Retention +14%", "recurring revenue > $20K/month", "Strengthen Competitive Moat", "Hire technical co-founder", "312 funded startups" | **DEFERRED — never generated.** Recommendations come ONLY from the real controllable levers (goal / duration / launch timing / title) + category-cohort observations, evidence-backed by Day-7 patterns. | None of retention / MRR / moat / team / co-founder exist in Kickstarter. Same honesty discipline as Days 4–10. |
| **C-2** | "Knowledge Base Retrieval / similar startups" implies Day-6 FAISS is back in the stack | **NO FAISS.** "Similar startups" = the **same-category cohort in the shared pandas item frame** (`cf.evidence.frame`), exactly as the Day-9 copilot already does. Day-6 semantic retrieval stays out of the mandatory path. | Aggregating the existing Day 8/9/10 pipelines is **faiss-free, torch-free → ZERO OpenMP concern** (verified: those pipelines' docstrings all read "no faiss"). Re-introducing FAISS would revive the lightgbm+faiss load-order / `OMP_NUM_THREADS=1` hazard for no honest gain. |
| **C-3** | `recommendation_engine.pkl` | **`recommendation_engine.json`** — content-hash JSON manifest with `verify()`, mirroring `EvolutionSerializer` / `CounterfactualSerializer`. | The engine is a **configuration over existing models**, not a trained binary. JSON-not-pickle is the project-wide rule (Days 4–10). |
| **C-4** | "Recommendation systems (rule-based + ML-assisted)"; recommendation categories = Product / Market / Team / Financial / Technology / GTM | **Deterministic aggregation, NO LLM, NO new model.** Real categories only: **Fundraising readiness** (goal), **Campaign strategy** (duration / timing / title), **Category positioning** (from Day-7 patterns + Day-9 category/competition). Product / Team / Financial / Technology / GTM strategies → **DEFERRED**. | Consistent with Day-9 (deterministic templates, no hallucination). We can't recommend on data we don't have. |
| **C-5** | "Investment Readiness 82%", "Growth Potential", "Startup Score", "Investment Score", "Risk Score" as independent features | **Honest definitions:** Investment Readiness = Day-10 readiness stage + calibrated prob (REAL). Risk = `1 − p` + Day-9 risk level (REAL). **Growth Potential → reframed as "improvement headroom" = `optimized_p − base_p`** (reachable gain, REAL) — NOT a growth trajectory. The vanity "Startup/Investment Score" collapse into ONE transparent, labeled **readiness index**; true multi-month growth potential is **DEFERRED**. | No post-launch trajectory is observable. Composite scores are fine only if transparent and labeled model-derived, not ground truth. |
| **C-6** | Confidence factor "Model agreement" (ambiguous — we have one production model) | **KEPT as a REAL signal** — compare the **three Day-4 models** (`calibrated`, `best`, `logistic_regression`) on the label. Plus the other real factors (see IMP-2). | Three trained models genuinely exist → agreement is measurable, not fabricated. |

**IMPROVEMENTS beyond the draft:**
- **IMP-1 — one truly warm stack.** Add a backward-compatible `cf_pipeline=None` param to
  `VentureCopilotPipeline.__init__` so Day 11 shares ONE `CounterfactualPipeline` (one model in
  memory) across copilot + evolution + scenario + the aggregator's direct `cf.analyze`. Without it,
  the copilot self-builds a second model (~2× load). Small, additive Day-9 edit, flagged here.
- **IMP-2 — a principled, transparent Confidence score** from REAL signals only:
  prediction margin, cohort agreement (frame), Day-7 pattern support, Day-10 scenario stability,
  Day-8 counterfactual stability, and (optional) 3-model agreement — weighted into `[0.5, 0.99]`
  with a per-factor breakdown and a plain-language reason. This is the genuinely new intellectual
  content of Day 11.
- **IMP-3 — honesty-tiers-as-data.** Every report carries `deferred_*` dicts; the decision object
  documents its own boundary.
- **IMP-4 — audience split from ONE intelligence object.** Founder vs investor strategies are two
  *views* of the same aggregated intelligence (no divergent computation), so they can't disagree.

---

## 1. Honesty tiers (what may appear in outputs)

**Tier 1 — REAL, controllable / directly computed (may drive recommendations):**
goal / `usd_goal_real`, campaign_duration (deadline), launch timing (launched/weekday), title;
calibrated funding probability + threshold (Day 4); readiness stage + reachable optimum (Day 10);
category cohort funding rates + competitive intensity (Day 9 over the frame); success/failure
patterns with support counts (Day 7); counterfactual what-ifs (Day 8); scenario table (Day 10).

**Tier 2 — DERIVED, must be labeled model-derived (not ground truth):**
confidence score + factor breakdown; readiness index; improvement headroom (`optimized_p−base_p`);
risk score (`1−p` / Day-9 level); investor recommendation band (Strong / Proceed with Caution /
Monitor / Do Not Invest Yet) mapped from prob + stage + confidence. All captioned
"model-simulated, not causal."

**Tier 3 — DEFERRED, NEVER fabricate (record as `deferred_*`):**
customer retention, MRR / revenue / revenue growth, unit economics / CAC-LTV / burn, competitive
moat / defensibility / named competitors, founder experience / team size / hire co-founder,
pricing strategy, market size / TAM / CAGR, product traction, funding rounds (pre-seed/Series A),
"delay fundraising N months", startup lifecycle (Idea→MVP→PMF), multi-month growth trajectory /
Growth Potential, Product/Team/Financial/Technology/GTM strategy categories.

---

## 2. The honest Unified Venture Decision object

```
VentureDecision = {
  campaign:            {name, main_category, usd_goal_real},
  investment_readiness:{stage, funding_probability, threshold, readiness_index},   # Day 10 + Day 4
  improvement_headroom: optimized_p − base_p,                                       # reachable gain (Tier 2)
  confidence:          {score, factors{margin, cohort_agreement, pattern_support,
                                       scenario_stability, cf_stability, model_agreement},
                        reason},                                                    # Day 11 (IMP-2)
  priority_improvements:[ {rank, action, category, expected_impact_points, evidence,
                           feasibility, confidence} ],                              # Day 8/10 reframed + prioritized
  risks:               [...],          # Day 9 risk_analysis + Day 7 failure patterns
  opportunities:       [...],          # Day 7 success patterns + category headroom
  founder_strategy:    {headline, actions[], caption},                             # deterministic view
  investor_strategy:   {recommendation, reason, confidence},                       # deterministic view
  final_strategy:      "<deterministic synthesis sentence>",
  deferred:            {...aggregated Tier-3...},
  honesty_note:        "..."
}
```

**Investor recommendation band (honest mapping, real numbers):**
- **Strong Investment** — `base_p ≥ threshold + strong_margin` (Strong stage) AND confidence high.
- **Proceed with Caution** — Ready, or Fixable with `optimized_p` comfortably above bar.
- **Monitor Progress** — Fixable but borderline / low confidence.
- **Do Not Invest Yet** — Not ready (`optimized_p < threshold`, even optimized).

---

## 3. Environment & reuse

- **NO NEW DEPS. NO FAISS. NO TORCH. NO OpenMP concern** (mandatory path). lightgbm + pandas only,
  same as Day 10. Run `PYTHONPATH=. ./.venv/bin/python`.
- **Reuse (build ONCE, inject everywhere):** ONE `CounterfactualPipeline` (Day-8 model + item
  frame) shared into `VentureCopilotPipeline` (Day 9, via IMP-1), `EvolutionPipeline` +
  `ScenarioPipeline` (Day 10), and the aggregator's own `cf.analyze` / `LocalExplainer` (Day 5).
- **PREFLIGHT (before Phase B):** import cf + copilot + evolution + scenario together, run all four
  `analyze/run` on one campaign in one process, confirm no error and no second model load when the
  cf is injected. Confirm faiss/torch absent from the import graph.

---

## 4. Phases (build order) — corrections folded in

### Phase A — Decision Intelligence notes (conceptual)
Why multiple ML outputs must be *synthesized* before recommending: multi-model orchestration,
recommendation hierarchy, evidence aggregation, confidence as a first-class output. Ground it in
THIS platform: nine layers already exist but behave as separate systems; the aggregator is the
"brain." **Deliverable:** `docs/day11_phaseA_decision_intelligence_notes.md` + `src/recommendation/__init__.py`.

### Phase B — Intelligence Aggregation Engine → `src/recommendation/intelligence_aggregator.py`
Collect outputs from all prior layers into ONE `VentureIntelligence` object. Composes the shared
cf + copilot + evolution + scenario (all on one warm stack). Merges: prediction+threshold (Day 4),
SHAP top features (Day 5), Day-7 patterns (via cf evidence), counterfactual `ranked_recommendations`
+ `optimization_plan` (Day 8), copilot `investor_objections`/`risk_analysis`/`swot`/`category`/
`competition`/`recommendations` (Day 9), evolution `stage`/`roadmap`/`milestones`/`timeline`
(Day 10), scenario `ranked_scenarios`/`marginal_values`/`best`/`efficient` (Day 10). **"Similar
startups" = same-category cohort from `cf.evidence.frame` (C-2, no faiss).** Config `day11.aggregator`.
**Deliverable:** `IntelligenceAggregator.aggregate(raw) → VentureIntelligence`.

### Phase C — Recommendation Generator → `src/recommendation/recommendation_generator.py`
Generate strategic recommendations from the aggregated intelligence. **Real categories only (C-4):**
Fundraising readiness (goal), Campaign strategy (duration/timing/title), Category positioning
(Day-7/9). Each recommendation = {action, category, priority_label, expected_impact_points, reason,
evidence (Day-7 statement), confidence}. Reuses Day-8 `ranked_recommendations` / Day-10 milestones
as the source of truth (no re-derivation). Product/Team/Financial/Tech/GTM → `deferred_categories`.
Config `day11.generator`. **Deliverable:** `RecommendationGenerator`.

### Phase D — Recommendation Prioritization → `src/recommendation/prioritization.py`
Rank recommendations by a transparent weighted score over REAL factors: funding-probability
improvement (Day 8 delta), historical evidence strength (Day 7 support), investor-concern overlap
(Day 9 objections), scenario gain (Day 10 marginal value), feasibility (Day 8), confidence. No
re-ranking that contradicts Day 8 — this ADDS the cross-layer signals (investor overlap, scenario
gain) on top. Output: ordered `prioritized_actions` with per-factor breakdown. Config
`day11.prioritization`. **Deliverable:** `RecommendationPrioritizer`.

### Phase E — Founder Strategy Engine → `src/recommendation/founder_strategy.py`
Founder-facing view of the SAME intelligence (IMP-4). Deterministic templates over the real levers:
e.g. "Lower your goal to $X before launch", "Shorten the campaign to N days", "Time your launch for
Tuesday", ordered as a pre-launch action list, tied to the readiness stage + confidence. **DEFERRED
founder advice** (retention/pricing/co-founder/MRR/"delay 3 months") recorded, not emitted. Config
`day11.founder`. **Deliverable:** `FounderStrategyEngine`.

### Phase F — Investor Strategy Engine → `src/recommendation/investor_strategy.py`
Investor-facing view. Emits one of **Strong Investment / Proceed with Caution / Monitor Progress /
Do Not Invest Yet** (band mapping §2), a reason grounded in the readiness stage + top risk (Day 9)
+ headroom, and the confidence. Config `day11.investor`. **Deliverable:** `InvestorStrategyEngine`.

### Phase G — Confidence Scoring Engine → `src/recommendation/confidence.py` (IMP-2, the new core)
Estimate confidence in the final recommendation from REAL signals, each in `[0,1]`, weighted:
1. **prediction margin** — `min(|base_p − τ|/scale, 1)` (decisive vs borderline).
2. **cohort agreement** — fraction of same-category cohort (frame) whose outcome matches the
   predicted label (no faiss).
3. **pattern support** — normalized Day-7 support behind the top recommendations.
4. **scenario stability** — Day-10 monotonicity / low erratic spread of scenario probs.
5. **cf stability** — Day-8 recommendations robust (deltas not wildly sensitive).
6. **model agreement (optional, REAL)** — 3 Day-4 models agree on the label (C-6).
Aggregate → `score ∈ [0.5, 0.99]` + factor breakdown + plain reason ("supported by similar
campaigns, historical patterns, and stable scenarios"). Config `day11.confidence` (weights).
**Deliverable:** `ConfidenceScorer`.

### Phase H — Unified Recommendation Pipeline → `src/recommendation/pipeline.py`
One entry point building the full `VentureDecision`: aggregate → generate → prioritize → confidence
→ founder view → investor view → final strategy synthesis. ONE warm stack, ONE model-heavy pass
threaded (reuse `analysis=` where modules accept it). Config `day11.pipeline`
(→ `artifacts/recommendation/unified_recommendations.json`). **Deliverable:** `RecommendationPipeline`.

### Phase I — Serialization → `src/recommendation/serialization.py` (C-3)
`RecommendationSerializer` mirroring `EvolutionSerializer`: content-hash JSON manifest
(`recommendation_engine.json`) over the Day-11 config (aggregator/generator/prioritization/
confidence weights/investor bands) + `readiness_definition`; **dependency hashes** = Day-4 model +
Day-8/9/10 manifests; **artifact hashes** = the produced JSONs; `deferred_items`; `build/save/
verify/describe`. JSON not pickle. **Deliverable:** reusable, verifiable engine.

### Notebook — `notebooks/11_Recommendation_Engine.ipynb`
End-to-end on ONE worked campaign: intelligence aggregation → prioritized recommendations →
confidence with factor breakdown → founder view → investor band → final strategy → serialized
manifest verify. Show the honesty boundary (deferred list) explicitly. One warm stack; assert
shared-cf identity.

---

## 5. Repository & artifacts after Day 11

```
src/recommendation/{__init__, intelligence_aggregator, recommendation_generator,
                    prioritization, founder_strategy, investor_strategy, confidence,
                    pipeline, serialization}.py
notebooks/11_Recommendation_Engine.ipynb
artifacts/recommendation/{unified_recommendations, founder_strategy, investor_strategy,
                          confidence_scores, prioritized_actions, recommendation_engine}.json   # all JSON (C-3)
configs/preprocessing.yaml → day11.{aggregator, generator, prioritization, founder,
                                     investor, confidence, pipeline, serialization}
docs/day11_phaseA_decision_intelligence_notes.md
```

**Input contracts (confirmed against code):**
- `cf.analyze(raw)` → `baseline_p`, `ranked_recommendations` (evidence DICT), `optimization_plan`, `meta`.
- `copilot.analyze(raw)` → `funding_probability`, `predicted_label`, `decision_threshold`,
  `investor_objections`, `risk_analysis`, `swot`, `category_analysis`, `competition_analysis`,
  `recommendations` (evidence STRING). ⚠️ Day-8 API duality (dict vs string evidence) applies.
- `evo.run(raw)` → `stage`, `roadmap`, `milestones`, `timeline`, `deferred`.
- `scn.run(raw)` → `ranked_scenarios`, `marginal_values`, `best_scenario`, `most_efficient_scenario`, `recommendation`.

---

## 6. Non-negotiables (frozen)

1. **No fabrication** — retention/MRR/moat/team/pricing/growth-trajectory DEFERRED, never emitted (C-1, C-5).
2. **No faiss / no torch / no OpenMP** on the mandatory path — aggregate the existing frame-based
   pipelines; "similar startups" = category cohort (C-2).
3. **Deterministic, no LLM, no new trained model** (C-4).
4. **JSON not pickle**, content-hash + `verify()` (C-3).
5. **ONE warm stack** — build cf once, inject everywhere (IMP-1); founder & investor are two views
   of ONE intelligence object (IMP-4).
6. **Every recommendation evidence-backed** (Day-7 support) and **confidence transparent** (IMP-2).
7. Teach every phase via Framework 1, grounded in real numbers; end each with an interview soundbite.

---

*Frozen 2026-08-03. Supersedes the Day-11 draft. Build on `build Phase <X>`.*
