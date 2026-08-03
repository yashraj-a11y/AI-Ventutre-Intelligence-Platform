# 🚀 Day 9 — Investor Objection Simulator & Venture Copilot (Roadmap V1.1 — FROZEN)

> **Status: FROZEN engineering contract** (2026-08-01). Corrections from the draft flagged 🔧;
> changelog + honesty tiers at the end. Implements Architecture **Layer 8 — Investor Objection
> Simulator** + the first phase of the **Venture Copilot**. Synthesises the *real* outputs of
> Days 4–8 into VC-style objections / risk / SWOT / category & competition analysis /
> recommendations — via **deterministic templates, not an LLM**.

## Goal
Prepare a founder for a pitch: from a campaign, generate the investor concerns a data-driven
VC would raise, the risks, a SWOT, a category/competition read, and strategic recommendations —
each **grounded in a real signal** from a prior layer (prediction, SHAP, patterns,
counterfactuals) and framed honestly (model/historical evidence, not VC ground truth).

---

## Data / model reality (Source of Truth) 🔧
Day 9 **computes nothing new** — it *reframes* the real outputs of prior layers into investor
language. Available real signals:

- **Day 4** — calibrated funding probability, label, threshold, venture scores (funding_prob,
  failure_risk, startup_score, investment_readiness).
- **Day 5** — SHAP contributions (which real features push the probability up/down).
- **Day 7** — success/failure patterns, category profiles, year-trend (saturation), the
  campaign's matching patterns + lift.
- **Day 8** — counterfactual improvements (goal/duration/timing/title) + the optimization plan.
- **Day 6** — semantic retrieval cohort (**OPTIONAL**; uses faiss — see OpenMP note).

**FABRICATED — the draft's staples DO NOT EXIST → DEFER, never generate** 🔧 (same discipline as
Days 4–8): named competitors (**OpenAI / Microsoft / Anthropic** — absurd for Kickstarter),
**TAM / SAM / SOM / market CAGR**, founder/team quality, traction, **unit economics / burn /
retention / runway**, business model, moat / proprietary-data / network-effects, regulatory &
technology risk. The draft's examples ("Why won't Microsoft build this?", "AI Healthcare CAGR
23%", "71% exhausted runway before Series A", "recurring revenue") are invention.

### Non-obvious FROZEN decisions (easy to get wrong)
- **NO LLM** 🔧. The draft says "LLM reasoning / LLMs orchestrate modules." We use **deterministic
  template synthesis** over the real ML outputs — reproducible, free, offline, and it **cannot
  hallucinate** a competitor or a market number, which is the whole point. An LLM layer is
  optional *future* work (needs an API key, is non-deterministic, and would break the honesty
  discipline). Every phase is a template map from real signals to investor framing.
- **Objections from STRUCTURED dimensions, not raw SHAP tokens** 🔧. Raw SHAP surfaces noisy
  title tokens ("your title has too many capitals" is not a VC concern). Generate objections
  from a fixed set of *assessable* dimensions — **funding-odds, goal size, category headwind,
  timing/duration, cohort** — each fired only when its real evidence supports it, severity from
  the magnitude (SHAP contribution, pattern lift, probability gap).
- **"Market Intelligence" → Category / Segment Analysis** 🔧 (Phase E). No external market data
  exists. Honest market read = *our own data by category*: category funding rate + lift, the
  **year-trend** (Day-7 saturation), **volume/crowding**, and goal norms. DEFER TAM/SAM/SOM/CAGR.
- **"Competitor Intelligence" → Category Competition Analysis** 🔧 (Phase F). No named
  competitors. Honest competition read = *other campaigns in your category*: crowding (category
  volume), **what winners do differently** (funded-vs-failed norms from Day-7), your position vs
  the cohort. DEFER named companies / moat / proprietary-data.
- **Recommendations reuse Day-8 counterfactuals** 🔧 (Phase G) — don't duplicate. Add strategic
  framing (delay/reconsider if probability far below threshold; category-trend advice). Real,
  controllable actions only (goal/duration/timing) — never "acquire an advisor / expand the team".
- **No faiss in the core → no OpenMP issue** 🔧. Core uses Day-4 + Day-5 SHAP + Day-7 + Day-8
  (lightgbm/shap/pandas). **Verified in preflight**: SHAP + counterfactual in one process, no
  segfault. Day-6 retrieval is OPTIONAL; if enabled, reuse Day-6's load-order-safe pipeline
  (predictor before faiss, `OMP_NUM_THREADS=1`).
- **Honesty caption on everything** 🔧: outputs are model predictions + historical associations
  reframed as VC-style reasoning, **not** a real investor's judgement or verified market facts.
- **JSON, not pickle** 🔧 (`venture_copilot.pkl` → `venture_copilot.json`).
- **`src/copilot/` already exists** — Day-5 created `venture_copilot.py` (honesty-tiered
  SWOT-lite prototype). Day 9 adds the new modules; the Day-9 `VentureCopilotPipeline`
  supersedes/extends that prototype (Phase D's `swot_analysis.py` is the full SWOT).

## Environment (resolved) 🔧
No new dependencies — reuses Day-4 `PredictionEngine`, Day-5 `LocalExplainer` (SHAP), Day-7
patterns / item frame, Day-8 `CounterfactualPipeline`. All installed. Preflight verified (SHAP +
counterfactual coexist, no segfault). Run with `PYTHONPATH=. ./.venv/bin/python`.

---

## Phase A — Understanding Venture Capital Thinking
**Deliverable:** notes (`docs/day9_phaseA_vc_thinking_notes.md`): the VC decision framework, and
**which dimensions we can honestly assess** (funding odds, goal appropriateness, category/timing
headwinds, cohort comparison) **vs which we defer** (founder, market size, named competition,
financials). Grounded in the real preflight signals.

## Phase B — Investor Objection Generator → `src/copilot/objection_generator.py`
Generate concerns from **structured dimensions**, each with `{concern, severity, reason
(real evidence), suggested_response (from Day-8 counterfactual)}`: below-threshold odds; high
goal (SHAP + pattern + counterfactual); category headwind (lift < 1); long/short timing; weak
cohort. Rank by severity. **🔧 No named-competitor objections.** **Deliverables:**
`ObjectionGenerator`, `investor_questions.json`.

## Phase C — Risk Analysis Engine → `src/copilot/risk_analysis.py`
Real, assessable risk categories: **funding risk** (1 − prob, vs threshold), **goal/financial
risk** (goal too high), **category-fit risk** (low-success category), **timing risk**
(duration/launch), **cohort risk** (similar campaigns mostly failed). Each `{category, severity,
reason, historical_evidence (real lift/rate), source_layer}`. **🔧 DEFER** technology/founder/
regulatory/team risk (no data). **Deliverables:** `RiskAnalysisEngine`, `risk_report.json`.

## Phase D — SWOT Analysis Engine → `src/copilot/swot_analysis.py`
From real signals (supersedes Day-5 SWOT-lite): **Strengths** = SHAP positives + matching
success patterns; **Weaknesses** = SHAP negatives + matching failure patterns; **Opportunities**
= Day-8 improvement levers; **Threats** = category/timing/cohort headwinds + below-threshold odds.
**🔧 DEFER** market/competitor threats (flagged, not fabricated). **Deliverables:**
`SWOTAnalysisEngine`, `swot_report.json`.

## Phase E — Category & Segment Analysis 🔧 (reframed from "Market Intelligence") → `src/copilot/market_analysis.py`
Own-data category read: funding rate + lift, **year-trend** (rising/declining — Day-7
saturation), **volume/crowding**, goal norms (funded vs failed median). **🔧 DEFER**
TAM/SAM/SOM/CAGR/external market data. **Deliverables:** `CategoryAnalysisEngine`,
`market_analysis.json` (with a documented `deferred: external market data` note).

## Phase F — Category Competition Analysis 🔧 (reframed from "Competitor Intelligence") → `src/copilot/competitor_analysis.py`
Competition within the category: crowding (volume), **what funded campaigns do differently**
(funded-vs-failed goal/duration norms), the campaign's position vs the cohort. **🔧 DEFER** named
competitors / moat / proprietary-data. **Deliverables:** `CompetitionAnalysisEngine`,
`competitor_analysis.json`.

## Phase G — Strategic Recommendation Engine → `src/copilot/recommendation_generator.py`
Reuse the **Day-8 counterfactual plan** as the core actions; add strategic framing: a
**delay/reconsider** note if probability ≪ threshold, and a **category-trend** note (e.g.
"funding rates in your category have declined"). **🔧 Real controllable actions only.**
**Deliverables:** `RecommendationEngine`, `recommendations.json`.

## Phase H — Venture Copilot Pipeline → `src/copilot/pipeline.py`
Integrate every layer into one copilot brief: Day-4 prediction + Day-5 SHAP + Day-7 patterns +
Day-8 counterfactuals → objections + risks + SWOT + category analysis + competition +
recommendations. One warm stack (shared `PredictionEngine`, `LocalExplainer`, item frame,
`CounterfactualPipeline`); **no faiss → no OpenMP**. **Deliverables:** `VentureCopilotPipeline`,
`venture_copilot_brief.json`.

## Phase I — Serialization → `src/copilot/serialization.py`
Persist the copilot config + templates + dependency hashes (Day-4/5/7/8 artifacts) as **JSON**
(`venture_copilot.json`) with a content hash + `verify()` (Day-4/6/8 pattern). **Deliverables:**
reusable copilot.

---

## Artifacts After Day 9 🔧
```
artifacts/copilot/  investor_questions.json  risk_report.json  swot_report.json
                    market_analysis.json  competitor_analysis.json  recommendations.json
                    venture_copilot_brief.json  venture_copilot.json   (NOT .pkl)
```

## Deliverables Checklist
✅ objection_generator.py · risk_analysis.py · swot_analysis.py · market_analysis.py (category) ·
competitor_analysis.py (category competition) · recommendation_generator.py · pipeline.py ·
serialization.py · 09_Investor_Objection_Simulator.ipynb · investor objection simulator · SWOT ·
risk · category & competition analysis · strategic recommendations — **all from real signals,
deterministic, honesty-tiered**

## Curriculum / Architecture
Layer 8 (Investor Objection Simulator) + Venture Copilot Phase 1 ✅ · New techniques: VC decision
framing, risk/SWOT/competition synthesis, template reasoning over structured ML outputs.
(Mandatory Week 8 — Evaluation — still ⏳.)

---

## 🔒 Changelog — corrections from the draft
1. **Scoped to real signals** — everything reframes Days 4–8 outputs; founder/market/competitor/
   financial fabrications deferred.
2. **NO LLM** — deterministic template synthesis (reproducible, offline, can't hallucinate);
   LLM orchestration is optional future work.
3. **Market → Category/Segment analysis** (own-data category rate + year-trend + crowding);
   **Competitor → Category competition** (cohort norms). No named companies / CAGR / TAM.
4. **Objections from structured dimensions**, not noisy raw SHAP tokens; severity from real magnitude.
5. **Recommendations reuse Day-8 counterfactuals**; real controllable actions only.
6. **No faiss in the core → no OpenMP** (preflight-verified); Day-6 retrieval optional + load-order-safe.
7. **Honesty caption** on every output; **JSON not pickle**.
8. **Reuses** Day-4/5/7/8 (no new deps, no duplication); notes existing `src/copilot/venture_copilot.py`.

## 💡 Improvements (beyond the draft)
- **Every objection/risk carries its source layer + real evidence** (SHAP magnitude / pattern lift
  / probability gap) — auditable, not asserted.
- **Severity grounded in magnitude**, not hand-set labels.
- **Cross-layer coherence**: SWOT weaknesses ↔ risks ↔ objections ↔ counterfactual levers all
  trace to the same real signals (a weakness *is* an objection *is* a fixable lever).
- **Honesty tiers as data**: deferred analyses (market/competitor/founder/financial) are explicit
  placeholders in every report.

✅ **Spec frozen.** Say "build it" or "build Phase A/B/C…" to implement + teach phase-by-phase
via Framework 1.
