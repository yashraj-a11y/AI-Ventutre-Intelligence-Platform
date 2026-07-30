

# 🚀 Day 7 — Pattern Mining Engine & Venture Pattern Discovery (Roadmap V1.1 — FROZEN)

> **Status: FROZEN engineering contract** (2026-07-30). Corrections from the draft flagged 🔧;
> changelog + improvements + honesty tiers at the end. Implements Architecture **Layer 6 —
> Pattern Mining Engine**. Introduces association-rule mining (support / confidence / lift),
> statistical pattern extraction, and cluster-based discovery — **all on real features**.

## Goal
Discover the recurring, statistically-grounded patterns in **real** pre-launch campaign
features that separate funded from failed campaigns, rank them by rigorous measures
(support / confidence / lift / significance), and explain them honestly (association, **not**
causation). The output is a reusable **pattern repository** that downstream layers
(counterfactuals, recommendations, copilot) consume.

---

## Data / model reality (Source of Truth) 🔧
Mining runs on the **train knowledge base** already built in Day 6
(`artifacts/embeddings/metadata.csv`, **265,171 train campaigns**, train-only → patterns are
discoverable knowledge, validatable on the untouched test set). **Train base funding rate =
0.4038** — every lift is measured against this.

**Real features available to mine** (discretized into categorical *items*): `main_category`
(15), `category` (159), `country` (22), `currency`, `goal_band` (5, reuse Day-6 `goal_band`),
`launch_year`, `launch_month`, `launch_weekday`, `launch_quarter`, `is_weekend_launch`, plus
derivable **campaign_duration band**, **title-length band**, **word-count band**, `has_number`
(all from the 12 columns / Day-3 feature builder).

**Real patterns exist and are strong** (measured, train): Dance funds at 64.8% (lift 1.60),
Technology 23.7% (lift 0.59); very-small goal lift 1.36 vs very-large goal lift **0.25**; the
2-feature rule `{Dance, small goal} → funded` has confidence 0.711, **lift 1.76**.

**FABRICATED — the draft's entire pattern lists DO NOT EXIST → DEFER, never mine** 🔧
(same discipline as Days 4–6): experienced/technical **founders**, **TAM / market size**,
**traction**, **recurring revenue**, **retention**, **CAC / LTV / unit economics**, **burn
rate**, **product-market fit**, **funding rounds / stage**, **team composition / size**,
**revenue**, **business model**, **user / revenue growth**, **product iteration**. There is no
founder / market / financial / time-series data. Any pattern citing these is invented.

### Non-obvious FROZEN decisions (easy to get wrong)
- **"Growth Pattern Mining" is impossible here** 🔧 — Kickstarter records a *single* outcome
  (funded/failed), not a growth trajectory; there is no user/revenue/retention time series.
  **Phase E is reframed to TEMPORAL / TIMING patterns** (real: how funding rate trends across
  `launch_year`, and seasonality by month / quarter / weekday / weekend — the honest slice of
  "market timing"). True growth patterns are **deferred**. (Module stays `growth_patterns.py`
  to match the checklist; content = temporal patterns + a documented growth-deferral.)
- **Lift is relative to the 0.4038 base rate**: `lift = confidence(rule→funded) / 0.4038`.
  Lift > 1 = success-leaning; lift < 1 = failure-leaning. Report support, confidence, **and**
  lift — never confidence alone (a 60%-confidence rule with lift 1.0 says nothing).
- **Multiple-testing rigor** 🔧: mining thousands of feature-value rules guarantees spurious
  "significant" ones. Enforce **min support** (e.g. ≥ 0.2% ≈ 500 campaigns), **min lift**
  (|lift−1| ≥ 0.15), and a **chi-square / Fisher significance test** with a **Bonferroni note**
  on the comparison count. A pattern must clear support **and** effect size **and** significance.
- **Association, NOT causation** 🔧: every pattern is a historical correlation. "Dance
  campaigns funded more" ≠ "being Dance causes funding." Phase-G explanations carry a mandatory
  honesty caption; no causal/PMF/investor-psychology narratives (the draft's Phase-G example is
  forbidden).
- **From-scratch focused miner, no heavy dep** 🔧: with ~13 categorical items × a few values,
  enumerate 1- and 2-item rules → outcome directly (support/confidence/lift/chi2). No `mlxtend`;
  `scipy.stats` (already present) for significance. Teaches the concept per Framework 1.
- **Repository = JSON, not pickle** 🔧 (`pattern_repository.pkl` → `pattern_repository.json`):
  patterns are pure data; JSON is portable and honest (Day-6 lesson).
- **No faiss/torch/lightgbm in Day 7** — mining is descriptive over the metadata table;
  clustering uses `MiniBatchKMeans` (sklearn) on the Day-6 embeddings via `np.load` (numpy
  only). So **none of the OpenMP conflicts from Day 6 apply**.

## Environment (resolved) 🔧
No new dependencies. Uses `pandas`, `numpy`, `scipy.stats` (chi-square / Fisher), `sklearn`
(`MiniBatchKMeans`) — all installed. Mining input = `artifacts/embeddings/metadata.csv`;
optional cluster features = `artifacts/embeddings/startup_embeddings.npy` (numpy load, no torch).
Run with `PYTHONPATH=. ./.venv/bin/python`.

---

## Phase A — Understanding Pattern Mining
**Deliverable:** notes (`docs/day7_phaseA_pattern_mining_notes.md`): prediction vs retrieval vs
pattern mining; frequent itemsets; **support / confidence / lift** (grounded in real numbers);
**correlation vs causation**; the **multiple-testing** trap; why lift needs a base rate.

## Phase B — Success Pattern Mining → `src/pattern_mining/success_patterns.py`
Mine single- and pair-feature rules `{items} → funded` with **lift > 1**, above min support,
significant. Rank by lift × support. **🔧 Real items only** (category/goal/country/timing/title
bands). **Deliverables:** `SuccessPatternMiner`, `success_patterns.json`.

## Phase C — Failure Pattern Mining → `src/pattern_mining/failure_patterns.py`
Symmetric: rules `{items} → failed`, i.e. **lift < 1** for funding (over-represented among
failures — e.g. very-large goal, Technology). Ranked by (1/lift) × support. **Deliverables:**
`FailurePatternMiner`, `failure_patterns.json`.

## Phase D — Funding Pattern Discovery → `src/pattern_mining/funding_patterns.py`
The funding-specific, **multi-feature** view: strongest 2-feature combinations, goal-threshold
patterns (funded vs failed median goals per category), and per-category funding profiles. **🔧
Analyze only real fields** (category, country, goal, timing) — funding *rounds / stage / team /
revenue* deferred. **Deliverables:** `FundingPatternAnalyzer`, `funding_patterns.json`.

## Phase E — Temporal / Timing Patterns 🔧 (reframed from "Growth") → `src/pattern_mining/growth_patterns.py`
Real timing patterns: funding-rate **trend across `launch_year`**, seasonality by
`launch_month` / `launch_quarter` / `launch_weekday` / weekend. **Defer** true growth
(user/revenue/retention — no data). **Deliverables:** `TemporalPatternEngine` +
`growth_patterns.json` (with an explicit `deferred: growth trajectory` note).

## Phase F — Statistical Pattern Ranking → `src/pattern_mining/pattern_ranker.py`
The shared rigor layer. For any pattern compute **support, confidence, lift, chi-square p-value,
effect size (lift / risk-ratio)**; apply **min-support / min-lift** filters and a significance
threshold; flag the **multiple-comparison** count (Bonferroni-adjusted α). Rank. **🔧 Example on
real features** ("`goal_band = very small` → funded: support 13%, confidence 55%, lift 1.36").
**Deliverables:** `PatternRanker`, `ranked_patterns.csv`.

## Phase G — Pattern Explanation Generator → `src/pattern_mining/explanation_generator.py`
Turn a ranked pattern into a business-readable sentence with a **readable-feature map** (reuse
Day-5/6 label style) + a **mandatory honesty caption** (learned historical association, not a
cause). **🔧 Rewrite examples onto real features**: "Historically, Dance campaigns with modest
goals funded at 71% versus a 40% baseline (lift 1.76) — a strong association in the data, not a
guaranteed cause." **Deliverables:** human-readable pattern explanations.

## Phase H — Pattern Knowledge Repository → `src/pattern_mining/repository.py`
Store all discovered patterns (success / failure / funding / temporal) as **JSON** 🔧, organized
by type, with provenance (source split, base rate, thresholds, timestamp) — an extension of the
Venture Knowledge Base. `save` / `load` / `query(by type / min lift)`. **Deliverables:**
`PatternRepository`, `pattern_repository.json`.

## Phase I — Pattern Mining Pipeline → `src/pattern_mining/pipeline.py`
Chain: load train KB → **cluster similar ventures** (`MiniBatchKMeans` on Day-6 embeddings →
per-cluster success rate + dominant real features) → mine success/failure/funding/temporal
patterns → rank (Phase F) → explain (Phase G) → store (Phase H). **🔧 Clustering is the honest
"cluster-based pattern discovery"** (unsupervised, real). **Deliverables:** `PatternMiningPipeline`.

---

## Rigor / validity improvements (the "measure, don't assume" habit) 🔧
- **Base-rate-relative lift** everywhere; never report confidence alone.
- **Significance + multiple-testing**: chi-square/Fisher p-values with a stated Bonferroni α;
  patterns must clear support **and** lift **and** significance.
- **Train-only mining**; optionally **validate top patterns on the untouched test set** (does
  the lift hold out-of-sample? — the honest check that a pattern is real, not overfit).
- **Correlation≠causation caption** on every explanation.

---

## Repository Structure After Day 7
```
src/pattern_mining/  success_patterns.py  failure_patterns.py  funding_patterns.py
                     growth_patterns.py    pattern_ranker.py    explanation_generator.py
                     repository.py         pipeline.py
notebooks/           07_Pattern_Mining.ipynb
```

## Artifacts After Day 7 🔧
```
artifacts/patterns/  success_patterns.json  failure_patterns.json  funding_patterns.json
                     growth_patterns.json   ranked_patterns.csv    pattern_repository.json  (NOT .pkl)
```

## Deliverables Checklist
✅ success_patterns.py · failure_patterns.py · funding_patterns.py · growth_patterns.py (temporal)
· pattern_ranker.py · explanation_generator.py · repository.py · pipeline.py ·
07_Pattern_Mining.ipynb · success / failure / funding / temporal pattern repositories (real
features, ranked, significant, honestly explained)

## Curriculum / Architecture
Layer 6 (Pattern Mining) ✅ · New techniques: association-rule mining, frequent itemsets,
support/confidence/lift, statistical significance, clustering (cluster-based discovery),
knowledge extraction. (Mandatory Week 8 — Evaluation — still ⏳; Day 7 is architecture.)

---

## 🔒 Changelog — corrections from the draft
1. **Scoped to real features.** Founders, TAM, traction, revenue, retention, CAC/LTV, burn,
   PMF, funding rounds, team, business model, growth — **all removed / deferred** (don't exist).
2. **"Growth patterns" reframed to temporal/timing patterns** (real launch-year/seasonality);
   true growth deferred — no time-series data.
3. **Multiple-testing rigor added**: min support + min lift + chi-square significance +
   Bonferroni note (mining many rules manufactures false positives).
4. **Lift measured against the 0.4038 base rate**; confidence never reported alone.
5. **From-scratch focused miner** (no mlxtend); scipy.stats for significance.
6. **Repository is JSON, not pickle** (portable, honest).
7. **Association-not-causation caption mandatory**; Phase-G causal/PMF example forbidden.
8. **No new deps; no faiss/torch/lightgbm** — no OpenMP conflicts in Day 7.

## 💡 Improvements (beyond the draft)
- **Out-of-sample validation** of top patterns on the untouched test set (does the lift hold?).
- **Cluster-based discovery** via MiniBatchKMeans on Day-6 embeddings (per-cluster success rate
  + dominant features) — the honest form of the draft's "cluster similar ventures."
- **Provenance in the repository** (split, base rate, thresholds, timestamp) for reproducibility.
- **Effect size + significance**, not just frequency — a pattern must be big *and* real.

✅ **Spec frozen.** Say "build it" or "build Phase A/B/C…" to implement + teach phase-by-phase
via Framework 1.
