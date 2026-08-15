# Research Roadmap — Version 2.0

> **Everything in this document is *proposed, not built*.** None of it is implemented, and none of it should
> be described as a feature of V1. Each item states the **data it needs**, how **feasible** it is with what
> exists today, what it would **unlock**, and the **first experiment** that would tell you whether it is worth
> doing. Expected results are deliberately not promised.

**The organising idea.** V1's ceiling is set by its data, not its algorithms: the dataset has a title,
category, goal, country, currency and two dates — nothing about founders, products or markets. So this roadmap
leads with *data and evaluation*, and treats fashionable methods (graphs, vision, RL) as downstream of having
something for them to learn from.

---

## 1. What the data can and cannot support today

Checked directly against `data/raw/ks-projects-201801.csv` (378,661 rows, 15 columns):

| Column group | Present | Missing — which blocks |
|---|---|---|
| Campaign facts | title, category (159), main category (15), currency, country, goal, launch, deadline | product description / blurb → richer text, pitch analysis |
| Outcome | final state, pledged, backers (+ USD-converted) | **daily pledge history** → trajectory forecasting |
| Entities | — | **creator ids, backer ids, investor ids** → graph / network methods |
| Media | — | images, video, pitch decks → vision-language models |
| Money & team | — | revenue, retention, burn, team, founder background → the deferred scores and market size |
| Period | 2009-04 to 2017-12 | anything recent → current-day validity |

Also note a **labeling choice**: only `successful` and `failed` (331,675) are used. 38,779 `canceled`,
1,846 `suspended`, 2,799 `live` and 3,562 `undefined` campaigns are excluded. Cancellation is often a
disguised failure, so its exclusion may bias the model — see item 0.6.

---

## 2. Tier 0 — repair V1's known weaknesses (no new data)

These come straight from [PERFORMANCE_REPORT.md §12](PERFORMANCE_REPORT.md#12-weaknesses-and-limits). They are
the cheapest, highest-credibility work, and they should come **before** any new method.

| # | Proposal | Why | First experiment | Decision rule |
|--:|---|---|---|---|
| 0.1 | **Chronological evaluation** | Random split mixes years; `launch_year` is a top-5 feature; success rate drifted ~48% → 32–43% | Train ≤2015, validate 2016, test 2017; compare with the random-split numbers | If test PR-AUC falls materially (say >0.03), report the honest number and test dropping or re-encoding `launch_year` |
| 0.2 | **Uncertainty on every metric** | Single split, no intervals; boosted-tree differences are ≤0.005 | Bootstrap 95% CIs for ROC/PR-AUC; repeat with 5 seeds | Only claim a model ranking whose intervals separate |
| 0.3 | **Per-segment performance** | Aggregates can hide failure on a category, country or goal band | Metrics and calibration by main category, goal quintile, year | Flag segments materially worse than overall; consider segment-aware thresholds |
| 0.4 | **Prediction intervals** | A point probability overstates certainty | Conformal prediction on top of the calibrated model | Report coverage on held-out data |
| 0.5 | **Threshold per audience** | 0.329 is recall-oriented; an investor screen wants precision | Plot precision/recall across thresholds; propose a founder and an investor operating point | Return both from the API |
| 0.6 | **Re-examine excluded states** | 38,779 canceled campaigns are dropped | Add `canceled` as failure; compare metrics and drivers; report both | Keep only if it improves honesty of the estimate |
| 0.7 | **Embeddings as classifier features** | SBERT is used for retrieval only; never tested for prediction | Add 384-d vectors (or a PCA of them) to the fused matrix; ablate | Keep only if the chronological test improves |
| 0.8 | **Backtest the committee** | The six-agent verdict was never scored against real outcomes | Run the aggregator-only path (no FAISS) on a held-out sample; compare verdict with actual outcome | If the verdict adds nothing over the model probability, say so |
| 0.9 | **Reconcile the two verdicts** | The founder page ("Proceed with Caution") and investor page ("Do Not Invest") come from different mechanisms | Define one verdict, or label both clearly | One consistent story across pages |
| 0.10 | **Build and test the container; load test; add auth and rate limiting** | Docker is unbuilt; no concurrency test; no authentication | `docker build`, `locust`/`k6` run, JWT stub → real | Documented capacity and a proven image |
| 0.11 | **Drift monitoring** | No monitoring beyond probes | Track input and score distributions over time | Alert on distribution shift |

---

## 3. Tier 1 — new data to unlock the deferred scores

Five scores plus market size are "deferred — no data basis". Each needs a specific kind of data; none can be conjured from
Kickstarter titles. **This is the roadmap's centrepiece**, because it changes V1's ceiling rather than polishing it.

| Deferred score | Data it needs | Plausible modeling | Evaluation it requires |
|---|---|---|---|
| **Founder Strength** | founder/team history: prior ventures, exits, years of experience, role mix | gradient-boosted model on team features; careful target (e.g. later funding) | out-of-time split; compare with the V1 model's lift |
| **Business-Model** | revenue model, pricing, unit economics, customer segments | structured features + text classification of the business description | label quality is the hard part — who decides "good"? |
| **Scalability** | traction over time: users/revenue growth, cost structure | growth-curve features, time-series models | needs longitudinal data, ≥ several periods |
| **Market-Fit** | retention, NPS, customer interviews, conversion | cohort/retention features | only meaningful with post-launch behavioural data |
| **Growth Potential** | revenue/user trajectory, funding history | forecasting models (§4) | out-of-time forecasting error |
| **Market size / opportunity** *(a category-level opportunity label from Kickstarter history already exists; market size does not)* | market size, competitors, category growth (TAM/SAM/CAGR) | external market datasets joined on category | provenance and freshness of every external figure |

**Candidate sources — availability and terms unverified; check licences before use**

| Kind | Examples | Notes |
|---|---|---|
| Newer Kickstarter data | public scraped dumps (e.g. Web Robots) | extends the period beyond 2017 and fixes drift, but still lacks founders and finances |
| Funding rounds | SEC EDGAR Form D filings (public), Crunchbase / PitchBook (licensed) | different population: venture-backed companies, not crowdfunding |
| Market data | industry reports, government statistics | joins at category level only |
| Campaign text/media | scraped campaign pages | terms-of-service and privacy review needed |

**Caution.** Switching from Kickstarter to venture data changes the *task* (a different outcome, population
and base rate). It is a new project phase, not a patch — and it should keep the honesty-boundary design: if a
score still cannot be supported, it stays deferred.

---

## 4. Tier 2 — the roadmap's research extensions, with honest feasibility

| Extension | Needs | Feasibility with today's data | What it would add | First experiment |
|---|---|---|---|---|
| **Temporal forecasting** (predict the funding trajectory, not just the outcome) | daily pledge snapshots per campaign | ❌ only final totals exist | early-warning: "campaign at day 5 is on/off track" | scrape or obtain daily pledge series for a few thousand campaigns; baseline = linear extrapolation |
| **Knowledge graph** of the ecosystem | entities and links: creators, categories, backers, investors, companies | ❌ no creator/backer ids | relationship features, queryable evidence | start from category/country/currency relations (available) and measure whether graph features beat one-hot |
| **Graph neural networks** | a real graph with enough nodes/edges | ❌ blocked by the same missing entities | collective effects (creator track record, backer overlap) | needs the graph first; compare against a feature-engineered baseline using creator history |
| **GraphRAG** | a knowledge graph + text corpora | ❌ blocked by the graph | grounded Q&A over relationships | only after the graph exists; today's retrieval already covers similar-campaign evidence |
| **Vision-language models for pitch decks** | decks / campaign images and video | ❌ none in the data | judge visual and narrative quality — a real missing signal | collect a small labelled sample of campaign pages; test whether image features add lift over V1 |
| **Reinforcement learning for investment strategy** | sequential decisions, costs and *returns* | ❌ no returns; no logged investor decisions | portfolio-level strategy | a simulator only: define a toy portfolio problem; **do not present as real investing advice** |
| **LLM fine-tuning / generation** | report text pairs, preference data | ⚠️ feasible but risky | fluent reports and Q&A | constrain generation to the evidence bundle; test faithfulness (every figure traceable) and refusal on out-of-scope questions |
| **Improved multi-agent collaboration** | outcome labels to learn from | ✅ feasible now | learned agent weights (stacking), disagreement as uncertainty | backtest (0.8), then fit weights on a train fold and compare to the hand-set ones |
| **Online / continual learning** | a stream of new labelled campaigns | ⚠️ needs a feed; labels arrive after 30–60 days | a model that stays current | simulate on the 2009–2017 timeline: retrain monthly vs frozen, measure drift recovery |

---

## 5. Suggested order

| Phase | Scope | Why this order |
|---|---|---|
| **V1.1** (days to weeks) | Tier 0: 0.1–0.9 | Fixes what an informed reviewer will question first; needs no new data |
| **V1.2** | 0.10–0.11 + newer Kickstarter data | Makes the system operable and current |
| **V2.0** | Tier 1 (one deferred score at a time) + backtested multi-agent weights | Raises the ceiling with real signal |
| **V2.x research** | Tier 2 items that have data by then: trajectories, online learning, constrained LLM text | Only once there is something for them to learn from |

**The smallest high-value next step:** item 0.1, the chronological evaluation. It needs no new data, answers
the biggest open question about V1's reported numbers, and its result decides how to treat `launch_year`.

---

## 6. Standards before anything is claimed

A V2 result counts only if it has:
1. a **chronological** train/validation/test split;
2. **confidence intervals** (bootstrap or repeated seeds), not a single number;
3. an **ablation** showing the new component adds lift over the simpler baseline;
4. **no leakage** — every learned statistic fit on training data only;
5. a stated **negative result** when it doesn't help — publish those too;
6. its limits written next to its numbers, as in the V1 [Performance Report](PERFORMANCE_REPORT.md).

## 7. Risks and ethics

- **Causality.** Predictive features are associations; counterfactuals describe the model. Don't let new
  methods blur this — it is the line between decision support and bad advice.
- **Fairness.** Country and currency are inputs. Any new data (founder background especially) can encode
  bias; evaluate by segment before shipping.
- **Misuse.** Scores about real people's projects can mislead funding decisions. Keep uncertainty and
  deferral visible.
- **Data provenance.** Scraped or licensed data brings terms-of-service, privacy and licence obligations; check them first.
- **Honesty boundary.** Every extension must keep the rule: if the data can't support a value, return it as
  deferred, not invented.

## 8. How V2 would show up in the architecture

| V1 layer | Likely change |
|---|---|
| 1 Data & features | new sources, chronological split, embedding features |
| 4 Prediction | intervals (conformal), segment-aware thresholds, possibly a retrained model on newer data |
| 12 Multi-agent | backtested, learned agent weights |
| 13 Reports | optional constrained-generation layer with evidence citations |
| `core` / serving | auth, rate limiting, drift monitoring, load-tested container |
| deferred scores | each one moves from "deferred" to "computed" only when its data arrives |
