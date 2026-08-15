# Interview Guide

How to explain, defend and extend this project out loud. Every answer is grounded in the code or an
artifact (named in [§11](#11-if-asked-x-open-y)); where the honest answer is "I didn't do that", the guide
says so and gives you the words. Companion documents: [PORTFOLIO_AND_RESUME.md](PORTFOLIO_AND_RESUME.md)
(what you claim) and [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) (the numbers and weaknesses).

**Format.** Each question has a **one-sentence answer** to say first, then **if they dig** for the detail.
Stop after the one sentence and let them ask — long unprompted answers invite the question you don't want.

---

## 0. The project in 60 seconds

"It analyzes a Kickstarter campaign. You submit title, category, goal, country, currency and dates; it
returns a calibrated funding probability, a SHAP explanation, similar past campaigns, the changes that would
help most, and a committee-style investor verdict. The model is a tuned, calibrated LightGBM at 0.78 ROC-AUC
on a held-out test set; around it are thirteen layers sharing one warm process, a FastAPI backend and a
React dashboard. The design principle is honesty: the API rejects fields the data can't support, and
scores with no data basis — founder strength, market fit — are shown as 'deferred', not invented."

## 1. Numbers to know cold

| | |
|---|---|
| Data | 378,661 → **331,464** decided campaigns, **2009–2017** |
| Split | 265,171 train · 53,035 validation (carved from train) · **66,293 test (scored once)** |
| Features | **865** = 800 TF-IDF + 65 tabular (866 before selection) |
| Model | tuned **LightGBM + sigmoid calibration**, threshold **0.329** |
| Test | ROC-AUC **0.776** · PR-AUC **0.688** · F1 **0.666** · recall **0.813** · precision **0.565** · accuracy 0.671 · Brier 0.187 |
| Baseline | Logistic Regression: PR-AUC 0.626, ROC-AUC 0.727 (validation) |
| Chance | 40.4% succeed → "always failed" = 59.6% accuracy |
| Retrieval | `all-MiniLM-L6-v2`, 384-d, exact FAISS, ~265K vectors, 7× category lift |
| Latency (p50) | `/predict` 14 ms · `/explain` 37 ms · `/retrieve` 20 ms · `/analyze` 3.2 s |
| Scale | 13 layers · 6 agents · 95 tests (53 in CI) · 12 endpoints |

---

## 2. System design

**Why "multimodal" learning?**
*One sentence:* The title is free text and the rest is structured data, and each carries different signal, so
I fuse both into one feature matrix.
*If they dig:* "Multimodal" here means **text + tabular** — not images or audio. Words like *thesis* or
*album* and title shape (`word_count`, `uppercase_ratio`) add information the category and goal don't.
Individually the words are weak (the best word ranks 25th by SHAP), which is why I don't oversell it.

**Why feature fusion — and how?**
*One sentence:* A single model needs a single matrix, and sparse concatenation keeps it cheap.
*If they dig:* Numeric branch (log → outlier cap → `StandardScaler`), categorical branch (one-hot +
frequency), text branch (TF-IDF) are stacked with `scipy.sparse.hstack` → 866 columns, then feature selection
(variance, redundancy, mutual information) → 865. It's one fit/transform object, so "fit on train only" is
guaranteed by construction rather than by discipline.

**Why TF-IDF, and why 800 features?**
*One sentence:* Titles are short, so a small interpretable vocabulary captures most of the signal and lets
SHAP name the word.
*If they dig:* Titles average ~5.6 words; 800 terms (1–2 grams) keep the head of the distribution, and a
stale config value of 2,500 would mostly add near-empty columns. Bigrams catch phrases like "board game". The
vocabulary and IDF weights are learned on train only. *(I did not test SBERT vectors as classifier features —
embeddings are used for retrieval.)*

**Why semantic retrieval?**
*One sentence:* To show a founder real campaigns like theirs — evidence, not a prediction.
*If they dig:* TF-IDF matches words; SBERT matches meaning ("board game" ≈ "tabletop game"). I use an
**exact** FAISS `IndexFlatIP` over L2-normalized vectors, so cosine = dot product; at 265K × 384 it is
sub-millisecond, so approximate indexes would only lose recall. Quality is measured with proxies (same-category
neighbours 7.1× the base rate) because there is no ground truth for "similar", so I can't claim Precision@K.
Retrieval predicts outcome only weakly (ROC-AUC 0.69–0.71), so it does **not** feed the model.

**Why SHAP?**
*One sentence:* It gives additive, per-campaign explanations that reconcile exactly to the model's output.
*If they dig:* TreeSHAP for the LightGBM model, interventional mode (the default fast path was numerically
unreliable for this model); an automated test checks `base_value + Σ contributions = margin` to 3e-8.
It explains the **uncalibrated log-odds margin** — calibration is a monotone transform, so driver ordering is
unaffected. The UI shows a bar chart and a waterfall. Caveat I volunteer: it explains the *model*, not the world.

**Why FastAPI?**
*One sentence:* Pydantic validation enforces the honesty contract at the boundary, and the OpenAPI docs come free.
*If they dig:* `CampaignRequest` uses `extra="forbid"`, so an invented field is a `422`. A lifespan handler
loads the registry once; routes get it through dependency injection (503 until ready); services are plain
functions, so they're testable without HTTP. Typed error handlers map to 422/503/500 and a request-id
middleware tags every log line.

**Why Docker?**
*One sentence:* To pin a heavy, fragile environment — and I'll tell you up front I wrote the config but never
built the image.
*If they dig:* Python 3.14, CPU-only torch, `OMP_NUM_THREADS=1`, and NLTK/SBERT data fetched at build time are
easy to get wrong; a multi-stage image with artifacts **mounted read-only** keeps it lean (the image is still
multi-GB). There's no Docker on my machine, so it is reviewed configuration, not a proven image.

**Why one process instead of microservices?**
*One sentence:* The layers share a ~1 GB warm stack and FAISS is an in-process library; splitting would reload
the models per service.
*If they dig:* Layers compose — layer 13 holds 11 and 12, which hold the aggregator, which shares layer 8's
counterfactual pipeline, which holds the model. One model object serves everything. I'd split only for
independent scaling or team boundaries.

**Why deterministic, rule-based agents and no LLM?**
*One sentence:* Reproducibility and honesty — every output traces to a real signal, with no external service
and no secrets.
*If they dig:* The six agents (market, founder, financial, competition, risk, investment) read the same
aggregated signals; a confidence-weighted vote with a risk-agent veto produces the verdict. They are *correlated
views of one signal set*, so confidence isn't inflated by agent count, and the competition agent is
down-weighted (0.8) for overlap. An LLM could improve the prose; it could also invent facts, which this
project is designed to prevent.

---

## 3. Machine learning

**Why start with Logistic Regression?**
*One sentence:* It's the interpretable floor that tells me whether boosting is earning its complexity.
*If they dig:* PR-AUC 0.626 (ROC 0.727). L1, L2 and ElasticNet were indistinguishable, so regularization type
didn't matter. Boosting added **+0.062 PR-AUC** — worth it, but not dramatic, which is itself a finding.

**Why XGBoost? Why LightGBM?**
*One sentence:* Gradient boosting is the strong default for tabular data; I benchmarked both and chose
LightGBM on practicality.
*If they dig:* LightGBM, XGBoost and HistGradientBoosting finished within 0.005 PR-AUC of each other —
inside what one split can resolve — so I picked LightGBM: **2.9 s to train vs 30.2 s** for HistGB, which made
a 75-fit randomized search cheap. Tuned: 127 leaves, learning rate 0.05, 300 trees, subsample 0.7, colsample
0.8, `min_child_samples` 50, light L1/L2. Tuning moved validation PR-AUC 0.6798 → 0.6876. Random Forest and
Extra Trees were worse and 40–65× slower. *(I did not run repeated seeds, so I don't claim the boosting
ranking is statistically established.)*

**Why embeddings?** (see retrieval above) Dense semantic vectors for similarity search; 384-d MiniLM because
it is small, fast and good enough for short titles; L2-normalized and persisted as float32 with a hash, not as
a pickled torch model.

**Why sparse matrices?**
*One sentence:* The TF-IDF block is >99% zeros, so dense storage would waste memory and time.
*If they dig:* CSR format; the fused train matrix is ~14 MB compressed on disk. LightGBM consumes sparse input
directly.

**Why `StandardScaler`? Do trees need it?**
*One sentence:* Linear and distance-based models treat raw magnitude as importance, so numerics get put on one
scale — trees don't need it.
*If they dig:* `goal` after log1p lives around 4.6–12.6, `launch_month` on 1–12, `goal_per_day` elsewhere.
Trees split on thresholds and are invariant to monotone rescaling, so scaling is a toggle: the same pipeline
serves the Logistic baseline (needs it) and the boosted model (doesn't care). Fit on train only.

**Why `OneHotEncoder`? And why frequency encoding for `category`?**
*One sentence:* One-hot for low-cardinality columns, frequency encoding for the 159-level `category`.
*If they dig:* `main_category` (15), `currency` (14), `country` (22) become indicators with
`handle_unknown="ignore"`. One-hotting 159 sub-categories would add 159 sparse columns, so each level maps to
the fraction of training rows it covers. `category_freq` ended up a top-3 feature — it encodes how *common* a
category is; I'd flag that it's a popularity proxy, not a causal effect.

**Why calibration?**
*One sentence:* Scenarios, readiness and scores do arithmetic on the probability, so it needs to mean what it says.
*If they dig:* Sigmoid vs isotonic tied (Brier 0.1872 vs 0.1872), and calibration barely moved the raw model
(0.1875) — it was already close. I kept sigmoid as the simpler method. Brier is 22% better than predicting the
base rate.

**Why PR-AUC over accuracy?**
*One sentence:* Accuracy hides the trade-off, and the classes are mildly imbalanced (40/60).
*If they dig:* The always-failed baseline gets 59.6% accuracy, so 67% isn't impressive on its own. PR-AUC
(0.688 vs a 0.404 random line) measures ranking quality across thresholds, which is what I tuned on.

**Why threshold 0.329?**
*One sentence:* It maximises F1 on validation, trading precision for recall.
*If they dig:* At 0.5, F1 0.616 and recall 0.576; at 0.329, F1 0.668 and recall 0.817. Precision is 0.565 on test
— a deliberate choice that favours *finding* funded campaigns. A different audience (an investor screen) should
raise it; the API returns the threshold so a client can.

**How did you prevent leakage?**
*One sentence:* Outcome columns are excluded and every learned statistic is fit on train only.
*If they dig:* Pledged amount and backers (which reveal the outcome) are dropped. The feature pipeline —
scaler, encoders, TF-IDF vocabulary and IDF, feature selection — is one object fit on train; the knowledge
base for retrieval is built from the train split, so a query's own outcome is never in the index. The test set
was scored once, after all selection and tuning.

**Why feature selection by mutual information?**
*One sentence:* It works for sparse and one-hot columns, where correlation doesn't.
*If they dig:* Variance threshold drops constant columns; a redundancy step keeps the higher-MI column of a
correlated pair (`goal` ↔ `usd_goal_real` correlate at 0.957); MI ranks the rest. All fit on train; the mask is
applied to everything else.

**How do you know it isn't overfit?** Validation and test PR-AUC are within 0.0001 (selection didn't overfit
the validation set); train–test ROC-AUC gap is 0.040 (mild overfitting, inside my 0.10 limit). The bigger
generalization risk is *temporal*, not statistical — see §6.

---

## 4. Architecture — every layer from memory

Mnemonic for the 13 layers — **D**ata → **M**emory → **S**earch → **P**redict → **E**xplain → **P**atterns →
**W**hat-if → **O**bjections → **E**volve → **S**imulate → **R**ecommend → **V**ote → **R**eport.

| # | Say | Package |
|--:|---|---|
| 1 | **Data & features** — clean, encode, scale, TF-IDF, fuse | `preprocessing`, `features` |
| 2 | **Knowledge base** — train campaigns as metadata + text records | `knowledge_base` |
| 3 | **Search** — SBERT embeddings + exact FAISS | `retrieval` |
| 4 | **Predict** — calibrated LightGBM, threshold, scores | `models`, `training` |
| 5 | **Explain** — SHAP drivers, coefficients | `explainability` |
| 6 | **Patterns** — association rules with significance | `pattern_mining` |
| 7 | **What-if** — counterfactuals: single changes ranked by lift | `counterfactual` |
| 8 | **Objections** — investor objections, SWOT, category trend | `copilot` |
| 9 | **Evolve** — readiness stage and milestone roadmap | `evolution` |
| 10 | **Simulate** — scenario combinations over real levers | `scenario` |
| 11 | **Recommend** — aggregate 4–10 into priorities, strategy, confidence | `recommendation` |
| 12 | **Vote** — six agents + consensus | `agents` |
| 13 | **Report** — founder, investor, memo, executive | `reporting` |

**The one-breath data flow.** "A campaign enters the feature pipeline, the model scores it, SHAP and the
counterfactual engine explain and perturb it, an aggregator collects everything once, the recommendation
engine and the six agents read that single profile, and the report layer renders it — with retrieval running
alongside as evidence."

**Facts that show you really know it**
- Layers **compose, not recompute**: one model object serves all 13.
- `/recommend` and `/analyze` run on a **FAISS-free** stack; only `/retrieve` uses FAISS and torch. "Similar
  startups" inside recommendations is a same-category cohort.
- **Load order matters**: LightGBM stack first, FAISS/torch last, with `OMP_NUM_THREADS=1` set before imports,
  or the two OpenMP runtimes collide.
- **Counterfactuals** = candidate changes (goal → the category's funded median, duration, launch weekday,
  title number) re-scored by the model and ranked by probability gain; gains under 0.005 are dropped.
- **Patterns** are mined offline (support, confidence, lift, Bonferroni significance); serving reads their
  outputs but nothing re-mines at request time.
- **Reports are bundle-once**: one `ReportBundle`, then pure generators for all four reports and the copilot.

---

## 5. Engineering

**Folder structure.** `src/` (one package per layer + `api`, `services`, `core`, `mlops`, `utils`),
`frontend/`, `tests/`, `notebooks/` (executed per-day verification), `artifacts/` (trained objects),
`docs/`, `deployment/`, `scripts/`, `configs/`. Each layer has a `pipeline.py` entry point and a
`serialization.py`; the API reaches them only through `services/`.

**Deployment.** One container; artifacts mounted read-only; `.env` for port, workers (keep at 1 — each loads
~1 GB), log level, CORS allow-list; Compose with optional nginx and MLflow-UI profiles. Config is written and
reviewed but **unbuilt**.

**APIs.** Nine analysis endpoints + `/health`, `/live`, `/ready`. Strict request schema; responses carry an
`honesty_note` and/or `deferred` block; documentation generated from the live OpenAPI schema and real
responses. No authentication, no rate limiting — said plainly.

**Monitoring.** Probes only: `/live` (always 200), `/ready` (200 once loaded, else 503), `/health` (component
status + CPU/memory/disk via `psutil`). No metrics stack, tracing or alerting.

**Logging.** One logger namespace (`avip.*`), a request-id middleware that tags every line (`[ae0c119a] POST
/predict -> 200 (12ms)`) and returns an `X-Request-ID` header; level and file sink from env. Tracebacks stay
server-side; clients get a generic 500 body.

**MLOps.** MLflow with a local SQLite backend (no server); a JSON model registry with status history
(*Production* LightGBM, *Archived* Logistic baseline); a version manager that fingerprints a run by dataset
hash, split, seed, git commit and environment; and **gated promotion** — a challenger (XGBoost, F1 0.6576)
did not beat the incumbent, so the incumbent was kept. Large artifacts are referenced by hash instead of
copied (776.9 MB saved).

**Testing.** 95 tests: unit tests for cleaning, types, duplicates, missing values and integrity; model
validation against thresholds (7 checks); retrieval proxy checks; SHAP additivity; API contracts and latency
budgets; end to end. CI runs the 53 that need no artifacts; the rest run locally.

**Frontend.** React 19 + TypeScript (strict), types mirror the API schemas so a contract change is a compile
error; the UI renders `honesty_note` and `deferred` instead of hiding them; checked in headless Chrome at
desktop and 320–390 px.

---

## 6. Hard questions — and honest answers

**"How accurate is it?"** "ROC-AUC 0.78, PR-AUC 0.69, accuracy 67% versus 59.6% for guessing 'failed'.
That's real signal but a modest ceiling — pre-launch metadata only partly determines the outcome."

**"Why is precision only 0.56?"** "I tuned the threshold for recall — F1-optimal — so it catches 81% of funded
campaigns at the cost of flagging many failures. For a stricter use you raise the threshold; the API returns it."

**"Does it work on today's campaigns?"** "Probably less well than the numbers suggest. The data ends in 2017,
my split is random rather than chronological, and `launch_year` is a top-5 feature while the success rate
drifted from roughly 48% to 32–43%. A time-based evaluation is the first thing I'd add." *(This is your
strongest answer — you found it yourself.)*

**"Why a random split?"** "It was the standard stratified split from the curriculum and it kept the class
balance, but for this data a chronological split would be more honest. I documented that rather than hide it."

**"Is there a neural network?"** "No. I benchmarked tree ensembles and a linear baseline; the roadmap listed a
neural net but I never trained one, and I don't claim it."

**"Is it deployed?"** "No. There's a Dockerfile, Compose and deployment guide, but I haven't built the image —
no Docker on my machine. It runs locally and I verified the API and UI."

**"Is the multi-agent system AI?"** "They're deterministic, rule-based agents over real signals with a weighted
vote — decision logic, not language models. I'd call it a transparent ensemble of heuristics."

**"You have a FounderAgent but no founder data."** "Right — it doesn't evaluate a person. It measures
*creator execution*: what fraction of the reachable funding probability the creator has already captured
(base probability ÷ optimized probability), based on controllable choices like goal and title. Founder
experience, team and track record are deferred because the data has none."

**"Are counterfactuals causal?"** "No. They say what the *model* would predict if one feature changed; they
don't say what would happen to a real campaign. The UI says so."

**"Why does `uppercase_ratio` hurt a campaign?"** "I don't think it does. It's a correlation the model found in
2009–2017 titles. SHAP explains the model, not the world — which is why I label drivers as associations."

**"Your founder page says 'Proceed with Caution' but the investor page says 'Do Not Invest'."** "Two
mechanisms: a readiness band from the recommendation engine, and a six-agent committee vote. They answer
different questions and the UI doesn't reconcile them yet — it's on my list of known rough edges."

**"Could this scale?"** "Not as built — one process, ~1 GB per worker, no load test. I'd scale horizontally with
more containers, put retrieval behind its own service if it became the bottleneck, and add real metrics."

**"What about security and privacy?"** "No authentication yet (the frontend is auth-ready, not auth-built), no
rate limiting. The data is public Kickstarter data, there are no secrets, and CORS is deny-by-default."

**"Why not deep learning or an LLM?"** "For ~330K rows of short titles and tabular fields, boosted trees are
the strong default and stay interpretable. An LLM could polish the report text, but it can invent facts —
exactly what I designed the system not to do."

**"What would you do differently?"** "(1) Chronological evaluation first. (2) Bootstrap confidence intervals
and per-segment metrics. (3) Reconcile the two verdicts. (4) Build and test the Docker image. (5) Richer, more
recent data — founder, team and market signals — which would also unlock the deferred scores."

**"What's the biggest limitation?"** "The data. It has title, category, goal and dates, and nothing about the
founders, product or market — so the ceiling is the dataset, not the algorithm."

---

## 7. Behavioral stories

Three ready-made STAR stories — the honesty boundary, the bug found by rendering the UI, and finding the
2017-data weakness myself — are in [PORTFOLIO_AND_RESUME.md §6](PORTFOLIO_AND_RESUME.md#short-stories-for-tell-me-about-a-challenge).

## 8. Whiteboard prompts

1. **Draw the serving architecture** (request → routes → services → registry → pipelines; load order).
2. **Draw the feature pipeline** (three branches → hstack → selection) and mark where "fit on train only" applies.
3. **Explain leakage** with a concrete example from this project (a fitted scaler or TF-IDF vocabulary
   touching the test set; pledged amount as a feature).
4. **Sketch a reliability diagram** and say what sigmoid vs isotonic would change.
5. **Design the chronological evaluation:** train 2009–2015, validate 2016, test 2017; compare to the random
   split; check `launch_year` ablation; report drift in base rate.

## 9. Rapid-fire (one line each)

- *Target?* successful = 1, failed = 0; live/cancelled/etc. excluded.
- *Base rate?* 40.4% successful.
- *Metric you optimised?* PR-AUC (average precision) in CV; threshold by F1.
- *Calibration method?* Sigmoid (Platt); isotonic tied.
- *Threshold?* 0.329.
- *Top feature?* `usd_goal_real`, then `campaign_duration`, `category_freq`, `launch_year`.
- *Embedding model?* `all-MiniLM-L6-v2`, 384-d.
- *Index?* FAISS `IndexFlatIP`, exact.
- *Why exact?* sub-millisecond at 265K; no recall loss.
- *Explainer?* TreeSHAP, interventional; additivity 3e-8.
- *Registry status?* LightGBM *Production*, Logistic *Archived*.
- *Workers?* 1 — scale by containers.
- *Auth?* None; frontend auth-ready.
- *Tests?* 95; CI runs 53.
- *Deferred?* Business-Model, Founder-Strength, Scalability, Market-Fit and Growth-Potential scores, plus market size (TAM/SAM). *(A category-level market-opportunity label from Kickstarter history is computed.)*
- *Where's pattern mining?* Offline; outputs feed counterfactuals and category analysis.
- *PDF export?* Browser print — no server PDF.
- *Neural net?* Not built.

## 10. Practice plan

1. **Session 1 (30 min):** say §0 and the §4 layer table aloud from memory, twice.
2. **Session 2 (30 min):** answer §3 questions with only the one-sentence answers; then add detail.
3. **Session 3 (30 min):** have someone ask the §6 questions in random order; keep each answer under 20 seconds.

## 11. If asked X, open Y

| If asked about… | Open |
|---|---|
| any metric, the confusion matrix, weaknesses | [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) |
| the architecture or load order | [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) |
| an endpoint or error | [API_DOCUMENTATION.md](API_DOCUMENTATION.md) |
| "is that claim true?" | [day20_phaseA_release_audit.md](day20_phaseA_release_audit.md) (evidence ledger) |
| TF-IDF sizing | `src/features/tfidf_vectorizer.py` |
| encoding choices | `src/preprocessing/encoding.py` |
| feature selection | `src/features/feature_selection.py` |
| calibration / threshold | `artifacts/evaluations/calibration.json` |
| model comparison | `artifacts/evaluations/benchmark.json` |
| MLOps | [day16_mlops_reference.md](day16_mlops_reference.md) |
| retrieval | [day15_retrieval_reference.md](day15_retrieval_reference.md) |
| the demo | [DEMO_GUIDE.md](DEMO_GUIDE.md) |
