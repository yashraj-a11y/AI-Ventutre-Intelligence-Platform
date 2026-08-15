# Portfolio and Resume Kit

Ready-to-paste text for your resume, GitHub, LinkedIn and conversations. **Every number and claim below is
backed by a file in this repository** (the evidence is named beside it, and the master list is the
[release audit ledger](day20_phaseA_release_audit.md)). Before you paste anything, read
[§9 Before you send](#9-before-you-send).

**Your links** (filled in below)

| | |
|---|---|
| Repository | <https://github.com/yashraj-a11y/AI-Ventutre-Intelligence-Platform> |
| GitHub profile | <https://github.com/yashraj-a11y/> |
| LinkedIn | <https://www.linkedin.com/in/yash-raj-a54a67323/> |

**Still to fill:** `[demo video URL]` and `[your name]` (not inferred from a URL — write it as you want it shown).

---

## 1. The rule

An interviewer will pick one line from your resume and ask "how?" — so each line here is something you can
open a file to prove. Where the original roadmap's draft bullets claimed more than the code does, this kit
rewrites them ([§7](#7-draft-bullets-that-were-rewritten-and-why)).

## 2. Project snapshot (verified)

| Fact | Value | Evidence |
|---|---|---|
| Data | 378,661 Kickstarter campaigns → **331,464** decided and cleaned, launched 2009–2017 | `data_validation.json`, raw CSV |
| Features | **865** = 800 title TF-IDF + 65 tabular, fused with `scipy.sparse.hstack` | `day5_global_importance_report.md` |
| Models benchmarked | Logistic Regression (L1/L2/ElasticNet) baseline vs **5 tree ensembles** (LightGBM, XGBoost, HistGradientBoosting, Random Forest, Extra Trees) | `benchmark.json`, `logreg_results.json` |
| Deployed model | Tuned **LightGBM + sigmoid calibration**, threshold 0.329 (F1-optimal), 75-fit randomized search | `model_metadata.json`, `best_config.json` |
| Test result (66,293 campaigns, scored once) | **ROC-AUC 0.776 · PR-AUC 0.688 · F1 0.666 · recall 0.813** · Brier 0.187 | `model_validation.json` |
| vs baseline | Logistic Regression PR-AUC 0.626 → **+0.062 PR-AUC**, +0.050 ROC-AUC (validation) | `benchmark.json` |
| Calibration | Brier **22% better** than a base-rate predictor *(derived: 0.1873 vs 0.2407)* | `calibration.json` |
| Generalization | validation and test PR-AUC agree to **0.0001** | `model_validation.json` |
| Explainability | SHAP additivity error **3.1e-08**, enforced by an automated check | `explainability_validation.json` |
| Retrieval | SBERT `all-MiniLM-L6-v2`, exact FAISS index, **~265k** vectors; same-category neighbours **7.1×** the base rate; ~20 ms p50 through the API | `retrieval_validation.json`, `benchmark.json` |
| Architecture | **13 layers**, one warm process; six-agent committee with a risk veto | [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) |
| API | FastAPI: nine analysis endpoints + three health probes; request schema rejects unknown fields; `/predict` p50 **14 ms**, `/analyze` p50 **3.2 s** | [API_DOCUMENTATION.md](API_DOCUMENTATION.md), `benchmark.json` |
| Quality | **95 tests** (53 run in CI), `ruff`, MLflow tracking + model registry | `tests/`, `.github/workflows/ci.yml` |
| Frontend | React 19 + TypeScript (strict) + Vite + Tailwind; ~1,500 lines; verified in headless Chrome on desktop and 320–390 px mobile | `frontend/`, `docs/assets/screenshots/` |
| Size | ~20,000 lines of Python in 20 packages, 19 notebooks, 60+ docs | repository |

---

## 3. Resume

**Entry header**

> **AI Venture Intelligence Platform** — end-to-end ML system · *Python, scikit-learn, LightGBM, SHAP, SBERT, FAISS, MLflow, FastAPI, React, TypeScript* · `https://github.com/yashraj-a11y/AI-Ventutre-Intelligence-Platform`

### 3a. The core four (use these if you have room for one project entry)

- Built an end-to-end venture-intelligence platform on **331K Kickstarter campaigns** — calibrated funding prediction, SHAP explanations, semantic retrieval, counterfactual and scenario analysis, and a six-agent due-diligence committee — served by a **FastAPI** backend and a **React/TypeScript** dashboard for founders and investors.
- Engineered a leakage-safe pipeline (fit-on-train-only; **800 TF-IDF + 65 tabular features fused with sparse `hstack`**) and benchmarked **five tree ensembles against a regularized Logistic Regression baseline**; a tuned, sigmoid-calibrated **LightGBM** reached **test ROC-AUC 0.776 / PR-AUC 0.688** (**+0.06 PR-AUC** over the baseline on validation), with the test set scored once.
- Implemented explainability and decision-support layers: **SHAP** (additivity verified to 3e-8), **SBERT + FAISS** exact retrieval over 265K campaigns (**7× category lift**), counterfactual search, a scenario simulator, and rule-based investor-objection generation.
- Shipped a typed **FastAPI** service (strict request schema, health probes, **14 ms** median prediction latency) with **MLflow** tracking, **95 automated tests** (53 in a GitHub Actions fast lane), and a written evaluation report that states the model's limits.

### 3b. One-line version (tight resumes)

- Built an end-to-end ML platform on 331K Kickstarter campaigns (calibrated LightGBM, test ROC-AUC 0.776; SHAP, SBERT/FAISS retrieval, six-agent analysis) with a FastAPI backend, React/TypeScript dashboard, MLflow and 95 tests.

### 3c. Bullet bank — swap in by what the role cares about

**Modeling and evaluation**
- Tuned a LightGBM classifier with a 75-fit randomized search and **sigmoid calibration**, choosing the F1-optimal threshold (0.329) on validation; recall rose from 0.58 to 0.82 versus the default 0.5 cut-off.
- Kept evaluation honest: held-out test set scored once, validation and test PR-AUC within 0.0001, and a performance report that documents the weak points (random split on 2009–2017 data, precision 0.56, no per-segment analysis).

**Explainability and trust**
- Built SHAP explanations as a **bar chart and an additive waterfall** that reconcile exactly to the model margin (baseline −0.70 → prediction −0.35), exposing `base_value` and `margin` through the API.
- Designed an **"honesty boundary"**: the API rejects fields the data can't support (422), and scores with no data basis — founder strength, market fit, scalability and two more, plus market size — return as *deferred* and render as visible placeholder cards, never as invented numbers.

**Systems and backend**
- Architected a 13-layer system as **one warm process** (LightGBM stack loaded before FAISS/torch to avoid an OpenMP conflict), where every layer composes the one below it so a single model object serves all of them.
- Generated API documentation **from the live OpenAPI schema and real responses**, including captured error cases, so the docs cannot drift from the code.

**Full-stack**
- Built a React + TypeScript dashboard (SHAP charts, live scenario slider, retrieval tables, printable reports) typed against the API's response schemas; verified end to end with **headless Chrome** across desktop and 320–390 px mobile.
- Caught three UI defects by rendering the real app — a negative committee verdict shown in a green badge, a mobile layout overflow, and clipped date fields — and fixed them.

**Engineering practice**
- Maintained a **release evidence ledger** mapping every README and resume claim to code or an artifact, dropping claims that couldn't be proven (e.g. a neural network that was never built).
- Added 95 tests across data validation, model validation, retrieval quality, SHAP additivity, API contracts and latency budgets; CI runs the fast lane (53 tests).

**Research flavour**
- Identified and documented a temporal-drift risk (success rate fell from ~48% to 32–43% across 2009–2017 while `launch_year` is a top-5 feature) and proposed a chronological evaluation as the next step.

### 3d. Skills line (only what the project actually used)

`Python · pandas · scikit-learn · LightGBM · XGBoost · SHAP · TF-IDF · sentence-transformers · FAISS · MLflow · FastAPI · pytest · React · TypeScript · Vite · Tailwind CSS · Docker (config) · Git/GitHub Actions`

---

## 4. GitHub

**Repository name.** The repo is currently `AI-Ventutre-Intelligence-Platform` — the typo ("Ventutre") is
visible on your profile and in every link. Renaming it to `ai-venture-intelligence-platform` (Settings →
General → Repository name) is your call: GitHub redirects the old URL, but links printed on a resume should
use the final name, so **decide before you send anything**. After a rename, update your local remote:
`git remote set-url origin https://github.com/yashraj-a11y/ai-venture-intelligence-platform.git`.

**About (≤350 characters)**

> End-to-end ML platform for Kickstarter campaigns: calibrated funding prediction, SHAP explanations, SBERT/FAISS retrieval, counterfactuals and a six-agent committee, served by FastAPI and a React dashboard. Shows "deferred" where the data can't support a score.

**Topics:** `machine-learning` · `explainable-ai` · `shap` · `lightgbm` · `fastapi` · `react` · `typescript` · `faiss` · `sentence-transformers` · `mlflow` · `kickstarter` · `data-science`

**Social preview:** upload [`docs/assets/social-preview.png`](assets/social-preview.png) (Settings → Social preview).

**Pin it, and make the first screen count:** the README already opens with the honesty note, a screenshot and the results table. Don't add badges you can't back (no build badge until CI is public and green, no coverage badge).

---

## 5. LinkedIn

**Project entry (Projects section)**

> **AI Venture Intelligence Platform** — I built an end-to-end machine-learning system that analyzes Kickstarter campaigns: a calibrated funding prediction, SHAP explanations of why, similar past campaigns found with sentence embeddings and FAISS, counterfactual "what would help" analysis, and a six-agent due-diligence verdict — behind a FastAPI backend and a React/TypeScript dashboard.
>
> Held-out test ROC-AUC 0.776; on validation it beat a logistic-regression baseline 0.777 to 0.727. I documented the limits as carefully as the results: the data stops in 2017, and anything the data can't support (founder strength, market fit, …) is shown as "deferred" rather than invented.
>
> Stack: Python, LightGBM, SHAP, SBERT, FAISS, MLflow, FastAPI, React, TypeScript. `https://github.com/yashraj-a11y/AI-Ventutre-Intelligence-Platform`

*(The baseline was scored on validation only, so the comparison is quoted on validation: `logreg_results.json`, `model_metadata.json`. The 0.776 is the final model's separate test-set score.)*

**Post (≈180 words) — edit to your voice**

> I just finished a 20-day build: an AI platform that analyzes Kickstarter campaigns end to end.
>
> You submit a campaign; it returns a calibrated funding probability, the features driving it (SHAP, with a waterfall that adds up exactly), similar campaigns that funded and failed, the single changes that would help most, and a committee-style investor verdict.
>
> Two things I'm proudest of:
>
> **1. It says "I don't know."** The data has no founder, team or revenue information, so the scores and market-size figure a dashboard would normally show are rendered as "deferred — no data basis". The API even rejects fields it was never trained on.
>
> **2. I wrote down where it's weak.** Test ROC-AUC is 0.78 — real signal, modest ceiling — and the data ends in 2017 with a random split, so real-world performance is probably lower. That's in the report, not hidden.
>
> Rendering the UI for screenshots also caught a bug the tests missed: a "Do Not Invest" verdict was showing in green.
>
> Code, docs and a demo guide: `https://github.com/yashraj-a11y/AI-Ventutre-Intelligence-Platform`
>
> `#machinelearning #explainableai #fastapi #react`

**Media to attach:** `02-founder-submitted.png`, `04-shap-bar-waterfall.png`, `10-investor-verdict.png`, and `[demo video URL]` once recorded (all in [`docs/assets/`](assets/README.md)).

---

## 6. Pitches

**10 seconds**
"I built an ML platform that predicts Kickstarter funding, explains why, and shows its own limits."

**30 seconds**
"I built an end-to-end system on about 330,000 Kickstarter campaigns. You submit a campaign and get a calibrated funding probability, a SHAP explanation, similar past campaigns, what to change, and a due-diligence verdict — through a FastAPI backend and a React dashboard. The model reaches 0.78 ROC-AUC on a held-out test set — about 0.05 above logistic regression on validation. What I care most about is that it's honest: it refuses inputs the data can't support and marks the scores it can't compute as deferred, instead of inventing them."

**2 minutes** *(30-second version, then:)*
"Technically: the features are 800 title TF-IDF terms plus 65 tabular ones, fused as a sparse matrix, and the pipeline is fit on train only so leakage can't happen by construction. I benchmarked five tree ensembles against a regularized logistic baseline, tuned LightGBM and calibrated it, and scored the test set once. Retrieval is sentence embeddings with an exact FAISS index over 265,000 campaigns. Thirteen layers share one warm process and one model object.
The limits matter too. The data ends in 2017 and my split is random rather than chronological, so the numbers are probably optimistic for a campaign launched today — I wrote that up and listed a time-based evaluation as the next step. Precision is only 0.56 at the threshold I chose for recall. And I never built a neural network or any founder-level scoring, because there's no data for it."

### Short stories for "tell me about a challenge"

**The honesty boundary.** *Situation:* the roadmap wanted founder-strength and market-fit scores. *Problem:* the dataset has no founder, team or market columns, so any number would be invented. *Action:* made the API reject unsupported fields, return those scores as deferred with a stated basis, and render them as visible placeholders. *Result:* the UI can't show a number the backend didn't send — and I could defend every figure.

**Catching a bug by looking.** *Situation:* all tests and the typecheck were green. *Action:* I rendered the real UI with headless Chrome for portfolio screenshots. *Result:* found a "Do Not Invest" verdict in a green badge (a substring match on "invest"), a mobile layout overflow, and clipped date fields — fixed all three. Lesson: tests verify what you thought to test.

**Finding the evaluation weakness myself.** *Situation:* validation and test scores agreed to 0.0001. *Action:* while writing the performance report I checked the data's date range and noticed it ends in 2017, the split is random, and `launch_year` is a top-5 feature. *Result:* documented that scores likely overstate performance on new campaigns, and proposed a chronological split — rather than letting a good-looking number stand.

---

## 7. Draft bullets that were rewritten, and why

The original roadmap suggested four bullets. They are not safe as written:

| Draft wording | Problem | Used instead |
|---|---|---|
| "**multimodal** machine learning" | The "modes" are title text and tabular fields — no images, audio or video. Interviewers hear "multimodal" as vision/language. | "text + tabular feature fusion (TF-IDF + sparse `hstack`)" |
| "for **startup** funding intelligence" | The data is Kickstarter crowdfunding, not venture-backed startups. | "Kickstarter campaigns" |
| "**multi-agent** analysis" | Reads as LLM agents. | "six-agent due-diligence committee" and "rule-based" where relevant |
| "deployment-ready architecture", "**Dockerized deployment**" | The Docker image was never built. | "Docker config"; skills line says "Docker (config)" |
| "experiment tracking with MLflow" | True, but local-store only. | kept, with the registry mentioned where space allows |
| *(implied)* "production" | One-machine, no load test, no real auth. | "typed FastAPI service", "CI", no "production" |

Do not add: a neural network, an LLM, "state-of-the-art", "deployed", real authentication, founder or
market scoring, or "predicts startup success".

## 8. Tailoring

| Role | Lead with | Cut |
|---|---|---|
| **ML engineer** | 3a bullets 1, 2, 4; "Systems and backend" bank | research-flavour |
| **Applied scientist / data scientist** | bullet 2; "Modeling and evaluation"; "Explainability and trust"; the drift bullet | frontend bullets |
| **Research internship** | the drift bullet; honest-evaluation bullet; link the [Performance Report §12](PERFORMANCE_REPORT.md#12-weaknesses-and-limits) and [research roadmap](RESEARCH_ROADMAP.md) | engineering-practice bullets |
| **Full-stack / AI-app** | bullet 1; "Full-stack" bank; screenshots | modeling detail |

## 9. Before you send

- [ ] **Push the work** — GitHub currently lacks Days 18–20 (README, docs, frontend, assets), so your link would show an empty repo. Pushing is your call.
- [ ] Decide on the repo rename (§4) *before* putting the URL on a resume.
- [ ] Fill `[demo video URL]` and `[your name]`.
- [ ] **Open the file behind every number you keep** (§2 names it). If you can't, delete the number.
- [ ] Record the demo video ([DEMO_GUIDE.md §9](DEMO_GUIDE.md#9-screenshot-and-recording-shot-list)) — it is the one asset still missing.
- [ ] Rehearse the weaknesses out loud: the 2017 data, the random split, precision 0.56, no Docker build.
- [ ] Read the [interview guide](INTERVIEW_GUIDE.md) *(Phase I)* so each bullet has a prepared "how?".
- [ ] Don't claim more than the bullets say — the strongest thing about this project is that every line survives a follow-up question.
