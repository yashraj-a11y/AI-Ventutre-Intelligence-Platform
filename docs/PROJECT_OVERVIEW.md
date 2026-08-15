# Project Overview

## The problem

A Kickstarter campaign either reaches its funding goal or it doesn't. A bare "62% likely to fund" is
not actionable. Founders want to know **what to change**; investors want to know **what to doubt**. This
project builds the whole chain from raw data to a usable decision-support product.

## Who uses it

| User | Gets |
|---|---|
| **Founder** | Funding probability, the drivers behind it, similar successes and failures, the single changes that help most, a readiness roadmap, a founder report |
| **Investor** | Committee verdict and confidence, likely objections, comparables, an investor report |

Both submit the same campaign (title, category, goal, country, currency, launch and deadline dates); the
portals differ in framing, not in the underlying analysis.

## What it is — and is not

The data is ~378,000 historical Kickstarter campaigns. That fixes what can honestly be claimed.

| Real (computed from data) | Deferred (no data basis — shown, never faked) |
|---|---|
| Funding probability, failure risk, startup / investment score (all functions of the calibrated probability) | Business-Model score |
| SHAP drivers, similar campaigns, patterns, counterfactuals, scenarios | Founder-Strength score |
| Category funding rate and trend ("market timing" as a Kickstarter-history signal) | Scalability score |
| Readiness roadmap and strategy from the above | Market-Fit and Growth-Potential scores; market size (TAM / SAM / CAGR) |

The backend rejects any field outside the real campaign schema (`422`), and every response carries an
`honesty_note` and/or a `deferred` block. The frontend renders those as UI.

## The 13 layers

| # | Layer | Code | Day |
|---|---|---|---|
| 1 | Data, preprocessing and multi-modal feature fusion | `src/preprocessing`, `src/features` | 2–3 |
| 2 | Venture knowledge base | `src/knowledge_base` | 6 |
| 3 | Venture intelligence engine (SBERT + FAISS retrieval) | `src/retrieval` | 6, 15 |
| 4 | Prediction engine (LogReg baseline → tuned LightGBM, calibrated) | `src/models`, `src/training` | 4 |
| 5 | Explainability (SHAP, coefficients) | `src/explainability` | 5 |
| 6 | Pattern mining | `src/pattern_mining` | 7 |
| 7 | Counterfactual engine | `src/counterfactual` | 8 |
| 8 | Investor objection simulator, copilot, category analysis | `src/copilot` | 9 |
| 9 | Startup evolution planner | `src/evolution` | 10 |
| 10 | Scenario simulator | `src/scenario` | 10 |
| 11 | Recommendation and strategy engine | `src/recommendation` | 11 |
| 12 | Multi-agent intelligence (6 agents + consensus) | `src/agents` | 12 |
| 13 | Report generation | `src/reporting` | 13 |

Around the layers: `src/api` + `src/services` (FastAPI, Day 14), `src/mlops` (Day 16), `tests/` (Day 17),
`Dockerfile` + `deployment/` (Day 18), `frontend/` (Day 19).

## Key design decisions

1. **One process, not microservices.** The layers share a ~1 GB warm stack; FAISS is an in-process
   library. Splitting would reload the models per service.
2. **Calibrated probability is the single source of truth.** Every score, scenario and report derives
   from it, so the platform never contradicts itself.
3. **Honesty as an interface.** Deferred values are returned as `None` with a basis and rendered as
   such; the UI cannot show a number the API did not send.
4. **Deterministic intelligence layers.** Objections, agents and reports are rule-based over real
   signals — no LLM, no external services, no secrets.
5. **Evidence over adjectives.** Every claim in the docs traces to a file or artifact
   ([release audit](day20_phaseA_release_audit.md)).

## Technology

Python 3.14 · pandas · scikit-learn · LightGBM · XGBoost · SHAP · sentence-transformers
(`all-MiniLM-L6-v2`) · FAISS · MLflow · FastAPI · pytest · React 19 · TypeScript · Vite · Tailwind CSS ·
Recharts · Docker (configured).

## How it was built

Twenty days, one layer at a time, each with an executed notebook (`notebooks/01`–`20`; Day 14 is the
backend in `src/api/` and has no notebook) and a frozen spec (`docs/dayN_roadmap.md`). The day-by-day
record is in [CHANGELOG.md](../CHANGELOG.md).

## Where to go next

[README](../README.md) · [Installation](INSTALLATION.md) · [Deployment](DEPLOYMENT.md) ·
[Architecture](SYSTEM_ARCHITECTURE.md) · [API](API_DOCUMENTATION.md) · [Performance](PERFORMANCE_REPORT.md) · [Demo guide](DEMO_GUIDE.md) · [Portfolio and resume kit](PORTFOLIO_AND_RESUME.md) ·
[Interview guide](INTERVIEW_GUIDE.md) · [Research roadmap](RESEARCH_ROADMAP.md) ·
[Release audit](day20_phaseA_release_audit.md).
