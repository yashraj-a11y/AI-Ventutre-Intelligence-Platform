# Day 14 — Production Backend, API Layer & System Integration (FROZEN SPEC V1.1)

**Roadmap V1.1 — Production phase begins.** Wrap the whole Architecture-V1.0 platform (Days 1–13) in
a single FastAPI backend: strongly-typed request validation, a singleton model loader (one warm
stack), a thin service layer over the *existing* pipelines, REST endpoints for every layer, an
`/analyze` orchestrator, production error handling + logging, and an API test suite.

> **Status:** FROZEN 2026-08-05 (V1.1). Read before building any phase. Build phase-by-phase on
> `build Phase <X>`; each phase = implement **and** teach via Framework 1, grounded in real numbers.
> No notebook this day — the deliverables are a running server + a pytest suite + a smoke script.

---

## 0. Preflight findings (verified before freezing)

- **Deps present:** `pydantic 2.13`, `httpx 0.28`, `pytest 9.1`. **MISSING (must install):** `fastapi`,
  `uvicorn[standard]`, `python-multipart`. → Day 14 **adds dependencies** (justified — it's the web
  layer; the first new-deps day since Day 6).
- **🔴 OpenMP coexistence VERIFIED:** the lightgbm stack (Days 4–13) and Day-6 faiss+SBERT/torch
  **coexist in ONE process** when `OMP_NUM_THREADS=1` is set *before any import* **and** the lightgbm
  stack is loaded *before* faiss. Ran `/analyze` + `/retrieve` interleaved, no crash/hang. → **one
  FastAPI backend can serve every endpoint** (no microservice split needed for Day 14).
- **Dirs clean:** `src/api`, `src/core`, `src/services` are new. `src/utils/{config,logger}.py`
  already exist → **reuse them** (don't duplicate as `core/config.py`, `core/logger.py`).

---

## 1. What changed from the draft (changelog + why)

| # | Draft said | 🔧 Frozen correction | Why |
|---|---|---|---|
| **C-1** | Request schemas for "Founder profile", "Financial metrics", "Pitch deck metadata" | **Request schema = REAL Kickstarter campaign fields** (`name`, `main_category`, `category`, `usd_goal_real`, `goal`, `currency`, `launched`, `deadline`, `country`, optional `blurb`/`name_clean`). Founder / financials / pitch-deck → **rejected** (don't exist; the platform never modeled them). | Same honesty discipline as Days 4–13. The API must accept what the models actually consume. |
| **C-2** | (implicit) each endpoint runs its module | **🔴 `OMP_NUM_THREADS=1` set at the TOP of the entrypoint before any import + model loader builds the lightgbm stack BEFORE the faiss retriever.** | The only way lightgbm + faiss+torch survive in one process (preflight-verified). Get this wrong → segfault/hang on the first `/retrieve`. |
| **C-3** | `core/config.py`, `core/logger.py` | **Reuse `src/utils/config.py` + `src/utils/logger.py`** (already exist, used by all 13 layers). `core/` holds `model_loader.py` + `exceptions.py` only. | Don't fork config/logging; the whole platform already shares them. |
| **C-4** | services "run" the modules | **Services are THIN wrappers over the EXISTING pipelines** (Day-4 `PredictionEngine`, Day-5 `LocalExplainer`, Day-6 `VentureRetriever`, Day-8 `CounterfactualPipeline`, Day-10 evolution/scenario, Day-11 `RecommendationPipeline`, Day-12 `ConsensusEngine`, Day-13 `ReportingPipeline`). **No reimplementation.** The loader builds ONE shared warm stack (one `cf`) + one retriever, injected via FastAPI DI. | Reuse; the pipelines are already built, audited, and faiss-safe. |
| **C-5** | `/analyze` runs the full pipeline incl. retrieval | **`/analyze` = the faiss-free Day-13 `ReportingPipeline`** (its "similar startups" is the frame cohort, not faiss). Semantic **retrieval is its own endpoint** `/retrieve` (Day-6, faiss). | Keeps the core `/analyze` path faiss-free; faiss is exercised only by `/retrieve`, minimizing OpenMP surface. |
| **C-6** | (unspecified) auth / DB / rate-limit | **Out of scope for Day 14** (local dev backend): no auth, no DB, no rate-limiting, single uvicorn worker. Add CORS (permissive dev) only. Note as deferred to later production days (15+/deploy). | Day 14 = API surface + orchestration + validation + errors + tests. Security/scaling come later. |

**IMPROVEMENTS beyond the draft:**
- **IMP-1 — singleton loader via FastAPI lifespan.** Build the warm stack ONCE at startup (lifespan
  event), store in an app-state registry, inject read-only into services. Never per-request.
- **IMP-2 — OMP set programmatically** at the very top of `main.py`/entrypoint (`os.environ`) as a
  belt-and-suspenders alongside the launch env var, so the mitigation can't be forgotten.
- **IMP-3 — honest response schemas** — surface the real outputs *and* the `deferred_appendix` /
  honesty notes; never invent fields.
- **IMP-4 — `/health` reports readiness + integrity** — loaded-artifact status + optional
  content-hash `verify()` of the Day-13 reporting manifest (the platform can self-certify on boot).
- **IMP-5 — typed error taxonomy** — `ValidationError→422` (pydantic, automatic), `ArtifactError→503`,
  `InferenceError→500`, with structured logging + a request id.
- **IMP-6 — deterministic API** — same request → same response body (modulo `generated_utc`); tests
  assert this.

---

## 2. Honesty tiers (API I/O)

**Tier 1 — REAL request fields (accepted):** name, main_category, category, usd_goal_real, goal,
currency, launched, deadline, country, blurb/name_clean (optional). **Tier 2 — DERIVED responses
(labeled model-derived):** funding probability, readiness, recommendations, committee decision,
reports — each carrying provenance / honesty notes from the layers. **Tier 3 — DEFERRED, rejected at
the schema boundary:** founder profile / experience / team, revenue / MRR / financials, pitch-deck
metadata, moat, expected return. The API **422s** unknown or fabricated fields rather than accepting
them.

---

## 3. Architecture

```
Client → FastAPI (main.py, OMP=1, lifespan-loaded registry)
           → routes/ (thin: validate -> service -> response schema)
             → services/ (thin wrappers over existing pipelines)
               → core/model_loader.py  (ONE warm stack: cf + retriever + reporting pipeline)
                 → Days 4-13 modules + Day-6 retriever  (loaded once, lightgbm-before-faiss)
```

**Endpoints (final):**

| Method | Path | Service → layer |
|---|---|---|
| GET | `/health` | loader status + optional manifest verify |
| GET | `/docs` | auto Swagger (FastAPI) |
| POST | `/predict` | Day-4 `PredictionEngine` |
| POST | `/explain` | Day-5 `LocalExplainer` |
| POST | `/retrieve` | Day-6 `VentureRetriever` (faiss) |
| POST | `/counterfactual` | Day-8 `CounterfactualPipeline` |
| POST | `/scenario` | Day-10 `ScenarioPipeline` |
| POST | `/recommend` | Day-11 `RecommendationPipeline` |
| POST | `/analyze` | Day-13 `ReportingPipeline` (full suite + committee) |
| POST | `/report/founder` | Day-13 founder report |
| POST | `/report/investor` | Day-13 investor report |

---

## 4. Phases (build order)

- **A** — Backend architecture notes (client/server, REST, DI, service layer, the OMP/load-order
  constraint). `docs/day14_phaseA_backend_notes.md` + `src/api/__init__`, `src/core/__init__`,
  `src/services/__init__`.
- **B** — FastAPI setup: **install `fastapi uvicorn[standard] python-multipart`**; `src/api/main.py`
  (app, `OMP_NUM_THREADS=1` at top, lifespan hook, CORS, router include, exception handlers); confirm
  `uvicorn src.api.main:app` boots + `/docs` renders + `/health` 200.
- **C** — Schemas: `src/api/schemas/{campaign.py, responses.py}` — `CampaignRequest` (real fields,
  validation) + typed response models (prediction / analysis / report), honest + `deferred` fields.
- **D** — `src/core/model_loader.py` **`ModelRegistry`**: build ONE warm stack (cf → reporting
  pipeline + consensus) **then** the faiss retriever (load-order), lazy singleton, `.ready()`.
- **E** — `src/services/` thin wrappers: prediction / explainability / retrieval / counterfactual /
  scenario / recommendation / report / **pipeline (analyze)** — each takes the registry, calls the
  existing pipeline, returns a plain dict.
- **F** — `src/api/routes/`: one router module per group; `src/api/dependencies.py` DI providers
  (`get_registry`, `get_<service>`). Thin: validate → service → response.
- **G** — `/analyze` orchestration in `pipeline_service.py` = `ReportingPipeline.run` (one bundle →
  reports + committee + copilot), returned as a typed `AnalysisResponse`.
- **H** — `src/core/exceptions.py`: `ArtifactError`/`InferenceError` + FastAPI exception handlers →
  422/503/500 with request id; structured logging via `src/utils/logger`.
- **I** — `tests/test_api.py`: FastAPI `TestClient` (httpx present) — `/health` 200, `/predict`
  valid+`422` on bad input, `/analyze` shape, `/report/*`, determinism, response-time sanity.

---

## 5. Repository after Day 14

```
src/api/{__init__, main.py, dependencies.py,
         schemas/{__init__, campaign.py, responses.py},
         routes/{__init__, health.py, predict.py, explain.py, retrieve.py,
                 counterfactual.py, scenario.py, recommend.py, analyze.py, report.py}}
src/core/{__init__, model_loader.py, exceptions.py}          # config+logger reused from src/utils
src/services/{__init__, prediction_service.py, explainability_service.py, retrieval_service.py,
              counterfactual_service.py, scenario_service.py, recommendation_service.py,
              report_service.py, pipeline_service.py}
tests/test_api.py
docs/day14_phaseA_backend_notes.md
configs/preprocessing.yaml → day14.{api}   (host/port/cors/verify_on_health)
requirements: + fastapi, uvicorn[standard], python-multipart
```

**Run:** `OMP_NUM_THREADS=1 PYTHONPATH=. ./.venv/bin/uvicorn src.api.main:app --port 8000`
(single worker). **PREFLIGHT (before Phase E wiring):** boot the server, hit `/health`, `/predict`,
`/analyze`, `/retrieve` — confirm 200s and **no OpenMP crash**.

---

## 6. Non-negotiables (frozen)

1. **`OMP_NUM_THREADS=1` at entrypoint top + lightgbm-before-faiss load order** (C-2, IMP-2) — the
   one thing that must not be missed.
2. **Reuse existing pipelines** — services are thin; no module reimplementation (C-4). ONE warm stack
   loaded once (IMP-1).
3. **Real request fields only** — founder/financial/pitch rejected at the schema (C-1); responses
   stay honest + deferral-aware (IMP-3).
4. **`/analyze` is faiss-free** (Day-13 pipeline); faiss only in `/retrieve` (C-5).
5. **Reuse `src/utils` config + logger** (C-3). No auth/DB this day (C-6).
6. **Deterministic responses** (modulo timestamp); typed errors 422/503/500 (IMP-5/6).
7. Teach every phase via Framework 1, grounded in real numbers; end with an interview soundbite.

---

*Frozen 2026-08-05. Supersedes the Day-14 draft. Build on `build Phase <X>`.*
