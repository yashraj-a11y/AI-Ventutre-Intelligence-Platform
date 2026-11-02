# Day 18 — Production Deployment, Dockerization & Cloud-Ready Architecture (FROZEN SPEC V1.1)

**Roadmap V1.1 — Deployment phase.** Package the platform (the Day-14 FastAPI app + all 13 layers) into
a reproducible Docker image + Compose stack, add environment config / file logging / health probes /
deploy scripts / a cloud blueprint / a security checklist — **reusing the existing app, logger, config,
and `/health`, not rebuilding them.**

> **🔴 READ THIS FIRST — two hard constraints shape the whole day.**
> 1. **Docker is NOT installed in this build environment.** Day 18 *writes* and *statically validates*
>    the Dockerfile / compose / scripts (YAML validity, `.dockerignore`, shell syntax, config loads via
>    TestClient) — but `docker build` / `docker run` can only be verified on a Docker-enabled machine or
>    CI. Every "it builds/runs" claim is explicitly **unverified here**, not asserted.
> 2. **FAISS is IN-PROCESS, not a service.** The vector search is an embedded library loaded *into* the
>    FastAPI process — there is **no separate vector-DB server**. The Compose stack is essentially **one
>    app container** (which holds models + FAISS + SBERT), optionally fronted by nginx.

> **Status:** FROZEN 2026-08-13 (V1.1). Build phase-by-phase on `build Phase <X>`; each phase =
> implement/validate **and** teach via Framework 1, grounded in real numbers.

---

## 0. What already exists vs. what's new

| Item | Status | Where |
|---|---|---|
| FastAPI app (10 routes) + warm registry + lifespan | ✅ **DONE** | `src/api/`, Day 14 |
| `/health` (already a readiness probe: `reg.ready()`) | ✅ **DONE** | `src/api/routes/health.py` |
| Logger with a **FileHandler** already wired | ✅ **DONE** | `src/utils/logger.py` |
| Config loader (`load_config` over `configs/preprocessing.yaml`) | ✅ **DONE** | `src/utils/config.py` |
| pydantic input validation (`extra="forbid"` → 422) + typed errors | ✅ **DONE** | Day 14 |
| NLTK-data download script | ✅ **DONE** | `scripts/download_nltk_data.py` |
| `psutil` (for health diagnostics) | ✅ **INSTALLED** | — |
| **Dockerfile + `.dockerignore`** | ❌ **NEW** | Day 18 |
| **docker-compose.yml** (app + optional nginx / mlflow-ui) | ❌ **NEW** | Day 18 |
| **`.env.example` + env-aware config** | ❌ **NEW** | Day 18 |
| **`/live` + `/ready` probes; disk/memory in `/health`** | ❌ **NEW** | Day 18 |
| **`scripts/{build,run,stop,deploy}.sh`** | ❌ **NEW** | Day 18 |
| **`deployment/{nginx.conf,cloud_architecture.md,deployment_guide.md,security_checklist.md}`** | ❌ **NEW** | Day 18 |
| **`notebooks/18_Deployment.ipynb`** | ❌ **NEW** | Day 18 |

---

## 1. What changed from the draft (changelog + why)

| # | Draft said | 🔧 Frozen correction | Why |
|---|---|---|---|
| **C-1** | (implied) build & run the image | **Docker is NOT available here** — write + **statically validate** (compose/YAML parse, `.dockerignore`, `bash -n` script syntax, app boots via TestClient). `docker build`/`run` verified by the **user / CI**, not here. | Honesty: never claim an image builds/runs when the daemon isn't present. |
| **C-2** | docker-compose service **"Vector Database"** | **No separate vector-DB service — FAISS is in-process.** The stack is **one app container** (FastAPI + models + FAISS + SBERT) + optional **nginx** + optional **mlflow-ui**. | FAISS is an embedded library loaded into the API process (verified Days 6/14), not a DB server. |
| **C-3** | "FAISS/**ChromaDB**" | **FAISS only — ChromaDB rejected** (Day-15 C-3). | The index is already persistent FAISS; ChromaDB adds nothing and no new dep. |
| **C-4** | **Redis** (optional caching) service | **DEFER — discussion only.** No caching layer exists; adding Redis = new service + dep + code. Named in the blueprint, not built. | Keep the stack honest to what the platform actually does (deterministic, no cache). |
| **C-5** | **MLflow** as a runtime service | **MLflow is a LOCAL sqlite file store (Day 16), not a serving dependency.** The API doesn't import mlflow at serve time. In Compose it's at most an **optional `mlflow ui`** dev container over the local db. | The serving path never needs a tracking server; overclaiming infra is dishonest. |
| **C-6** | "TF-IDF (2,500 features)" + "Neural Networks" | **TF-IDF = 800 (fused 865); Neural Network DEFERRED (none trained).** | Same repeated boilerplate corrected on Days 16–17; the real cap is 800 and no NN exists. |
| **C-7** | (implied) COPY everything into the image | **🔑 The trained artifacts are git-ignored and large** (models 21 MB, FAISS index **407 MB**, embeddings **407 MB**, metadata.csv 73 MB). **Mount `artifacts/` + `data/` as volumes** (Compose, small image); document **baking them in** (COPY) as the self-contained cloud option. The SBERT model + NLTK data are fetched at **build** time (reuse `download_nltk_data.py` + a model-warm step). | A repo/CI build has no access to the git-ignored artifacts; volumes keep the image small and the build reproducible. |
| **C-8** | (implied) small image | **Image is inherently large (~GBs): torch + faiss + sentence-transformers.** Use `python:3.14-slim`, **CPU-only torch**, multi-stage build; note that free cloud tiers may not fit. | "Cloud-ready" must be honest about the multi-GB ML image, not pretend it's tiny. |
| **C-9** | Build **new** `logger.py` / `config.py` / `health.py` | **Reuse + extend** the existing ones: configure the logger's **existing FileHandler** for `logs/application.log`; add env overrides to `load_config` (no competing `configs/config.py`); add `/live` + `/ready` beside the existing `/health`. | They exist and work; rebuilding risks regressing a committed system (same discipline as Days 14–17). |
| **C-10** | `.env` stores **API keys / secrets** | **The platform has NO external API keys/secrets** (deterministic, no LLM, no external services). `.env` manages **ports, artifact/data paths, log level, deployment mode, worker count, CORS origins** — not fabricated secrets. | Don't invent a secrets-management need the system doesn't have. |
| **C-11** | (implied) many workers | **Default 1 uvicorn worker** — each worker loads the full stack (models + 407 MB FAISS ≈ 1 GB RSS, measured Day 17). **Scale by more containers, not more workers.** | Multi-worker multiplies the 1 GB footprint; honest scaling is horizontal replicas behind nginx. |

**IMPROVEMENTS:** IMP-1 artifacts via **volume mount** (small image) for Compose; baking documented for
cloud. IMP-2 **CPU-only torch** + multi-stage to shrink the image. IMP-3 **reuse** logger/config/health.
IMP-4 `.dockerignore` excludes `.venv`, `artifacts/`, `data/`, `.git`, tests, notebooks → tiny build
context. IMP-5 Compose **healthcheck** + `OMP_NUM_THREADS=1` env (faiss+torch). IMP-6 config/probes
verified by **TestClient** locally (the one thing runnable without Docker).

---

## 2. Honesty tiers

**Tier 1 — REAL:** the Day-14 FastAPI app + 10 routes + `/health`, in-process FAISS, local sqlite
MLflow (training-time only), real models/artifacts (mounted), the logger's file handler, pydantic
input validation + typed errors, `scripts/download_nltk_data.py`, measured 1 GB RSS.
**Tier 2 — DERIVED (labeled), statically validated not build-run here:** Dockerfile, docker-compose,
`.dockerignore`, `.env.example`, env-aware config, `/live` + `/ready`, deploy scripts, nginx.conf,
CORS config.
**Tier 3 — DEFERRED / discussion, never fabricated:** ChromaDB, Redis caching, a separate vector-DB
server, a remote MLflow tracking server, Kubernetes / horizontal autoscaling, actual cloud deployment
(blueprint only), secrets management (no secrets exist), Neural Network, rate-limiting at app layer
(nginx-level noted).

---

## 3. Phases (build order)

- **A** — **Containerization notes** (`docs/day18_phaseA_containerization_notes.md`): containers vs VMs,
  images/layers, Dockerfile, registries, Compose, why "in-process FAISS = one container", the
  git-ignored-artifacts→volume reality, the multi-GB ML-image tradeoff. Framework 1.
- **B** — **`Dockerfile` + `.dockerignore`:** multi-stage, `python:3.14-slim`, **CPU-only torch**,
  install pinned deps, run `scripts/download_nltk_data.py` + warm the SBERT model at build, copy `src/`
  + `configs/`, `ENV OMP_NUM_THREADS=1`, `EXPOSE 8000`, `CMD uvicorn src.api.main:app`. Artifacts NOT
  copied (mounted). **Validate:** `.dockerignore` correctness, Dockerfile lint (manual), that the CMD
  matches the real app path. *(No `docker build` here.)*
- **C** — **`docker-compose.yml`:** service **`api`** (build ., ports 8000, `OMP_NUM_THREADS=1`, volumes
  `./artifacts`, `./data`, `./logs`, healthcheck → `/health`), optional **`nginx`** (reverse proxy,
  `deployment/nginx.conf`), optional **`mlflow-ui`** (dev). **No** vector-db/redis/chromadb service.
  Validate: `yaml.safe_load` + service/volume/network shape.
- **D** — **Environment management:** `.env.example` (PORT, LOG_LEVEL, LOG_FILE, DEPLOY_MODE,
  ARTIFACTS_DIR, DATA_DIR, WORKERS, CORS_ORIGINS — **no secrets**) + extend `load_config`/registry to
  read env overrides; validate required vars. Reuse `src/utils/config.py`; no competing `config.py`.
- **E** — **Logging:** configure the **existing** logger's FileHandler → `logs/application.log`, level
  from `LOG_LEVEL`; ensure API request/prediction/error events flow to it (Day-14 already logs). Rotate
  optionally. `logs/` git-ignored.
- **F** — **Health & diagnostics:** add **`/live`** (process up → always 200) and **`/ready`**
  (`reg.ready()` → 503 if loading) beside the existing `/health`; enrich `/health` with disk + memory
  (psutil). TestClient-verified.
- **G** — **Deploy scripts** (`scripts/{build,run,stop,deploy}.sh`): build image, `compose up -d`,
  `compose down`, clean; `set -euo pipefail`; `bash -n` syntax-validated. `deploy.sh` = compose-based.
- **H** — **Cloud blueprint** (`deployment/cloud_architecture.md`): client → nginx → FastAPI
  (in-process FAISS/models) → local sqlite MLflow; external artifact storage, SSL termination,
  auto-restart, horizontal replicas; discuss EC2/Azure/GCP/Render/Railway/Fly + the **image-size /
  artifact-volume** reality. Blueprint, not deployed.
- **I** — **Security & hardening** (`deployment/security_checklist.md` + code): add **CORS middleware**
  (starlette, no new dep); **cite** already-done input validation (pydantic `extra=forbid`) + error
  sanitization (typed errors) + dependency pinning; rate-limiting = **nginx-level** (or slowapi
  deferred); HTTPS via nginx/blueprint. Honest done-vs-deferred checklist.
- **J** — **Docs** (`deployment/deployment_guide.md`): purpose / setup / docker workflow / env vars /
  deploy steps / troubleshooting / example commands.
- **Notebook** — `notebooks/18_Deployment.ipynb`: parse+validate the Dockerfile/compose/env, boot the
  app via TestClient, hit `/health` `/live` `/ready`, show the config + logging, the artifact-volume &
  image-size reality — **honest that Docker itself isn't run here**.

---

## 4. Environment & repo

- **No new Python deps** — CORS is built into starlette/fastapi; env via `os.environ`; `psutil` already
  present. Docker/Compose/nginx are **external tooling**, not pip deps. **CPU-only torch** is a
  Dockerfile install choice, not a repo change.
- **🔴 `OMP_NUM_THREADS=1`** baked into the image env (faiss + torch coexistence).
- **New files:** `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `.env.example`,
  `scripts/{build,run,stop,deploy}.sh`, `deployment/{nginx.conf,cloud_architecture.md,deployment_guide.md,security_checklist.md}`,
  `docs/day18_{phaseA_containerization_notes}.md`, `notebooks/18_Deployment.ipynb`. **Extend** (don't
  rebuild) `src/api/routes/health.py`, `src/utils/{logger,config}.py`, `src/api/main.py` (CORS).
- **Artifacts:** mounted via volume in Compose (image stays lean); baking documented for cloud.

---

## 5. Non-negotiables (frozen)

1. **Docker not run here** — write + statically validate; build/run is the user's/CI's step (C-1).
2. **One app container; FAISS in-process** — no separate vector-DB/Redis/ChromaDB service (C-2/C-3/C-4).
3. **MLflow is local/training-time**, at most an optional dev UI container (C-5).
4. **Reuse existing logger/config/health**; extend, don't rebuild (C-9).
5. **Artifacts by volume (small image); document baking**; CPU-only torch; honest multi-GB size (C-7/C-8).
6. **No fabricated secrets/API-keys, no NN, TF-IDF 800** (C-10/C-6).
7. Teach every phase via Framework 1, grounded in real numbers; end with an interview soundbite.

---

*Frozen 2026-08-13. Supersedes the Day-18 draft. Day 18 = Dockerize + Compose + env/logging/health/
deploy-scripts/cloud-blueprint/security over the EXISTING FastAPI platform; statically validated
(Docker build/run is the user's step). Next: Day 19 — Frontend & Dashboard.*
