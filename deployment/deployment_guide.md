# Day 18 · Phase J — Deployment Guide

*Operator runbook for deploying the AI Venture Intelligence Platform. Ties together the Dockerfile
(Phase B), Compose stack (C), environment (D), logging (E), health probes (F), scripts (G), cloud
blueprint (H), and security (I).*

> **One container.** The whole platform (FastAPI + all 13 layers + FAISS in-process) runs in a single
> `avip-api` container. The trained artifacts are **mounted**, not baked in.

---

## 1. Prerequisites

- **Docker** (Engine + Compose v2). `docker --version` and `docker compose version`.
- **The trained artifacts on disk.** These are git-ignored and ~900 MB — they must be present locally to
  be mounted:
  - `artifacts/models/calibrated_model.joblib`, `artifacts/models/model_metadata.json`
  - `artifacts/vector_db/faiss.index` (407 MB), `artifacts/embeddings/startup_embeddings.npy`
  - `artifacts/feature_pipeline/`, `artifacts/splits/`, `data/processed/kickstarter_processed.csv`
  - Rebuild them via the day notebooks / `src` modules if missing.
- **~4 GB disk** for the image, **~1.5 GB RAM** free per container (the warm stack is ~1 GB RSS).

---

## 2. Quick start

```bash
cp .env.example .env         # optional — sane defaults; NO secrets to set
./scripts/build.sh           # build avip-api:latest (multi-stage; ~minutes; NLTK + SBERT baked)
./scripts/run.sh             # docker compose up -d  (mounts artifacts/ + data/ read-only)
curl -fs localhost:8000/ready   # -> {"status":"ready"} once the model load finishes (~tens of seconds)
curl -s localhost:8000/health | python -m json.tool
```

Stop with `./scripts/stop.sh`. Redeploy (build + restart + wait for `/ready`) with `./scripts/deploy.sh`.

---

## 3. The Docker workflow

| Script | Does | Under the hood |
|---|---|---|
| `scripts/build.sh` | build the image | `docker build -t avip-api:latest .` |
| `scripts/run.sh` | start the stack | seeds `.env`, checks `artifacts/`+`data/`, `docker compose up -d` |
| `scripts/stop.sh` | stop the stack | `docker compose --profile proxy --profile dev down "$@"` |
| `scripts/deploy.sh` | build + restart + wait | build → `up -d` → poll `/ready` (60×3s), dump logs on timeout |

**Optional side-cars** (Compose profiles):

```bash
WITH_PROXY=1 ./scripts/run.sh          # + nginx reverse proxy (deployment/nginx.conf)
WITH_MLFLOW=1 ./scripts/run.sh         # + mlflow ui (dev viewer over the local sqlite store)
# or directly:
docker compose --profile proxy up -d
docker compose --profile dev up -d
```

---

## 4. Environment variables

Set in `.env` (copied from `.env.example`). **There are no secrets/API keys** — the platform is
deterministic with no external services.

| Var | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | host port the API listens on |
| `WORKERS` | `1` | uvicorn workers — **keep at 1**; each loads the ~1 GB stack. Scale by containers. |
| `LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` (validated; bad value fails fast) |
| `LOG_FILE` | `logs/application.log` | file sink; empty = console only |
| `DEPLOY_MODE` | `production` | `development`/`staging`/`production` |
| `ARTIFACTS_DIR` / `DATA_DIR` | `artifacts` / `data` | mounted read-only |
| `CORS_ORIGINS` | *(empty)* | comma-separated allow-list for a browser frontend; **empty = no cross-origin** |
| `HTTP_PORT` / `MLFLOW_PORT` | `80` / `5000` | side-car ports (proxy / dev profiles) |

---

## 5. Health probes

| Endpoint | Returns | Use |
|---|---|---|
| `GET /live` | `200 {"status":"alive"}` always | liveness — restart only if the process is dead |
| `GET /ready` | `200` when loaded, else **`503`** | readiness — gate traffic; don't route until 200 |
| `GET /health` | status + registry components + disk/memory/CPU | diagnostics / dashboards |

The Compose healthcheck probes `/health`; `deploy.sh` waits on `/ready`.

---

## 6. Logs

- Written to `logs/application.log` (mounted volume) when `LOG_FILE` is set — one file for all 13 layers.
- Also to stdout → `docker compose logs -f api`.
- Every API request logs `METHOD /path -> status (ms)` with an `x-request-id`.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: docker not found` | Docker not installed / not on PATH | install Docker Engine + Compose v2 |
| `./artifacts not found` on run | trained artifacts absent (git-ignored) | put the artifacts in place (rebuild via notebooks) before mounting |
| `/ready` stuck on `503` | still loading, or artifacts missing/mismatched | wait ~tens of seconds; check `docker compose logs api` for load errors |
| Container OOM-killed | < ~1.5 GB RAM available | give the host more memory; run **1 worker**; scale by containers not workers |
| Image build fails on torch | CPU-index version / arch mismatch | confirm the pinned `torch` exists on `download.pytorch.org/whl/cpu`; adjust the pin |
| Image very large (multi-GB) | torch + faiss + transformers | expected — use the slim base + CPU torch already in the Dockerfile; some free tiers won't fit |
| Browser CORS errors | `CORS_ORIGINS` not set | add your frontend origin to `CORS_ORIGINS` (comma-separated) |
| First request slow (~seconds) | cold model warm-up | one-time; `/ready` gates traffic until warm (probes already handle this) |

---

## 8. Example commands (cheat-sheet)

```bash
# lifecycle
./scripts/build.sh                       # build
./scripts/run.sh                         # start
./scripts/deploy.sh                      # build + restart + wait for readiness
./scripts/stop.sh                        # stop
./scripts/stop.sh --volumes              # stop + remove named volumes

# inspect
docker compose ps
docker compose logs -f api
docker stats avip-api                    # live memory/CPU
curl -fs localhost:8000/ready

# call the API
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"name":"Artificer","main_category":"Games","usd_goal_real":30000,
       "launched":"2016-05-01 12:00:00","deadline":"2016-05-31 12:00:00"}' | python -m json.tool
```

---

> **Interview soundbite:** *"The deployment guide is an operator runbook, not prose — quick start in five
> commands, a table per script and env var, and a troubleshooting matrix that maps symptoms to causes.
> The failure modes it documents are the real ones for this platform: the 900 megabytes of artifacts
> aren't in the image so the guide tells you to mount them, a 503 on readiness usually means the model
> is still warming, and OOM means you tried to scale with workers instead of containers. It's honest that
> there are no secrets to configure and that the image is multi-gigabyte — the runbook's job is to make
> the walls I already know about visible before an operator hits them."*
