# Deployment

A summary and map. The operator runbook is [deployment/deployment_guide.md](../deployment/deployment_guide.md);
this page does not duplicate it.

## Honest status

| | |
|---|---|
| Dockerfile, Compose, nginx, scripts, env config, probes, security checklist | ✅ written (Day 18) |
| Application hardening (logging, CORS deny-by-default, health probes) | ✅ implemented and covered by API tests |
| **`docker build` / `docker compose up`** | ❌ **never run** — the development machine has no Docker |
| Cloud deployment | ❌ blueprint only |

Treat the container setup as reviewed configuration, not a proven image. Expect to fix small issues the
first time you build it.

## Shape

**One container** (`avip-api`): FastAPI plus all 13 layers, FAISS in-process. Trained artifacts and data
are **mounted read-only**, not baked in, which keeps the image lean (the artifacts are ~900 MB). The
image is still multi-GB because of torch and FAISS; some free tiers will not fit it.

```bash
cp .env.example .env     # optional — all values have defaults, there are no secrets
./scripts/build.sh       # build avip-api:latest
./scripts/run.sh         # docker compose up -d (WITH_PROXY=1 adds nginx, WITH_MLFLOW=1 adds the MLflow UI)
curl localhost:8000/ready
./scripts/stop.sh
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | 8000 | API port |
| `WORKERS` | 1 | keep at 1 — each worker loads the ~1 GB stack; scale by containers |
| `LOG_LEVEL` / `LOG_FILE` | INFO / `logs/application.log` | structured logging |
| `DEPLOY_MODE` | production | development / staging / production |
| `ARTIFACTS_DIR` / `DATA_DIR` | `artifacts` / `data` | mounted read-only |
| `CORS_ORIGINS` | *(empty)* | allow-list for a browser frontend; empty = no cross-origin |

## Health probes

`GET /live` (process alive) · `GET /ready` (`200` when loaded, else `503`) · `GET /health` (status plus
disk / memory / CPU diagnostics).

## Serving the frontend

`npm run build` in `frontend/` produces a static `dist/`. Serve it from any static host or the nginx
profile, and set the backend's `CORS_ORIGINS` to that origin (or put both behind one reverse proxy).

## Not included (deferred)

Real authentication / JWT, TLS termination, autoscaling, a managed artifact store, and metrics
dashboards. [deployment/cloud_architecture.md](../deployment/cloud_architecture.md) discusses the
artifact problem in the cloud and platform options;
[deployment/security_checklist.md](../deployment/security_checklist.md) lists done versus deferred.
