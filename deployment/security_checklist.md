# Day 18 · Phase I — Production Security Checklist

*Honest done-vs-deferred. Much of the platform's hardening was built on earlier days; Day 18 adds CORS,
the nginx reverse proxy, container hardening, and this checklist. The recurring honesty point: the
platform has **no external API keys or secrets**, so "secret management" is largely N/A — don't
provision protection for secrets that don't exist.*

---

## ✅ Done

| Control | How | Where |
|---|---|---|
| **Input validation** | `CampaignRequest` is pydantic `extra="forbid"` — fabricated fields (mrr, founder_experience, …) are **422-rejected** at the boundary; dates/goal validated | Day 14 |
| **Error sanitization** | typed exception handlers return `{error, request_id}` bodies (422 / 503 / 500) — **no stack traces or internals leaked**; every response carries an `x-request-id` | Day 14 |
| **CORS** | explicit allow-list from `CORS_ORIGINS`; **never wildcard-with-credentials** (credentials enabled only when explicit origins are set); default = no cross-origin | Day 18 · Phase I |
| **Reverse proxy + rate limiting** | nginx `limit_req` (~10 r/s/IP, burst 20), `client_max_body_size 1m`, gzip | Day 18 · Phase I (`nginx.conf`) |
| **TLS-ready** | nginx 443 block (certs user-supplied) + HTTP→HTTPS redirect template | Day 18 · Phase I |
| **Non-root container** | image runs as `appuser` (uid 10001), least privilege | Day 18 · Phase B |
| **Read-only data** | `artifacts/` + `data/` mounted **`:ro`** — the container cannot mutate models/dataset | Day 18 · Phase C |
| **Dependency pinning** | `requirements.txt` fully pinned (`==`) — reproducible, no surprise upgrades | Days 1–18 |
| **No secrets to leak** | deterministic platform, no external APIs; `.env` holds only ports/paths/log-level and is **git-ignored** | Day 18 · Phase D |
| **Health-gated traffic** | `/ready` returns 503 until loaded so a proxy/LB doesn't route to a warming container | Day 18 · Phase F |

---

## ⏳ Deferred / operator-supplied (named, not pretended-built)

| Item | Why deferred | Where it belongs |
|---|---|---|
| **App-layer rate limiting** | would add a dependency (e.g. slowapi); nginx `limit_req` covers it at the edge | nginx (done) or future middleware |
| **HTTPS certificates** | certs are environment-specific (Let's Encrypt / cloud ACM) | operator supplies; nginx 443 block ready |
| **Authentication / authorization** | the API is currently open (read-only inference); auth is a Day-19 concern (founder/investor portals) | Day 19 frontend + auth layer |
| **Secret management** | **N/A — the platform has no secrets** (no LLM/external APIs) | would apply only if secrets are added |
| **WAF / DDoS protection** | infrastructure-level | cloud load balancer / CDN |

---

## Verification

- CORS allow-list behavior is TestClient-verified (no wildcard-with-credentials; default no cross-origin).
- `nginx.conf` was brace/structure-checked (nginx not installed here) — run `nginx -t` on the host.
- Input-validation (422) and error-sanitization (503 typed body) are covered by the API test suite.

> **Interview soundbite:** *"My security checklist is honest about what's built versus operator-supplied.
> Input validation and error sanitization were already there — the request schema 422-rejects any
> fabricated field and errors return typed bodies with a request id, no stack traces. Day 18 added a
> proper CORS allow-list that never does wildcard-with-credentials, nginx rate limiting and a TLS-ready
> block, a non-root container, and read-only data mounts. The honest part is that I don't invent a
> secrets manager, because the platform has no secrets — it's deterministic with no external APIs — and
> I defer auth to the frontend day rather than bolting on a half-measure. Naming what's deferred is
> itself a security practice: a checklist that claims everything is done is the insecure one."*
