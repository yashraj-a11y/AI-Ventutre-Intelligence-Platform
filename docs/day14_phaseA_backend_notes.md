# Day 14 · Phase A — Backend Architecture (conceptual notes)

*Production phase begins. No server code — the "why" before the "how".*

---

## 1. The problem: 13 layers, no product

By the end of Day 13 the platform is complete — but it's a *library*, not a *service*. Every
capability lives behind a Python import: to get a prediction you write code, load config, construct a
pipeline. Another application (a web frontend, a partner system, a notebook) can't consume it without
becoming a Python program that reaches into `src/`. A real product exposes its capabilities over a
**network boundary** — a client sends a request, gets JSON back, and never touches the internals.

**Day 14 wraps the platform in a backend**: a single FastAPI service that turns 13 layers of Python
into an HTTP API. This is the step that makes the platform *deployable and consumable*.

---

## 2. Client–server & REST — the contract

The backend interposes a **contract** between the caller and the code:

```
Client  --HTTP POST /analyze {campaign json}-->  FastAPI  --JSON response-->  Client
```

- The client speaks **HTTP + JSON**, not Python. It's decoupled from our implementation language,
  our imports, our file layout.
- **REST** maps *resources/actions* to HTTP verbs + paths: `POST /predict`, `POST /analyze`,
  `GET /health`. A POST with a JSON body in, a JSON body out.
- The contract is **typed** (Phase C, pydantic): the request is validated at the boundary, so a
  malformed campaign is rejected with a clear `422` *before* any model runs.

The value: the platform becomes a black box with a stable, documented, validated interface —
consumable by anything that speaks HTTP.

---

## 3. Layered backend architecture (separation of concerns)

Production backends are layered so each concern is isolated and testable:

```
route (HTTP)   →  validate the request, call one service, shape the response   (knows HTTP, not ML)
service        →  business logic: call the existing pipeline, return a dict     (knows ML, not HTTP)
model registry →  load artifacts ONCE at startup, hold the warm stack           (knows loading)
ML pipelines   →  Days 4-13 (unchanged)                                         (knows the domain)
```

- **Routes stay dumb** — validate → call → respond. No ML logic leaks into HTTP handlers.
- **Services stay web-agnostic** — they take the registry and return plain dicts, so they're unit-
  testable *without* spinning up a server.
- **The registry loads once** — not per request (§5).

This is the classic **service-layer** pattern: the API and the domain never depend on each other
directly; the service is the seam.

---

## 4. The defining constraint: reuse, don't rebuild

The single most important design decision for Day 14: **the services are thin wrappers over the
pipelines that already exist.** `/analyze` doesn't re-orchestrate anything — it calls the Day-13
`ReportingPipeline`. `/predict` calls the Day-4 `PredictionEngine`. `/retrieve` calls the Day-6
`VentureRetriever`. The API layer adds **zero** new intelligence; it adds a *network surface* over
audited, deterministic pipelines.

Why this matters: those pipelines are already built, tested (44/44, 49/49 audits), faiss-safe, and
honest. Re-implementing their logic in the API would duplicate ~thousands of lines, invite drift, and
throw away the integrity guarantees. The backend's job is **exposure and orchestration**, not
analysis — the mirror of Layer 13's "presentation, not analysis," now at the transport layer.

---

## 5. Dependency injection + the singleton registry

Loading the model, the SHAP explainer, the item frame, and the faiss index is *expensive* (seconds).
Doing it **per request** would make every call unusably slow and waste memory. So:

- **Load once, at startup** — FastAPI's *lifespan* hook builds one `ModelRegistry` when the server
  boots, holding the warm stack in app state.
- **Inject it** — routes/services receive the registry via FastAPI's **dependency injection**, not by
  importing a global. DI keeps the dependency explicit and makes services trivially testable (pass a
  fake registry).

This is the **singleton** pattern done cleanly: one instance, created at a well-defined time,
provided through the framework rather than a module global.

---

## 6. The runtime landmine: OpenMP + faiss (verified)

The one thing that can silently break this backend. The lightgbm stack (Days 4-13) and Day-6's
faiss + SBERT/torch **both use OpenMP**, and loading them naively in one process **segfaults or
hangs** on the first faiss call — the exact hazard that forced Day-6 to run isolated during
whole-platform verification. A single FastAPI process must host both (for `/analyze` *and*
`/retrieve`), so the mitigation is mandatory and **preflight-verified**:

1. **`OMP_NUM_THREADS=1` set at the very top of the entrypoint — before any import** (numpy, lightgbm,
   faiss, torch all read it at import time).
2. **Load order: the lightgbm stack before the faiss retriever** (the model registry enforces this).

With both in place, `/analyze` (lightgbm) and `/retrieve` (faiss) were run interleaved in one process
with no crash. This is non-negotiable #1 for every later phase.

---

## 7. Honest I/O at the boundary

The API preserves the platform's honesty. The **request schema accepts only real Kickstarter fields**
(name, category, goal, launched, deadline, country) — the draft's "founder profile / financial
metrics / pitch-deck metadata" are *rejected* at the boundary, because the platform never modeled
them. The **response schemas surface the real outputs plus their deferral appendices and honesty
notes** — the API never invents a field the pipelines didn't produce. Validation is where honesty
becomes enforced: fabricated inputs `422` before they can mislead.

---

## 8. Backend architecture — the pattern

Day 14 is the **API-gateway / service-layer** pattern: a thin, typed, network-facing shell over a
warm, singleton-loaded, reused domain core, with strict separation between HTTP concerns and ML
concerns. It computes nothing new; its value is *exposure, validation, orchestration, and lifecycle*
(load once, inject, handle errors) — turning a Python library into a consumable product.

> **Interview soundbite:** *"Day 14 turns the thirteen-layer library into a single FastAPI service.
> The architecture is a classic service layer: dumb routes that validate and respond, thin services
> that wrap the existing pipelines and stay web-agnostic, and a singleton model registry loaded once
> at startup via the lifespan hook and injected by dependency injection — never per request. The
> defining decision is reuse: the API adds zero new intelligence, it just exposes audited pipelines
> over HTTP, so /analyze literally calls the Day-13 reporting pipeline. The one real landmine is
> OpenMP — the lightgbm stack and the faiss retriever both use it and collide in one process, so I
> set OMP_NUM_THREADS to one before any import and load lightgbm before faiss, a mitigation I
> preflight-verified by running analyze and retrieve interleaved without a crash. And the request
> schema accepts only real campaign fields, so fabricated founder or financial inputs are rejected at
> the boundary — honesty enforced by validation."*
