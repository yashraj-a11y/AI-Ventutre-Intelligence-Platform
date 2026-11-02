# Day 18 · Phase A — Containerization (conceptual notes)

*Deployment phase. These notes recap why Docker is the industry standard for shipping AI systems, and
ground the concepts in **this** platform's real architecture before Day 18 packages it.*

> **Note:** Docker is **not installed in this build environment**, so Day 18 *writes and statically
> validates* the deployment artifacts (compose YAML, `.dockerignore`, script syntax, app boot via
> TestClient). `docker build` / `docker run` is the reader's / CI's step — never claimed as run here.

---

## 0. Context — a working platform that runs on one laptop

Days 1–17 built a working FastAPI platform: 10 routes, 13 layers, a warm model registry, 91 tests, CI.
But it runs in *this* virtualenv, pinned to *this* machine's Python 3.14.2, torch 2.13, faiss 1.14,
lightgbm 4.7. "Works on my laptop" is not a production system. Day 18's job is to make it run **the same
way everywhere** — which is exactly what containers provide.

---

## 1. Containers vs. virtual machines

A **VM** virtualizes *hardware*: it runs a full guest OS (its own kernel), so it's heavy (GBs), slow to
boot (seconds–minutes), and you run one app per bulky image. A **container** virtualizes the *process*:
it shares the host's kernel and isolates just the filesystem, dependencies, and process tree — so it's
light, boots in milliseconds, and packages exactly one application with its dependency closure.

For an AI system the win is **dependency reproducibility**: torch, faiss, and the exact CUDA/OpenMP
runtime that a model was validated against travel *inside* the image. The platform's whole
`OMP_NUM_THREADS=1` faiss+torch coexistence discipline is baked into the image environment, so it can't
be forgotten in production.

---

## 2. Images, containers, and the Dockerfile

- An **image** is an immutable template — the app + its dependencies, built once.
- A **container** is a running instance of an image (you can start many from one image).
- A **Dockerfile** is the recipe: base image → install deps → copy code → set the start command.

Each Dockerfile instruction adds a **layer**, and layers are cached. That's why order matters:
install dependencies (which rarely change) *before* copying source (which changes constantly), so a
code edit doesn't re-install torch. A **registry** (Docker Hub, GHCR) stores and distributes images —
build once, `pull` anywhere.

---

## 3. Why THIS platform is ONE container, not many

The draft imagines separate services — a "vector database", a caching layer, an MLflow server. But the
platform's architecture says otherwise. The model registry builds **all 13 layers as one warm stack in
a single process** (`model_loader.py`): the lightgbm stack first, the faiss retriever last, coexisting
under `OMP_NUM_THREADS=1`. **FAISS is an in-process library, not a server** — the 407 MB index is loaded
*into* the FastAPI process, not queried over a network.

So the honest container topology is a **monolith**: one image holding the API + models + FAISS + SBERT.
Splitting the 13 layers into microservices would force each to **reload the shared warm stack** —
multiplying the ~1 GB memory footprint and breaking the Day-13 "build the bundle once" discipline. One
process → one container is the *right* design here, not a compromise. The only optional side-cars are an
**nginx** reverse proxy and, for development, an **`mlflow ui`** viewer over the local sqlite store.

---

## 4. The artifact reality — small image, mounted data

A good image contains **code**, not gigabytes of data. But this platform's trained artifacts — the
calibrated model (21 MB), the **FAISS index (407 MB)**, the **embeddings (407 MB)**, `metadata.csv`
(73 MB) — are **git-ignored** (too big for git) and not in the build context. Two honest options:

- **Mount as a volume** (Compose, local): the image stays lean (code only); `artifacts/` and `data/`
  are provided at runtime. Reproducible builds, small image.
- **Bake in** (`COPY`, cloud): a self-contained multi-GB image that needs no external storage — the
  option for a cloud host with no shared volume.

The SBERT model (`all-MiniLM-L6-v2`) and the NLTK corpora aren't in the repo either, so they're fetched
at **build** time (reusing `scripts/download_nltk_data.py` + a model-warm step) — never at first request,
which would make the first user pay a cold download.

---

## 5. The honest cost — a multi-GB ML image

"Cloud-ready" is often sold as "tiny container." That's a lie for a real ML stack: `torch` +
`faiss-cpu` + `sentence-transformers` are inherently large — the dependency layer alone is **several
GB**. We shrink it honestly with a **`python:3.14-slim` base**, **CPU-only torch** (no CUDA libraries we
don't use), and a **multi-stage build** (compile deps in one stage, copy only the runtime into a clean
final stage). But it stays multi-GB — so some free cloud tiers (image-size caps) simply won't fit, and
the blueprint says so rather than pretending.

---

## 6. Containerization — the pattern

Package the *process and its exact dependency closure* into an immutable image so it runs identically in
dev, test, and prod. For this platform that means **one lean app container** (the 13-layer monolith,
FAISS in-process), **artifacts mounted** rather than copied, the **environment pinned** (Python, torch,
`OMP_NUM_THREADS=1`), and an honest acknowledgement that a real ML image is multi-GB. Docker Compose
then wires the container to its volumes, an optional nginx, and healthchecks.

> **Interview soundbite:** *"Containerizing this platform, the key architectural call was that it's one
> container, not many — the 13 layers load as a single warm stack in one process and FAISS is an
> in-process library, not a database server, so splitting it into microservices would just reload a
> gigabyte of models per service and break the build-once discipline. I keep the image lean by mounting
> the git-ignored 400-megabyte artifacts as a volume rather than copying them, and fetching the
> transformer and NLTK data at build time so the first request doesn't pay a cold download. And I'm
> honest that a torch-plus-faiss image is inherently multi-gigabyte even with a slim base and CPU-only
> torch — some free tiers won't fit it, and the blueprint says so instead of pretending the container
> is tiny."*
