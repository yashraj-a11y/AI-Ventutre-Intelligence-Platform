# Day 15 — Venture Knowledge Base, Vector DB & Semantic Retrieval (FROZEN SPEC V1.1)

**Roadmap V1.1 — Layers 2 & 3 (Knowledge Base + Intelligence Engine).**

> **🔴 READ THIS FIRST — Day 15's core is ALREADY BUILT.** The Venture Knowledge Base, SBERT
> embeddings, FAISS vector DB, semantic search, and success/failure retrieval were implemented on
> **Day 6** (committed `b525445`) and integrated into the backend on **Day 14** (`/retrieve`). This
> spec does **not** rebuild them — it **audits** the existing system and adds the genuinely-new,
> not-yet-done pieces: **retrieval evaluation, latency optimization, documentation, a notebook.**

> **Status:** FROZEN 2026-08-05 (V1.1). Build phase-by-phase on `build Phase <X>`; each phase =
> implement/audit **and** teach via Framework 1, grounded in real numbers.

---

## 0. What already exists vs. what's new (the whole point)

| Day-15 draft item | Status | Where |
|---|---|---|
| `embedding_generator.py` (SBERT) | ✅ **DONE** | Day 6 |
| `vector_database.py` (FAISS index) | ✅ **DONE** | Day 6 |
| `semantic_search.py` | ✅ **DONE** | Day 6 |
| `venture_retriever.py` (success/failure retrieval) | ✅ **DONE** | Day 6 |
| `pipeline.py` (retrieval pipeline) | ✅ **DONE** | Day 6 |
| `startup_embeddings.npy`, `faiss.index`, metadata | ✅ **DONE** | `artifacts/vector_db/` |
| `retrieval_service.py` + `/retrieve` endpoint (Phase H integration) | ✅ **DONE** | Day 14 |
| **`retrieval_evaluator.py` (Phase G — retrieval quality metrics)** | ❌ **NEW** | Day 15 |
| **Latency optimization** (the ~10s first-`/retrieve` measured on Day 14) | ❌ **NEW** | Day 15 |
| **Documentation of retrieval components (Phase I)** | ❌ **NEW** | Day 15 |
| **`notebooks/15_Vector_Retrieval.ipynb`** | ❌ **NEW** | Day 15 |

**So Day 15 = AUDIT (B–F, H) + BUILD the 4 new pieces (evaluator, optimization, docs, notebook).**

---

## 1. What changed from the draft (changelog + why)

| # | Draft said | 🔧 Frozen correction | Why |
|---|---|---|---|
| **C-1** | Build KB from `historical_startups.csv`, `founders.csv`, `industries.csv`, `funding_rounds.csv` | **DEFER/reject — these don't exist.** The KB is (already) built from the **Kickstarter processed dataset** (campaign text + category + goal + outcome). No founders / funding-rounds / separate industry datasets. | Kickstarter has no founder or funding-round data (established Days 4–14). Day 6 built the KB honestly from what exists. |
| **C-2** | Example: "SaaS healthcare startups, Seed 2023, experienced founders, recurring revenue, burn rate, weak moat" | **Reject — fabrication.** Retrieval returns **similar Kickstarter campaigns** (by title text + category), split by real **funded/failed** outcome, with **category + funding-goal benchmarks** — no founders/revenue/moat. | Same honesty discipline; retrieval can only surface what's in the KB. |
| **C-3** | Add **ChromaDB** as a persistent vector store | **Keep FAISS only (already persistent on disk).** ChromaDB = **discussion, not built** — no new dependency. | The FAISS index is already saved to `artifacts/vector_db/faiss.index`; ChromaDB adds nothing we lack. NO-NEW-DEPS. |
| **C-4** | Phase B/C/D/E/F build the modules | **AUDIT the existing Day-6 modules** (verify they load, retrieve, and produce success/failure evidence correctly) — do **not** rebuild. | They exist, work, and are committed; rebuilding risks regressing a working system. |
| **C-5** | **Phase G — Precision@K / Recall@K / MRR** | **🔑 Honest proxy metrics — there are NO ground-truth "campaign X is similar to Y" labels.** Use signals we actually have: **category-consistency@K**, **kNN outcome-prediction** (do the K neighbors' funded/failed labels predict the query's true label? → precision/recall/F1/accuracy against the *real* target), **MRR to first same-outcome (or same-category) neighbor**, **similarity-score distribution**, **self-retrieval sanity**. | Precision@K needs relevance ground truth we don't have. The kNN-outcome metric is legitimate — the query's `target` label IS ground truth, so "do neighbors predict it" is a real, honest retrieval-quality measure. |
| **C-6** | (implied) retrieval is fast | **Measure real per-query latency first** (Day-14 saw ~10s on first `/retrieve`). Optimize what's real: warm SBERT is already in the registry, so per-query cost should be ~encode-one-short-text + sub-ms FAISS; if slow, the bottleneck is query encoding — cache/batch, or confirm the 10s was one-time model warmup. | The roadmap's title promises *optimization*; do it on measured numbers, not assumptions. |

**IMPROVEMENTS:** IMP-1 evaluator uses the **test split** (never train) for honest held-out retrieval quality. IMP-2 evaluation report as JSON artifact (metrics-as-data). IMP-3 deterministic (fixed seed for the query sample). IMP-4 latency measured warm (registry-loaded SBERT), reported per-query p50/p95.

---

## 2. Honesty tiers

**Tier 1 — REAL:** campaign title text + `name_clean`, `main_category`/`category`, `usd_goal_real`,
`target` (funded/failed), SBERT embedding, FAISS cosine similarity, category cohort funding rates.
**Tier 2 — DERIVED (labeled):** retrieval-quality proxy metrics (category-consistency, kNN
outcome accuracy, MRR-proxy), similarity distributions, latency stats. **Tier 3 — DEFERRED, never
fabricate:** founder embeddings/summaries, funding-round history, revenue/retention/moat, "similar
Seed rounds", named companies, business-model similarity.

---

## 3. Phases (build order) — audit vs new

- **A** — Semantic retrieval notes (concepts): keyword vs semantic, embeddings, cosine similarity,
  ANN/FAISS, retrieval-augmented intelligence — **recap + ground in the existing Day-6 system**.
  `docs/day15_phaseA_semantic_retrieval_notes.md`.
- **B–F** — **AUDIT (no rebuild):** load the Day-6 retriever; verify embeddings + FAISS index +
  metadata are present and consistent (counts match), semantic search returns top-K with scores,
  `venture_retriever` splits success/failure + benchmarks correctly, and the honest reframe (no
  founders/funding-rounds) holds. Record findings; fix only genuine defects.
- **G** — **`src/retrieval/retrieval_evaluator.py` `RetrievalEvaluator` (NEW):** on the test split,
  compute category-consistency@K, kNN-outcome precision/recall/F1/accuracy vs real `target`, MRR-proxy,
  similarity distribution, self-retrieval sanity, for K∈{5,10}. → `artifacts/retrieval/retrieval_evaluation.json`.
  Config `day15.evaluation`.
- **H** — **Verify Day-14 integration** (`/retrieve` → `retrieval_service` → `VentureRetriever`) and,
  optionally, confirm retrieved evidence already flows to reports/copilot (it does via the frame cohort;
  note the distinction: Day-13 reports use the **frame cohort**, `/retrieve` uses **semantic FAISS** —
  both honest, different mechanisms).
- **Optimization** — measure warm per-query latency (SBERT encode + FAISS search); if the ~10s
  persists warm, profile + optimize (query-embedding cache, batch, FAISS `nprobe`); else document that
  it was one-time model warmup.
- **I** — **Documentation:** per retrieval module — purpose / inputs / outputs / algorithm / time
  complexity / storage format / example / assumptions / limitations. `docs/day15_retrieval_reference.md`.
- **Notebook** — `notebooks/15_Vector_Retrieval.ipynb`: one query → embedding → FAISS search → similar
  successes/failures + benchmarks → evaluation metrics → latency; honesty boundary shown.

---

## 4. Environment & repo

- **NO NEW DEPS** (faiss-cpu + sentence-transformers + torch already installed; no ChromaDB).
  **🔴 OMP_NUM_THREADS=1** — Day-15 work loads faiss + SBERT/torch (run
  `OMP_NUM_THREADS=1 PYTHONPATH=. ./.venv/bin/python`), same isolation as Day 6.
- **New files:** `src/retrieval/retrieval_evaluator.py`, `notebooks/15_Vector_Retrieval.ipynb`,
  `docs/day15_{phaseA_semantic_retrieval_notes,retrieval_reference}.md`, config `day15.evaluation`,
  `artifacts/retrieval/retrieval_evaluation.json`. **No changes** to the working Day-6 retrieval modules
  unless the audit finds a defect.
- **Input contract:** `VentureRetriever.retrieve(campaign, k) → {query, similar[], similar_successful[],
  similar_failed[], funding_benchmark, category_benchmark, deferred_benchmarks}`; the processed dataset
  + `artifacts/splits/test_index.npy` for held-out evaluation.

---

## 5. Non-negotiables (frozen)

1. **Don't rebuild Day 6** — audit the existing retrieval system; only fix real defects (C-4).
2. **No fabricated data sources** — KB is Kickstarter campaigns; founders/funding-rounds/revenue deferred (C-1/2).
3. **Honest evaluation** — proxy metrics against real labels (category, `target`), never Precision@K vs nonexistent similarity ground truth (C-5).
4. **No new deps, no ChromaDB** (C-3); OMP_NUM_THREADS=1 for the faiss+torch process.
5. **Evaluate on the test split only**; deterministic; metrics-as-data JSON (IMP-1/2/3).
6. Teach every phase via Framework 1, grounded in real numbers; end with an interview soundbite.

---

*Frozen 2026-08-05. Supersedes the Day-15 draft. Day 15 = audit Day-6 retrieval + build evaluator/optimization/docs/notebook.*
