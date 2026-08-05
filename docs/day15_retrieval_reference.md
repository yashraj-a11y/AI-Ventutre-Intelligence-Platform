# Day 15 · Phase I — Retrieval Component Reference

*Per-module reference for Layers 2 & 3 (Knowledge Base + Semantic Retrieval). The stack was built on
**Day 6** (committed `b525445`), served over the API on **Day 14** (`/retrieve`), and audited +
evaluated + latency-measured on **Day 15**. All numbers below are **real, measured** on the running
system, not illustrative.*

---

## 0. System at a glance

| Property | Value (measured) |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (Sentence-Transformers) |
| Embedding dim | **384** |
| Corpus size | **265,171** Kickstarter campaigns (the **train** split) |
| Normalization | L2-normalized → ‖v‖ = 1.0 (cosine == inner product) |
| Vector store | FAISS **`IndexFlatIP`** — exact inner product, **not** ANN |
| Index size on disk | **407 MB** (`artifacts/vector_db/faiss.index`) |
| Model load (cold) | ~0.9 s (paid once at registry startup) |
| First `/retrieve` (cold) | **~11 s** — one-time SBERT/torch warmup |
| Warm per-query | **p50 14 ms / p95 16 ms** end-to-end (encode ~4.4 ms + FAISS ~6.4 ms) |
| Retrieval quality | category-consistency **8.0× lift**, kNN-outcome ROC-AUC **0.66–0.68** |

**Data flow:** `campaign → serialize → SBERT encode (384-d, normalized) → FAISS IndexFlatIP search →
top-K rows → row→campaign_id mapping → metadata join → split by funded/failed + benchmarks`.

**🔴 Run constraint:** every entry point that loads FAISS + SBERT/torch must run with
`OMP_NUM_THREADS=1` (OpenMP collision between faiss and torch). The registry additionally enforces
**lightgbm-loaded-before-faiss** so the Days 4–13 stack and retrieval coexist in one process.

---

## 1. `EmbeddingGenerator` — text → 384-d vectors
[src/retrieval/embedding_generator.py](../src/retrieval/embedding_generator.py)

- **Purpose:** wrap `all-MiniLM-L6-v2` to turn campaign text into dense, L2-normalized 384-d vectors.
- **Inputs:** `encode(texts: list[str]) → np.ndarray (n, 384)`; config `day6.embeddings` (`model_name`,
  `out_dir`, `device`).
- **Outputs:** `float32` array, each row unit-norm (‖v‖ = 1.0). `save()` persists
  `startup_embeddings.npy` + `embedding_model.json` (name / library / dim, for provenance).
- **Algorithm:** transformer forward pass + mean-pooling (Sentence-Transformers) → L2 normalize.
- **Complexity:** O(n · L) in tokens; batched. Warm single-text encode ≈ **4.4 ms** (measured).
- **Storage:** `artifacts/embeddings/startup_embeddings.npy` (265171 × 384 float32 = **407 MB**),
  `embedding_model.json`.
- **Example:** `EmbeddingGenerator().encode(["Artificer — a board game"]).shape → (1, 384)`.
- **Assumptions:** train/serve use the **same** model + serialization; normalization on.
- **Limitations:** embeds **title text only** — not goal/country/dates. So outcome signal that lives
  in the goal is **not** in the vector (this is why kNN-outcome AUC is a modest ~0.66, honestly).

---

## 2. `VectorDatabase` — FAISS exact-cosine index + row→campaign mapping
[src/retrieval/vector_database.py](../src/retrieval/vector_database.py)

- **Purpose:** own the FAISS index and the **row→campaign_id** mapping (FAISS returns integer rows;
  the mapping records that row *i* ↔ `campaign_id`, traceable to the processed dataset).
- **Inputs:** `build(embeddings)`, `search(queries: np.ndarray, k) → (D, I)`, `campaign_id(row) → int`,
  `n_vectors` (**@property, not a method** — no parens).
- **Outputs:** `search` returns `(D, I)` — `D` = cosine similarities (descending), `I` = row ids.
- **Algorithm:** `faiss.IndexFlatIP` — exact brute-force inner product. Because vectors are
  normalized, inner product **is** cosine. **Deliberately exact, not ANN** (see §7).
- **Complexity:** search is **O(N · d)** per query (N=265,171, d=384). Measured: **~6.4 ms** per
  single query single-threaded (`OMP_NUM_THREADS=1`); **<1 ms/query batched** (BLAS matmul).
- **Storage:** `artifacts/vector_db/faiss.index` (**407 MB**), `mapping.json` (row→campaign_id).
- **Example:** `db = VectorDatabase().load(); D, I = db.search(qvec, k=5); db.campaign_id(I[0,0])`.
- **Assumptions:** query vectors normalized + same dim (384); index is the train split only.
- **Limitations:** exact search is **linear** in N — fine at 265K, would need ANN at tens of millions.

---

## 3. `SemanticSearchEngine` — query → ranked neighbours
[src/retrieval/semantic_search.py](../src/retrieval/semantic_search.py)

- **Purpose:** the query-side glue — serialize a raw campaign the **same way** KB records were
  serialized, embed it, search, and join FAISS rows back to human-readable metadata.
- **Inputs:** `serialize(campaign) → str`, `embed_query(query) → 384-d`,
  `search(query, k) → list[dict]`, `similar_to_id(campaign_id, k)`.
- **Outputs:** ranked list of `{rank, similarity, campaign_id, title, category, main_category}`,
  similarity descending.
- **Algorithm:** `serialize` reconstructs the KB text (title + goal_band + launch_year + category via
  `_kb.to_text`) so a query and a stored record are **identically represented** (train/serve
  consistency); then encode → `VectorDatabase.search` → metadata join.
- **Complexity:** one encode (~4.4 ms) + one FAISS search (~6.4 ms) + O(k) join.
- **Storage:** none of its own; reads `metadata.csv` (per-vector records) aligned to index rows.
- **Example:** `engine.search({"name": "Dungeon Forge", "main_category": "Games", ...}, k=5)`.
- **Assumptions:** metadata row order **matches** FAISS row order (verified in Phase B/E audit).
- **Limitations:** relevance == embedding proximity; no re-ranking, no query expansion.

---

## 4. `VentureRetriever` — retrieval-augmented evidence bundle
[src/retrieval/venture_retriever.py](../src/retrieval/venture_retriever.py)

- **Purpose:** turn neighbours into **decision evidence** — similar campaigns split by real outcome,
  plus benchmarks, with an explicit **deferred** boundary.
- **Inputs:** `retrieve(query, k) → bundle`; also `similar`, `similar_successful`, `similar_failed`.
- **Outputs:** `{query, similar[], similar_successful[], similar_failed[], funding_benchmark,
  category_benchmark, deferred_benchmarks}`. Each item carries `outcome`/`target`/`text`/`reason`.
- **Algorithm:** search a pool, split by the **real** funded/failed `target`, compute category +
  funding-goal benchmarks from the cohort, and mark founder/market/valuation/exit as **deferred**.
- **Complexity:** one search + O(pool) partition.
- **Storage:** none; composes the engine + `metadata.csv`.
- **Example:** `VentureRetriever().retrieve(campaign, k=5)["similar_failed"][0]["outcome"] → "failed"`.
- **Assumptions:** `target` is the ground-truth outcome label; benchmarks are cohort aggregates.
- **Limitations (honesty boundary):** can surface **only** what's in the KB — Kickstarter title text,
  category, goal, outcome. **No** founders, funding rounds, revenue/retention, or moat — those are in
  `deferred_benchmarks`, never fabricated.

---

## 5. `RetrievalEvaluator` — honest retrieval-quality metrics *(NEW, Day 15 · Phase G)*
[src/retrieval/retrieval_evaluator.py](../src/retrieval/retrieval_evaluator.py)

- **Purpose:** *measure* retrieval quality — Day 6 built the retriever but never evaluated it. There
  are **no ground-truth "campaign X ~ Y" similarity labels**, so Precision@K would be fabrication;
  this evaluates against labels we actually have.
- **Inputs:** `evaluate(n) → dict`, `generate_report()`; config `day15.evaluation` (`k_values=[5,10]`,
  `n_queries=1500`, `random_state=15`).
- **Outputs (→ `artifacts/retrieval/retrieval_evaluation.json`):**
  - **category-consistency@K** — frac of top-K sharing the query category, as **lift** over base rate.
  - **kNN outcome-prediction** — do neighbours' funded/failed labels predict the query's **real**
    `target`? accuracy / precision / recall / F1 / **ROC-AUC** + lift over the majority baseline.
  - **MRR (category)**, similarity distribution.
- **Algorithm:** sample 1500 **TEST**-split campaigns as held-out queries → batch-encode once → one
  FAISS search → metrics vs real labels. Deterministic (fixed seed).
- **Complexity:** one batched encode + one batched search over n queries (~15 s for 1500).
- **Measured results:** category-consistency **79%@5 vs 9.9% base = 8.0× lift**; kNN ROC-AUC
  **0.658@5 → 0.678@10**; accuracy 0.65–0.66 > 0.617 majority; MRR 0.886.
- **Assumptions:** the query's own `target` **is** ground truth (legitimate); test ∩ train = ∅.
- **Limitations:** proxy metrics, not human relevance judgements; AUC is honestly modest (~0.66)
  because the funding-determining feature (**goal**) is not in the title embedding.

---

## 6. `RetrievalLatencyBenchmark` — measure-before-optimize *(NEW, Day 15 · Optimization)*
[src/retrieval/latency_benchmark.py](../src/retrieval/latency_benchmark.py)

- **Purpose:** answer the Day-14 "~10s first `/retrieve`" question with data, not assumption (C-6):
  is it a per-query cost or one-time warmup?
- **Inputs:** `measure() → dict`, `generate_report()`; config `day15.optimization` (`n_warm=200`,
  `n_decompose=50`, `k=5`).
- **Outputs (→ `artifacts/retrieval/retrieval_latency.json`):** `cold_first_query_ms`, warm
  `p50/p95/mean`, `decomposition_ms` (encode vs FAISS), `cold_over_warm_ratio`, `verdict`.
- **Algorithm:** time the first call (cold) separately from a warm stream; decompose warm into
  encode-only vs search-only. Deterministic sample from the TEST split.
- **Measured results:** cold **~11 s**, warm **p50 14 ms / p95 16 ms**, ratio **784×** → the ~10 s was
  **100% one-time SBERT warmup**, already paid at registry startup. Decomposition: encode 4.4 ms +
  FAISS 6.4 ms.
- **Verdict:** **no FAISS bottleneck** at 265K×384 — optimization is a no-op beyond the startup warmup
  already in place; ANN would only trade recall for speed we don't need.
- **Limitations:** single-process, CPU, `OMP_NUM_THREADS=1`; absolute ms are machine-specific (the
  **ratios** and decomposition are the portable findings).

---

## 7. Cross-cutting design decisions (the honest defaults)

1. **Exact, not approximate.** `IndexFlatIP` gives ground-truth top-K with **zero recall loss**. ANN
   exists to make search sub-linear; at 265K×384 exact search is ~6 ms single-query / <1 ms batched —
   already fast enough. ANN is a **future scaling** move (tens of millions of vectors), noted only.
2. **Normalize → cosine is inner product.** Storing unit vectors lets FAISS's inner-product index
   compute cosine directly, no division.
3. **Train/serve serialization parity.** A query is serialized **identically** to a KB record, or
   proximity would be measured in two different text spaces.
4. **Honesty boundary as data.** `deferred_benchmarks` names what the KB can't surface (founders,
   funding rounds, revenue, moat) rather than inventing it.
5. **OpenMP + load order.** `OMP_NUM_THREADS=1` and lightgbm-before-faiss let the whole platform run
   in one process (verified Day 14 + Day 15 Phase H).

---

## 8. API surface & the two evidence mechanisms

- **`POST /retrieve`** → [src/api/routes/retrieve.py](../src/api/routes/retrieve.py) →
  [src/services/retrieval_service.py](../src/services/retrieval_service.py) → `VentureRetriever`.
  Returns `RetrievalResponse` (`query`, `similar`, `similar_successful`, `similar_failed`,
  `honesty_note`). Query param `k ∈ [1, 25]`, default 5.
- **Two honest, distinct mechanisms** (don't conflate them):
  - **`/retrieve`** uses **semantic FAISS** — the actual nearest campaigns by title embedding.
  - **Day-13 reports** use the **frame cohort** — the *category* `cohort_funding_rate` vs
    `platform_base_rate` (an aggregate, no FAISS call). Same honest data, different question.

---

*Phase I complete. Reference for the Day-6 retrieval stack + the two Day-15 additions
(`RetrievalEvaluator`, `RetrievalLatencyBenchmark`). Numbers measured 2026-08-05.*
