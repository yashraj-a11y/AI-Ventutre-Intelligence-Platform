# 🚀 Day 6 — Venture Knowledge Base & Semantic Retrieval (Roadmap V1.1 — FROZEN)

> **Status: FROZEN engineering contract** (2026-07-29). Corrections from the draft flagged 🔧;
> changelog + improvements + honesty tiers at the end. Implements Architecture **Layer 2
> (Venture Knowledge Base)** and **Layer 3 (Venture Intelligence Engine)**. Introduces
> sentence embeddings (SBERT), a vector database (FAISS), cosine similarity, and
> retrieval-augmented venture intelligence.

## Goal
Give the platform **historical memory**: store past campaigns as semantic embeddings, and
retrieve — for any query campaign — the most similar past campaigns, their real funding
outcomes, and category/funding benchmarks, as **evidence** that sits *alongside* the Day-4
prediction and Day-5 explanation. Not a replacement for the model — a second, retrieval-based
line of evidence.

---

## Data / model reality (Source of Truth) 🔧
The dataset is **Kickstarter campaigns**, not a VC knowledge graph. It has **12 columns**:
`name` (title, ~5.6 words), `category` (159), `main_category` (15), `currency`, `deadline`,
`goal`, `launched`, `state`, `country` (22), `usd_goal_real`, `name_clean`, `target`
(1=funded). **331,464 decided campaigns** (197,614 failed / 133,850 successful, 40.4% base rate).

**What we can honestly build a knowledge base from:** title text, category/main_category,
country, currency, goal/usd_goal_real, launch timing, and the **known outcome** (funded/failed).

**What the draft asked for that DOES NOT EXIST — DEFER, never fabricate** 🔧 (same discipline as
Day-4 venture scores and Day-5 copilot honesty tiers): startup **descriptions**, **pitch
summaries**, **founder backgrounds / summaries**, **funding rounds**, **valuation**, **exit
status/history**, **business model**, **team size**, **financial metrics**, **market reports**,
**accelerator data**. There is no founder/team/market/valuation signal in this data. Any
"similar founders", "founder benchmarks", "valuation history", or "exit history" retrieval is
**fabrication** and is deferred to a "requires richer data" placeholder.

### Non-obvious FROZEN decisions (easy to get wrong)
- **We embed a SERIALIZED VENTURE RECORD, not a "description"** 🔧. A bare 6-word title carries
  weak semantic signal; most real similarity is structural. So each campaign becomes a natural-
  language record — e.g. *"A Tabletop Games project (category: Games) based in the US, seeking
  about 5,000 USD."* — built from title + category + main_category + country + goal band, and
  **that** is embedded. Be transparent this is **hybrid** (title semantics + structured facts),
  not rich-description retrieval. (Fix `a/an` before vowels, as in Day-5 Phase G.)
- **Knowledge base = TRAIN split only** (`artifacts/splits/train_index.npy`, ~265k) 🔧. The KB is
  *historical* knowledge with **known outcomes**; test/new campaigns query it. This keeps
  "evidence supports the prediction" **non-circular** (the query's outcome is never in the KB it
  queries — and for a train-set query we **exclude self**, else it returns itself at cosine 1.0).
- **FAISS exact flat, not ANN** 🔧. At 265k × 384 dims, `IndexFlatIP` on **L2-normalized**
  vectors = exact cosine, instant, zero recall loss. Implement flat; **teach** IVF/HNSW ANN as
  the scale-up path (millions of vectors) — don't take approximate error we don't need.
- **Do NOT pickle the SBERT model** (`embedding_model.pkl` in the draft is wrong) 🔧. It's a
  torch model; persist the **model name + version + dim** in JSON and reload from the
  sentence-transformers cache. Artifact renamed `embedding_model.json`.
- **Evidence is grounded in REAL retrieved facts** 🔧. Never "weak retention" / "experienced
  founding team" (don't exist). Evidence = "of K neighbors: N funded (X%), median goal $Y,
  dominant category Z, dominant country C" — all computed from retrieved records' real columns.
- **Cosine via normalized inner product** — normalize embeddings once, use `IndexFlatIP`;
  similarity ∈ [−1, 1], typically [0, 1] for this corpus.

## Environment (resolved) 🔧
Installed + verified in `.venv`: **faiss-cpu 1.14.3**, **sentence-transformers 5.6.1**,
**torch 2.13.0**. Model **`all-MiniLM-L6-v2`** (384-dim) downloads cleanly from HuggingFace (no
SSL issue), embeddings L2-normalized. **Encode speed ~1,900 records/s → full 265k train in
~2.3 min** on CPU (a configurable `subsample` supports fast dev; full encode is a short
foreground or background run). Pin these three in `requirements.txt`. Always run Day-6 code with
`PYTHONPATH=. ./.venv/bin/python`.

---

## Phase A — Understanding Semantic Retrieval
**Deliverable:** semantic-retrieval notes (`docs/day6_phaseA_semantic_retrieval_notes.md`):
keyword vs semantic search, embeddings, vector spaces, cosine similarity, dense vs sparse
(contrast with Day-3 **TF-IDF** = sparse/lexical), ANN concept, retrieval-augmented intelligence,
and **why retrieval complements (not replaces) the Day-4 model**. Grounded in real numbers.

## Phase B — Venture Knowledge Base → `src/knowledge_base/venture_database.py`
**Tasks:** define the structured schema over **real columns only**; build records from the
**train split**; serialize each into a venture-record string (Phase-C input); keep a parallel
metadata table (category, country, goal, outcome) for evidence/filtering.
**🔧 Honesty:** schema fields limited to what exists; founder/valuation/exit fields **omitted**
(not stored as empty).
**Deliverables:** `VentureKnowledgeBase` (build / records / metadata / to_text).

## Phase C — Embedding Generation → `src/retrieval/embedding_generator.py`
**Model:** SBERT `all-MiniLM-L6-v2` (primary, 384-dim, lightweight); BERT/larger SBERT discussed.
Contrast **TF-IDF (sparse, lexical)** vs **SBERT (dense, semantic)**.
**Tasks:** encode venture-record strings in batches (`batch_size≈256`), **L2-normalize**, persist
`startup_embeddings.npy` (float32) + `embedding_model.json` (name/version/dim). Configurable
`subsample` for dev.
**🔧** Pitch/founder-summary embeddings deferred (no such text). **Deliverables:** `EmbeddingGenerator`.

## Phase D — Vector Database → `src/retrieval/vector_database.py`
**Index:** FAISS **`IndexFlatIP`** on normalized vectors (exact cosine). Topics: vector indexing,
ANN (IVF/HNSW — **discussed** as scale-up), FAISS (used), Chroma/Pinecone/Milvus (discussed).
**Tasks:** build index; keep `mapping.json` (row → campaign metadata); `save`/`load`
(`faiss.index` + mapping). **Deliverables:** `VectorDatabase`.

## Phase E — Semantic Search Engine → `src/retrieval/semantic_search.py`
**Tasks:** embed a query campaign (same serialization) → FAISS top-K → return neighbors ranked by
**cosine**, with **self-exclusion** and optional metadata filters (e.g. same main_category).
**🔧** "Similar founders/industries" reframed to **similar campaigns** + category/country facets
(real). **Deliverables:** `SemanticSearchEngine`.

## Phase F — Historical Venture Retriever → `src/retrieval/venture_retriever.py`
The core of Layer 3. **Retrieve (SUPPORTED):** similar campaigns, similar **successful**, similar
**failed** (filter neighbors by real `target`), **funding benchmarks** (goal/usd_goal_real stats
of neighbors), **category/industry benchmarks** (success rate + goal distribution by
main_category). **DEFER (no data):** founder benchmarks, market benchmarks, valuation/exit 🔧.
**Each result carries real fields:** similarity, funded?, goal, category, country — and a **reason
built from shared real attributes** ("same category, similar goal band, same country"), never
traction/founder profile. **Deliverables:** `VentureRetriever`.

## Phase G — Retrieval Evidence Generator → `src/retrieval/evidence_generator.py`
Convert retrieval into structured, **honest** evidence: "Retrieved K similar campaigns: N funded
(X%); median goal $Y; dominant category Z; dominant country C." 🔧 Replace the draft's
retention/founding-team reasons with real aggregates. **Improvement:** expose the neighbor
**success rate as a retrieval-based prior** — a k-NN estimate of P(fund) shown *next to* the
Day-4 calibrated probability (agreement = corroborating evidence; divergence = flagged).
**Honesty caption** on every evidence pack: "historical outcomes of similar past campaigns —
context for, not proof of, this campaign's result." **Deliverables:** `EvidenceGenerator`.

## Phase H — Venture Intelligence Pipeline → `src/retrieval/pipeline.py`
Chain: `query campaign → serialize → embed → FAISS search → similar / success / failure /
benchmarks → evidence package`. One warm embedder + one loaded index shared across calls.
**🔧 Cross-layer:** optionally attach the Day-4 `PredictionEngine` probability so a query returns
**prediction + retrieved evidence** together (the "evidence supports this prediction" story,
non-circular because KB is train-only). **Deliverables:** `VentureIntelligencePipeline`.

## Phase I — Knowledge Base Serialization → `src/retrieval/serialization.py`
Persist + reload: embeddings (`.npy`), FAISS index (`faiss.index`), metadata/mapping (`.json`),
`embedding_model.json` (name/version/dim), retrieval config. **SHA-256** of embeddings + index +
model descriptor for version identity (as in Day-4 serialization). `.verify()` re-hashes.
**Deliverables:** reusable, versioned knowledge base.

---

## Validity check (the "measure, don't assume" habit — Improvement) 🔧
Before trusting the KB, **prove the embedding space is meaningful for venture retrieval**:
- neighbors share `main_category` far above the random base rate;
- neighbor success-rate (k-NN prior) correlates with true outcome better than the global 40.4%
  base rate (a cheap retrieval-AUC);
- honestly report if serialized-record embeddings beat a trivial structured-only baseline —
  if titles add little over metadata, **say so** (echoes Day-4 L1-vs-L2 "no gain" honesty).

---

## Repository Structure After Day 6
```
src/knowledge_base/  venture_database.py
src/retrieval/       embedding_generator.py  vector_database.py  semantic_search.py
                     venture_retriever.py    evidence_generator.py  pipeline.py  serialization.py
notebooks/           06_Venture_Knowledge_Base.ipynb
```

## Artifacts After Day 6 🔧
```
artifacts/embeddings/  startup_embeddings.npy   metadata.json   embedding_model.json   (NOT .pkl)
artifacts/vector_db/   faiss.index              mapping.json
artifacts/retrieval/   retrieval_examples.json  evidence_examples.json  benchmark_report.json
```

## Deliverables Checklist
✅ venture_database.py · embedding_generator.py · vector_database.py · semantic_search.py ·
venture_retriever.py · evidence_generator.py · pipeline.py · serialization.py ·
06_Venture_Knowledge_Base.ipynb · startup embeddings · FAISS vector DB · semantic retrieval engine ·
**honest** historical evidence generator (real-outcome aggregates + deferred tiers)

## Curriculum / Architecture
Layer 2 (Knowledge Base) + Layer 3 (Intelligence Engine) ✅ · New techniques: sentence embeddings,
SBERT, dense vector search, cosine similarity, ANN (concept), retrieval-augmented intelligence.
(Mandatory Week 8 — Evaluation — remains ⏳; Day 6 is architecture, not a mandatory ML week.)

---

## 🔒 Changelog — corrections from the draft
1. **Scoped to real data.** Founders, valuation, exit, business model, team size, financial
   metrics, market/accelerator reports, funding rounds, descriptions/pitches — **removed or
   deferred** (they don't exist). (honesty)
2. **Embed a serialized venture record**, not an imaginary "startup description" — hybrid title +
   structured facts. (correctness/quality)
3. **KB = train split only + self-exclusion** on retrieval → leakage-free, non-circular evidence.
   (correctness)
4. **FAISS exact `IndexFlatIP`** (normalized = cosine); ANN taught as scale-up, not used. (executability)
5. **`embedding_model.pkl` → `embedding_model.json`** (name/version/dim); never pickle the torch
   model. (executability)
6. **Evidence from real retrieved aggregates**, not "weak retention"/"experienced team". (honesty)
7. **Deferred tiers** for founder/market/valuation retrieval — explicit placeholders, never faked.
   (honesty)
8. **Environment resolved** — faiss/sentence-transformers/torch installed + verified; ~2.3 min full
   encode measured. (blocker cleared)

## 💡 Improvements (beyond the draft)
- **Retrieval-based prior** (neighbor success rate) shown next to the Day-4 calibrated probability —
  a genuine second predictor, corroborating or flagging the model.
- **Validity check** proving the embedding space is meaningful (category concentration + retrieval-AUC
  vs base rate) before trusting the KB.
- **Cross-layer pipeline**: query → prediction (Day 4) + explanation-ready + retrieved evidence, one
  entry point, honesty-tiered.
- **Honesty caption** on every evidence pack (historical context, not proof).

✅ **Spec frozen.** Say "build it" or "build Phase A/B/C…" to implement + teach phase-by-phase via
Framework 1.
