# Day 15 · Phase A — Semantic Retrieval (conceptual notes)

*Layers 2 & 3 foundation. The core system was built on Day 6 — these notes recap the concepts and
ground them in the **real, already-running** retrieval engine, before Day 15 adds evaluation.*

---

## 0. Context — this already exists

The Venture Knowledge Base + semantic retrieval was implemented on **Day 6** and is served over the
API (`/retrieve`, Day 14). The real system, as it stands:

| Component | Real value |
|---|---|
| Embedding model | **`all-MiniLM-L6-v2`** (Sentence-Transformers) |
| Embedding dim | **384** |
| Corpus | **265,171** Kickstarter campaigns embedded |
| Normalization | **L2-normalized** (unit vectors) |
| Vector store | FAISS **`IndexFlatIP`** (exact inner-product = cosine) |
| Query latency | warm p50 **~14 ms** end-to-end (encode ~4.4 ms + single-query FAISS ~6.4 ms, single-threaded); batched FAISS amortizes to <1 ms/query. *Measured Day-15, not assumed.* |

So Phase A is *recap + grounding*, not new construction. Day 15's genuinely new work is **retrieval
evaluation** (§7) — Day 6 built the retriever but never *measured* its quality.

---

## 1. The problem: keyword search can't find "similar startups"

A prediction alone ("82% funded") is unpersuasive. Evidence makes it credible: *"campaigns like
yours funded at 44%; the ones that failed asked 2× as much."* To produce that, the platform must
**find historically similar campaigns**. Keyword search can't:

- A board game titled *"Artificer"* and one titled *"Dungeon Forge"* share **zero words** but are
  obviously similar. Keyword/TF-IDF match on surface tokens, so they'd score as unrelated.
- Conversely, *"Apple pie recipe book"* and *"Apple Watch accessory"* share the word "apple" but are
  unrelated. Keyword search over-matches on incidental token overlap.

Keyword search matches **strings**; startup similarity is about **meaning**. That gap is why we need
semantic retrieval.

---

## 2. Embeddings — meaning as geometry

An **embedding** maps text to a dense vector in ℝ³⁸⁴ such that *semantically similar texts land near
each other*. Our `all-MiniLM-L6-v2` model was trained (contrastively) so that paraphrases and related
sentences have nearby vectors and unrelated ones are far apart. The campaign title (cleaned) becomes
a 384-number point; the whole 265K corpus becomes a **point cloud** where proximity ≈ similarity.

This is the crucial shift: similarity stops being a string operation and becomes a **geometric**
one — "find similar campaigns" becomes "find nearby points."

---

## 3. Cosine similarity — the right metric for meaning

For two embeddings `u, v`, cosine similarity is `cos(u,v) = (u·v) / (‖u‖‖v‖) ∈ [−1, 1]`: 1 = same
direction (same meaning), 0 = orthogonal (unrelated). We use cosine (angle), **not** Euclidean
distance, because meaning lives in the *direction* of the vector, not its magnitude.

The implementation exploits a clean identity: since all embeddings are **L2-normalized** (‖u‖ = 1),
`cos(u,v) = u·v` — cosine *is* the inner product. So FAISS's inner-product index (`IndexFlatIP`)
computes cosine directly, no division needed. (Verified: our stored vectors have norm 1.0.)

---

## 4. Nearest-neighbor search — and the honest ANN caveat

"Retrieve the K most similar campaigns" = **k-nearest-neighbor search** in the 384-d space by cosine.
The draft (and most tutorials) reach for **Approximate Nearest Neighbor (ANN)** — indexes like IVF or
HNSW that trade a little recall for sub-linear search. **We deliberately use an *exact* flat index
(`IndexFlatIP`), not ANN**, and that's the honest choice at this scale:

- ANN exists to make search *sub-linear* when brute force is too slow. At **265K × 384-d**, exact
  brute-force inner product is **~6 ms** per single query single-threaded (and **<1 ms/query batched**)
  — already fast enough (measured Day-15; warm end-to-end retrieval is ~14 ms).
- So ANN would buy speed we don't need while **paying recall loss** (ANN can miss true neighbors).
  Exact search has **zero recall loss** — the top-K are provably the true top-K.
- ANN becomes worthwhile at *much* larger corpora (tens of millions+). It's a **scaling
  optimization**, noted for the future, not needed now.

Choosing exact-over-approximate here is a "don't optimize a bottleneck you don't have" call — and it
means the retrieval is *ground truth* for evaluation (no ANN noise to disentangle).

---

## 5. Retrieval-Augmented Intelligence — evidence for every claim

Retrieval turns predictions into *evidence-backed* claims. The `VentureRetriever` returns, for a query
campaign: its **similar** campaigns, split into **similar_successful** / **similar_failed** (by real
funded/failed outcome), plus **category & funding-goal benchmarks**. Downstream, this historical
evidence is what lets the platform say *why*, not just *what* — the difference between an ML model and
an intelligence system.

---

## 6. The honesty boundary (what retrieval can and can't surface)

Retrieval can only return what's **in the knowledge base** — Kickstarter campaigns (title text,
category, goal, funded/failed outcome). So the honest output is *"campaigns semantically similar to
yours, and how the funded vs failed ones differ."* The draft's example — *"SaaS healthcare startups
that raised Seed in 2023, with experienced founders and recurring revenue"* — is **fabrication**: the
KB has no founders, funding rounds, revenue, or moats. Those are **deferred**, never invented. Same
discipline as every prior layer, now at the retrieval boundary.

---

## 7. What Day 15 adds: honest retrieval evaluation

Day 6 built the retriever; it never *measured* it. But **Precision@K / Recall@K need ground-truth
"campaign X is truly similar to Y" relevance labels — which we don't have.** So Day 15 evaluates with
**proxy metrics against labels we *do* have**:

- **category-consistency@K** — do the top-K neighbors share the query's category? (a semantic-quality
  proxy: similar campaigns should cluster in similar categories).
- **kNN outcome-prediction** — do the K neighbors' *funded/failed* labels predict the query's **real**
  `target`? This is a *legitimate* metric: the query's outcome **is** ground truth, so "do retrieved
  neighbors predict it better than the base rate" is an honest measure of whether retrieval surfaces
  *outcome-relevant* evidence — precision/recall/F1/accuracy against a real label.
- **MRR-proxy**, **similarity-score distribution**, **self-retrieval sanity** (a campaign retrieves
  near-duplicates at rank 1).

All evaluated on the **held-out test split** — never the training corpus.

---

## 8. Semantic retrieval — the pattern

Layer 2/3 is **retrieval-augmented intelligence**: embed text into a geometric space, index it,
search by cosine to find meaning-neighbors, and feed the retrieved historical evidence to every
downstream layer. It's the platform's **memory**. The engineering discipline is honest defaults
(exact-not-approximate at this scale) and an honest boundary (surface only what's in the KB).

> **Interview soundbite:** *"Layers 2 and 3 are semantic retrieval — the platform's memory. I embed
> each campaign title with a Sentence-Transformer into a 384-dimensional space where proximity means
> semantic similarity, normalize to unit length so cosine similarity is just an inner product, and
> index all 265,000 campaigns in FAISS. The deliberate choice is an exact flat index rather than
> approximate nearest neighbor: ANN exists to make search sub-linear, but at 265K by 384 exact
> brute-force is already a few milliseconds per query — measured, not assumed — so ANN would only
> trade recall for speed I don't need —
> exact search is zero-recall-loss and gives ground-truth neighbors. Retrieval turns a bare
> probability into evidence: here are similar campaigns, split by whether they funded or failed. The
> honest part is that retrieval can only surface what's in the knowledge base — Kickstarter campaigns
> — so I reject the tempting 'similar SaaS startups with experienced founders and recurring revenue'
> framing as fabrication. And the new work this day is evaluation: since there are no ground-truth
> similarity labels, I measure retrieval quality with proxies I can actually ground — category
> consistency and whether the neighbors' funded-or-failed outcomes predict the query's real outcome."*
