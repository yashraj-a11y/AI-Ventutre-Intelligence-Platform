# Day 6 · Phase A — Understanding Semantic Retrieval

> Conceptual foundation for Layers 2 & 3. Grounded in this project's real numbers:
> a **265k-campaign** train knowledge base, embedded with **SBERT all-MiniLM-L6-v2**
> (384-dim, ~1,900 records/s → ~2.3 min full encode), searched with **FAISS exact
> cosine**. Complements — does not replace — the Day-4 model (calibrated LightGBM,
> test PR-AUC 0.688).

---

## 1. The problem Phase A solves

The Day-4 model answers *"what is the probability this campaign funds?"* — a number from
learned weights. But a venture-intelligence platform must also answer questions a weight
vector can't:

> *Which past campaigns are most like mine? Which similar ones **failed**? Which
> **succeeded**? How did comparable campaigns set their goals?*

Those are **retrieval** questions — you answer them by finding evidence in history, not by
running a classifier. Phase A defines *how* we find "similar," and why the obvious way
(keyword matching) fails.

---

## 2. Why keyword search fails (measured, not asserted)

Keyword search scores similarity by **shared words**. Two campaigns are "similar" if their
titles overlap. This breaks in both directions — here are three real title records run
through SBERT:

| Pair | Shared words? | Keyword (Jaccard) | Semantic (cosine) |
|---|---|---:|---:|
| "Handcrafted wooden **board game** for families" ↔ "Artisan **tabletop** dice adventure" | **none** | **0.00** | **0.337** |
| "Handcrafted wooden **board** game" ↔ "Mobile app for tracking **board** meetings" | "board" | **0.20** | **0.222** |

**Keyword search ranks these backwards.** It calls the same-meaning pair *unrelated* (0.00,
no shared words) and the unrelated pair *more related* (0.20, both say "board"). Semantic
search gets it right: the two physical games are closest (0.337), the board-**game** vs
board-**meeting** pair is farther (0.222). Meaning lives in **word co-occurrence patterns**,
not exact strings — and a 5.6-word Kickstarter title has almost no room for lexical overlap,
so keyword matching is especially hopeless here.

---

## 3. Embeddings — from words to meaning-vectors

An **embedding** maps text to a point in a continuous vector space where *distance encodes
meaning*: similar meanings → nearby points. SBERT (Sentence-BERT) reads a whole record and
outputs one 384-dimensional vector; a transformer trained so that paraphrases land close and
unrelated sentences land far apart.

Contrast with what we already built in Day 3:

| | **TF-IDF** (Day 3) | **SBERT** (Day 6) |
|---|---|---|
| Representation | **sparse** (800 dims, mostly 0) | **dense** (384 dims, all nonzero) |
| Dimension = | one per vocabulary term | latent semantic axes (no per-word meaning) |
| Similarity captured | **lexical** (shared terms) | **semantic** (shared meaning) |
| "board game" ~ "tabletop game"? | **No** (no shared token) | **Yes** (near in space) |
| Built from | corpus term statistics | a pretrained neural model |

TF-IDF and SBERT are complementary lenses — Day 3 fused TF-IDF into the *predictor*; Day 6
uses SBERT for *retrieval*.

---

## 4. Dense vs sparse representations

- **Sparse** (TF-IDF): dimension = vocabulary size (~800 here), a vector is mostly zeros, each
  axis is a human-readable word. Great for exact-term signals, blind to synonyms.
- **Dense** (SBERT): a short fixed-length vector (384), every entry active, axes are learned
  and *not* individually interpretable — but the *geometry* is. "Similar" = close in this space.

Dense wins for *meaning*; sparse wins for *interpretability and exact terms*. We embed a
**serialized venture record** ("A Games project based in the US, seeking about 5,000 USD"),
so the dense vector captures title semantics **and** structured facts together.

---

## 5. Cosine similarity — the distance that matters

We compare embeddings by the **angle** between them, not their length:

$$\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert} \in [-1, 1]$$

Cosine ignores magnitude and measures *orientation* — the right notion for "same meaning,
different length of text." Trick we use everywhere in Day 6: **L2-normalize** every embedding
once (‖a‖ = 1); then cosine reduces to a plain **dot product**, which is exactly what FAISS's
`IndexFlatIP` computes — fast and exact.

---

## 6. Vector databases & Approximate Nearest Neighbor (ANN)

A **vector database** stores millions of embeddings and answers "give me the K nearest to this
query." Done naively that's a full scan (compare the query to every stored vector). **ANN**
indexes (IVF, HNSW) trade a little recall for large speed-ups at web scale (billions of
vectors).

**Our honest scale call:** 265k × 384 is *tiny* for FAISS. A brute-force **exact** flat index
searches it in milliseconds with **zero recall loss**, so we use exact `IndexFlatIP` and *teach*
ANN as the scale-up path we'd reach for at millions of vectors — not take approximate error we
don't need. (Chroma / Pinecone / Milvus are the managed alternatives; overkill for a
self-contained local project.)

---

## 7. Retrieval-augmented intelligence — how retrieval *complements* the model

The Day-4 model gives a **calibrated probability** from learned weights. Retrieval gives
**evidence from history**: of the K most similar past campaigns, how many funded, at what
goals, in which categories. These are *two independent estimators*:

- The model is a smooth global function of 865 features.
- The **k-NN success rate of the neighbors** is a local, memory-based estimate of P(fund).

When they **agree**, the prediction is corroborated by precedent; when they **diverge**, that's
a flag worth surfacing. Retrieval doesn't overrule the model — it grounds every downstream
explanation, recommendation, and report in *"here are the actual similar campaigns and what
happened to them."* That is the shift from a prediction system to an **evidence-driven** one.

---

## 8. The honesty boundary (locked for all of Day 6)

Retrieval can only surface what the data contains: title, category, country, goal, and the
**real outcome**. There are **no** founder, team, valuation, exit, or market signals — so there
is no "similar founders" or "founder benchmark" retrieval; those are **deferred**, never faked.
Every evidence pack is framed as *historical context, not proof*: "similar past campaigns tended
to fund at rate X" is an association from precedent, not a guarantee about this campaign.

---

## 9. Interview soundbite

> *"Day 6 adds retrieval on top of the classifier. Keyword search fails on 6-word campaign
> titles — I measured a same-meaning pair with zero shared words scoring 0.0 on keyword overlap
> but 0.34 on SBERT cosine, while a pair sharing the word 'board' but meaning different things
> scored higher on keywords than the true match — so keyword ranks them backwards. SBERT maps
> each campaign to a 384-dim dense vector where cosine distance is semantic distance; I
> L2-normalize so cosine becomes a dot product and search a FAISS flat index — exact, because at
> 265k vectors I don't need ANN's recall trade-off. The payoff is retrieval-augmented
> intelligence: the neighbors' historical success rate is a second, memory-based estimate of
> funding probability that corroborates or flags the model's calibrated one, so every downstream
> insight is grounded in real precedent, not just weights."*
