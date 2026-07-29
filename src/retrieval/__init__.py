"""Layer 3 — Venture Intelligence Engine (Day 6).

SBERT embeddings + FAISS vector search + historical venture retrieval and
evidence generation. Retrieval complements the Day-4 prediction; it does not
replace it.

Environment fix (macOS): ``faiss-cpu`` and ``torch`` each bundle their own OpenMP
runtime. Running an SBERT encode (torch) and a FAISS search in the *same* process
segfaults on the OpenMP-region clash. The reliable fix is ``OMP_NUM_THREADS=1``, and it
must be set *before* either library initialises its OpenMP runtime — so we set it here,
at package import, ahead of the lazy ``import torch`` / ``import faiss`` inside the
modules. ``setdefault`` respects an explicit user override. (Cost: single-threaded batch
encoding; negligible for query-time encodes, and the corpus was already built in Phase C.)
"""

import os as _os

_os.environ.setdefault("OMP_NUM_THREADS", "1")   # faiss+torch OpenMP coexistence — see above
