"""Phase D — Vector Database (Day-6, Architecture Layer 2).

Store the 265k normalized SBERT embeddings in a FAISS index so we can answer
"give me the K most similar campaigns to this vector" in milliseconds.

**Index choice — exact, not approximate.** Because every embedding is L2-normalized
(Phase C), cosine similarity == inner product, so a FAISS ``IndexFlatIP`` returns the
*exact* K nearest by cosine. At 265k × 384 that brute-force search is already
sub-millisecond, so we take **zero recall loss**. Approximate-nearest-neighbor indexes
(IVF, HNSW) and managed stores (Chroma / Pinecone / Milvus) are the scale-up path at
millions–billions of vectors; here they'd only add error and dependencies. We *teach*
ANN as the next step and *use* flat exact.

:class:`VectorDatabase` owns the index plus a **row→campaign mapping**: FAISS returns
integer row ids, and the mapping records that row *i* corresponds to
``campaign_id = mapping[i]`` (traceable to the processed dataset). Row *i* is also
metadata row *i* (Phase C alignment), so a retriever joins neighbors to their real
outcome/category by that shared index.

Example
-------
>>> vdb = VectorDatabase().build()
>>> sims, rows = vdb.search(query_vec, k=5)      # exact cosine, descending
>>> vdb.save(); vdb2 = VectorDatabase().load()   # round-trips
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger


class VectorDatabase:
    """FAISS exact-cosine index over normalized embeddings + a row→campaign mapping.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.retrieval.vector_db")

        c = self.config.get("day6", {}).get("vector_db", {})
        self.index_type = c.get("index_type", "flat_ip")
        self.embeddings_path = Path(c.get("embeddings_path",
                                          "artifacts/embeddings/startup_embeddings.npy"))
        self.model_info_path = Path(c.get("model_info_path",
                                          "artifacts/embeddings/embedding_model.json"))
        self.metadata_csv = Path(c.get("metadata_csv", "artifacts/embeddings/metadata.csv"))
        self.out_dir = Path(c.get("out_dir", "artifacts/vector_db"))
        self.index_path = Path(c.get("index_path", self.out_dir / "faiss.index"))
        self.mapping_path = Path(c.get("mapping_path", self.out_dir / "mapping.json"))

        self.index_ = None
        self.mapping_: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ public
    def build(self, embeddings: Optional[np.ndarray] = None,
              campaign_ids: Optional[list[int]] = None) -> "VectorDatabase":
        """Create the FAISS index and the row→campaign mapping."""
        import faiss

        if embeddings is None:
            embeddings = np.load(self.embeddings_path)
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)  # FAISS wants C-contig f32
        n, dim = emb.shape

        if campaign_ids is None:
            campaign_ids = pd.read_csv(self.metadata_csv, usecols=["campaign_id"])[
                "campaign_id"].tolist()
        if len(campaign_ids) != n:
            raise ValueError(f"Mapping mismatch: {len(campaign_ids)} ids vs {n} vectors.")

        index = faiss.IndexFlatIP(dim)     # inner product; normalized vectors => cosine
        index.add(emb)
        self.index_ = index
        self.mapping_ = {
            "index_type": self.index_type,
            "metric": "cosine (inner product on L2-normalized vectors)",
            "n_vectors": int(n),
            "dim": int(dim),
            "embeddings_sha256": self._embeddings_sha(),
            "campaign_ids": [int(x) for x in campaign_ids],
        }
        self.logger.info("Built FAISS %s: %d vectors, dim %d.", self.index_type, n, dim)
        return self

    def search(self, queries: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-k by cosine. Returns (similarities[n,k], row_indices[n,k])."""
        self._ensure()
        q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        sims, rows = self.index_.search(q, k)         # sims = inner products = cosine
        return sims, rows

    def campaign_id(self, row: int) -> int:
        """Map a FAISS row index back to its processed-dataset campaign id."""
        self._ensure()
        return int(self.mapping_["campaign_ids"][row])

    def save(self) -> dict[str, Path]:
        """Persist the FAISS index + the mapping JSON."""
        import faiss

        self._ensure()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index_, str(self.index_path))
        self.mapping_path.write_text(json.dumps(self.mapping_))
        self.logger.info("Saved index -> %s and mapping -> %s",
                         self.index_path, self.mapping_path)
        return {"index": self.index_path, "mapping": self.mapping_path}

    def load(self) -> "VectorDatabase":
        """Load a persisted index + mapping."""
        import faiss

        self.index_ = faiss.read_index(str(self.index_path))
        self.mapping_ = json.loads(self.mapping_path.read_text())
        self.logger.info("Loaded FAISS index (%d vectors) + mapping.", self.index_.ntotal)
        return self

    @property
    def n_vectors(self) -> int:
        self._ensure()
        return int(self.index_.ntotal)

    # --------------------------------------------------------------- internals
    def _ensure(self) -> None:
        if self.index_ is None:
            if self.index_path.exists():
                self.load()
            else:
                self.build()

    def _embeddings_sha(self) -> Optional[str]:
        """Carry the Phase-C embedding hash so index + embeddings are tied together."""
        if self.model_info_path.exists():
            return json.loads(self.model_info_path.read_text()).get("sha256")
        return None
