"""Phase C — Embedding Generation (Day-6, Architecture Layer 2).

Turn every serialized venture record (Phase B) into a **dense semantic vector** with
SBERT. This is the representation shift that makes semantic retrieval possible:

    TF-IDF (Day 3)          SBERT (Day 6)
    ─────────────           ─────────────
    sparse, 800-dim         dense, 384-dim
    lexical similarity      semantic similarity
    "board game" ≠          "board game" ≈
      "tabletop game"          "tabletop game"

:class:`EmbeddingGenerator` wraps ``sentence-transformers`` `all-MiniLM-L6-v2` (primary;
BERT / larger SBERT are heavier discussion options). Two invariants:

- **L2-normalized output.** Every vector is unit length, so cosine similarity collapses to
  a dot product — exactly what FAISS ``IndexFlatIP`` computes in Phase D. Normalize once
  here, never again.
- **Row alignment with the KB.** ``embeddings[i]`` corresponds to ``kb.metadata()`` row *i*
  (same ``campaign_id``). So this class encodes exactly ``kb.records()`` and never
  subsamples on its own — a dev subsample belongs in ``day6.knowledge_base.subsample`` so
  metadata and embeddings shrink together.

We persist ``startup_embeddings.npy`` (float32) + ``embedding_model.json`` (name / library
version / dim / SHA-256) — **not** a pickle of the torch model (huge and fragile); the
model reloads from its name via the sentence-transformers cache.

Example
-------
>>> gen = EmbeddingGenerator()
>>> emb = gen.generate()              # builds KB, encodes 265k records (~2.3 min)
>>> emb.shape, np.linalg.norm(emb[0])
((265171, 384), 1.0)
>>> gen.save()                        # -> startup_embeddings.npy + embedding_model.json
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.knowledge_base.venture_database import VentureKnowledgeBase
from src.utils.config import load_config
from src.utils.logger import get_logger


class EmbeddingGenerator:
    """Encode venture-record strings into normalized SBERT embeddings.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.retrieval.embedding")

        c = self.config.get("day6", {}).get("embedding", {})
        self.model_name = c.get("model_name", "all-MiniLM-L6-v2")
        self.batch_size = int(c.get("batch_size", 256))
        self.normalize = bool(c.get("normalize", True))
        self.device = c.get("device", "cpu")
        self.out_dir = Path(c.get("out_dir", "artifacts/embeddings"))
        self.embeddings_path = Path(c.get("embeddings_path", self.out_dir / "startup_embeddings.npy"))
        self.model_info_path = Path(c.get("model_info_path", self.out_dir / "embedding_model.json"))

        self.model_ = None
        self.embeddings_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ public
    def load_model(self):
        """Lazily load the SBERT model (downloads once, then cached)."""
        if self.model_ is None:
            from sentence_transformers import SentenceTransformer  # heavy import
            self.model_ = SentenceTransformer(self.model_name, device=self.device)
            self.logger.info("Loaded SBERT '%s' (dim=%d) on %s.",
                             self.model_name, self.model_.get_sentence_embedding_dimension(),
                             self.device)
        return self.model_

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode a list of strings into normalized float32 embeddings (n, dim)."""
        model = self.load_model()
        t = time.perf_counter()
        emb = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,     # unit length -> cosine == dot product
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)
        dt = time.perf_counter() - t
        self.logger.info("Encoded %d records in %.1fs (%.0f rec/s), shape %s.",
                         len(texts), dt, len(texts) / max(dt, 1e-9), emb.shape)
        return emb

    def generate(self, kb: Optional[VentureKnowledgeBase] = None,
                 show_progress: bool = False) -> np.ndarray:
        """Build/accept the KB and encode all its records (row-aligned with metadata)."""
        kb = kb or VentureKnowledgeBase(self.config, self.logger).build()
        self.embeddings_ = self.encode(kb.records(), show_progress=show_progress)
        return self.embeddings_

    def save(self) -> dict[str, Path]:
        """Persist embeddings (.npy) + a model descriptor (.json, not a pickled model)."""
        if self.embeddings_ is None:
            raise ValueError("Nothing to save — call generate() first.")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings_)
        self.model_info_path.write_text(json.dumps(self._model_info(), indent=2))
        self.logger.info("Saved embeddings %s -> %s and descriptor -> %s",
                         self.embeddings_.shape, self.embeddings_path, self.model_info_path)
        return {"embeddings": self.embeddings_path, "model_info": self.model_info_path}

    # --------------------------------------------------------------- internals
    def _model_info(self) -> dict[str, Any]:
        import sentence_transformers as st
        emb = self.embeddings_
        return {
            "model_name": self.model_name,
            "sentence_transformers_version": st.__version__,
            "embedding_dim": int(emb.shape[1]),
            "n_embeddings": int(emb.shape[0]),
            "normalized": self.normalize,
            "dtype": str(emb.dtype),
            "sha256": hashlib.sha256(emb.tobytes()).hexdigest(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
