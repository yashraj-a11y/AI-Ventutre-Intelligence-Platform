"""Phase I — Knowledge Base Serialization (Day-6, Layers 2/3).

Ties the whole Venture Knowledge Base into one **versioned, tamper-evident bundle** — the
Day-4 serialization pattern (content-hash = identity) applied to retrieval. All the pieces
already exist (embeddings, FAISS index, metadata, model descriptor, mappings); this writes a
single ``knowledge_base_manifest.json`` that records, for each artifact, its path, size, and
**SHA-256**, plus the embedding-model / corpus / index descriptors and the retrieval config.

Why: a knowledge base is only reusable if you can prove the index, the embeddings, and the
metadata are the *same versions that were built together*. Hashing every artifact lets
``verify()`` detect any drift — a re-encoded embedding, a rebuilt index, an edited metadata
row — before it silently serves wrong neighbours. The manifest also carries which
capabilities are **deferred** (founder / market / valuation retrieval — no data), so the
bundle documents its own honesty boundary.

Example
-------
>>> ser = KnowledgeBaseSerializer().build()
>>> ser.save()
>>> ser.verify()["all_ok"]
True
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.utils.config import load_config
from src.utils.logger import get_logger

_DEFERRED_CAPABILITIES = [
    "similar founders / founder benchmarks — no founder or team data in this dataset",
    "market / competitor benchmarks — no market data",
    "valuation / exit history retrieval — not present",
]


class KnowledgeBaseSerializer:
    """Assemble, persist and verify the knowledge-base manifest.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.retrieval.serialization")

        d6 = self.config.get("day6", {})
        c = d6.get("serialization", {})
        self.version = str(c.get("version", "1.0.0"))
        self.manifest_path = Path(c.get("manifest_path",
                                        "artifacts/vector_db/knowledge_base_manifest.json"))
        self.artifacts = {k: Path(v) for k, v in c.get("artifacts", {}).items()}
        self._d6 = d6
        self.manifest_: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ public
    def build(self) -> "KnowledgeBaseSerializer":
        """Assemble the manifest: descriptors + retrieval config + artifact hashes."""
        self.manifest_ = {
            "version": self.version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "architecture_layers": [2, 3],
            "embedding_model": self._read_json(self.artifacts.get("embedding_model_json")),
            "corpus": self._corpus_descriptor(),
            "index": self._index_descriptor(),
            "retrieval_config": self._retrieval_config(),
            "artifacts": {name: self._artifact_entry(path)
                          for name, path in self.artifacts.items()},
            "deferred_capabilities": _DEFERRED_CAPABILITIES,
        }
        self.logger.info("Built KB manifest v%s over %d artifacts.",
                         self.version, len(self.artifacts))
        return self

    def save(self) -> Path:
        """Persist the manifest JSON."""
        if self.manifest_ is None:
            self.build()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest_, indent=2))
        self.logger.info("Saved KB manifest -> %s", self.manifest_path)
        return self.manifest_path

    def verify(self) -> dict[str, Any]:
        """Re-hash every artifact and compare to the saved manifest (drift detection)."""
        manifest = (self.manifest_ if self.manifest_ is not None
                    else json.loads(self.manifest_path.read_text()))
        results: dict[str, bool] = {}
        for name, entry in manifest["artifacts"].items():
            path = Path(entry["path"])
            current = self._sha256_file(path) if path.exists() else None
            results[name] = (current == entry["sha256"])
        all_ok = all(results.values())
        self.logger.info("KB verify: %s (%d/%d artifacts match).",
                         "OK" if all_ok else "MISMATCH",
                         sum(results.values()), len(results))
        return {"all_ok": all_ok, "artifacts": results, "version": manifest["version"]}

    def describe(self) -> str:
        """Human-readable one-paragraph summary of the bundle."""
        m = self.manifest_ or json.loads(self.manifest_path.read_text())
        em, cor, idx = m["embedding_model"], m["corpus"], m["index"]
        return (f"Venture KB v{m['version']}: {cor.get('n_records')} {cor.get('split')} campaigns, "
                f"{em.get('model_name')} ({idx.get('dim')}-dim), FAISS {idx.get('index_type')} "
                f"({idx.get('n_vectors')} vectors, {idx.get('metric')}). "
                f"{len(m['artifacts'])} artifacts hashed. "
                f"Deferred: {len(m['deferred_capabilities'])} capabilities (no data).")

    # --------------------------------------------------------------- internals
    def _artifact_entry(self, path: Path) -> dict[str, Any]:
        return {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": self._sha256_file(path) if path.exists() else None,
        }

    def _corpus_descriptor(self) -> dict[str, Any]:
        manifest = self._read_json(
            Path(self._d6.get("knowledge_base", {}).get("manifest_path",
                                                        "artifacts/embeddings/metadata.json")))
        return {k: manifest.get(k) for k in ("split", "n_records", "success_rate",
                                             "n_main_category", "n_category", "n_country")}

    def _index_descriptor(self) -> dict[str, Any]:
        mapping = self._read_json(self.artifacts.get("mapping"))
        return {k: mapping.get(k) for k in ("index_type", "metric", "n_vectors",
                                            "dim", "embeddings_sha256")}

    def _retrieval_config(self) -> dict[str, Any]:
        return {
            "embedding": {k: self._d6.get("embedding", {}).get(k)
                          for k in ("model_name", "batch_size", "normalize")},
            "semantic_search": {k: self._d6.get("semantic_search", {}).get(k)
                                for k in ("default_k", "pool_factor", "min_pool")},
            "venture_retriever": {k: self._d6.get("venture_retriever", {}).get(k)
                                  for k in ("k", "benchmark_pool")},
        }

    @staticmethod
    def _read_json(path: Optional[Path]) -> dict[str, Any]:
        return json.loads(path.read_text()) if path and path.exists() else {}

    @staticmethod
    def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
        """Stream a file through SHA-256 in 1 MB chunks (handles 400 MB artifacts)."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
