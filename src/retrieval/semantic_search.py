"""Phase E — Semantic Search Engine (Day-6, Architecture Layer 3).

The query side. Given a campaign (or free text), return the most similar historical
campaigns joined to their **real** metadata (title, category, country, goal, outcome).

    query campaign ──serialize──▶ SBERT ──▶ FAISS top-K ──▶ join metadata ──▶ neighbors
                     (same as KB)   (same model)   (exact cosine)    (row i)

Composes the Day-6 stack: :class:`EmbeddingGenerator` (embed the query with the **same
model + normalization** used for the corpus — train/serve consistency), the Phase-B
serialization (:meth:`VentureKnowledgeBase.to_text`, so a query is described exactly like a
corpus record), and :class:`VectorDatabase` (exact-cosine FAISS + row→campaign mapping).

Two behaviours from the frozen spec:

- **Self-exclusion.** A campaign is its own nearest neighbor at cosine 1.0. When the query
  is a KB member, pass its ``campaign_id`` and it is dropped from results (never "retrieve
  yourself"). New campaigns have no self to exclude.
- **Metadata filters, honest facets only.** Filter neighbors by real columns —
  ``main_category``, ``country``, ``outcome`` — not by founder/market fields (which don't
  exist). FAISS flat has no native filter, so we **over-fetch** a pool then post-filter.

Example
-------
>>> eng = SemanticSearchEngine()
>>> nbrs = eng.search({"name": "Handcrafted wooden board game", "category": "Tabletop Games",
...                    "main_category": "Games", "country": "US", "usd_goal_real": 8000,
...                    "launched": "2016-05-01"}, k=5, filters={"outcome": "successful"})
>>> nbrs[0]["title"], nbrs[0]["similarity"]
('...', 0.72)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd

from src.knowledge_base.venture_database import VentureKnowledgeBase, goal_band
from src.retrieval.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_database import VectorDatabase
from src.utils.config import load_config
from src.utils.logger import get_logger

# Columns surfaced on each neighbour (all real).
_NEIGHBOR_COLS = ["campaign_id", "title", "category", "main_category", "country",
                  "currency", "goal", "usd_goal_real", "launch_year", "outcome", "target", "text"]


class SemanticSearchEngine:
    """Embed a query and retrieve similar historical campaigns (exact cosine).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    vector_db, embedder:
        Optional pre-built collaborators (share a warm index / model across calls).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 vector_db: Optional[VectorDatabase] = None,
                 embedder: Optional[EmbeddingGenerator] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.retrieval.semantic_search")

        c = self.config.get("day6", {}).get("semantic_search", {})
        self.metadata_csv = Path(c.get("metadata_csv", "artifacts/embeddings/metadata.csv"))
        self.default_k = int(c.get("default_k", 5))
        self.pool_factor = int(c.get("pool_factor", 20))
        self.min_pool = int(c.get("min_pool", 300))

        self.vector_db = vector_db or VectorDatabase(self.config, self.logger).load()
        self.embedder = embedder or EmbeddingGenerator(self.config, self.logger)
        self._kb = VentureKnowledgeBase(self.config, self.logger)   # for to_text only
        self.metadata_ = pd.read_csv(self.metadata_csv)
        if len(self.metadata_) != self.vector_db.n_vectors:
            raise ValueError("Metadata rows != index vectors — rebuild Day-6 artifacts.")

    # ------------------------------------------------------------------ public
    def serialize(self, campaign: Union[dict, pd.Series]) -> str:
        """Describe a raw query campaign exactly like a KB record (same serialization)."""
        c = dict(campaign)
        s = pd.Series(c)
        s["usd_goal_real"] = float(c.get("usd_goal_real", c.get("goal", 0.0)))
        s["goal_band"] = goal_band(s["usd_goal_real"])
        if "launch_year" not in c:
            s["launch_year"] = pd.to_datetime(c.get("launched"), errors="coerce").year \
                if c.get("launched") is not None else np.nan
        s["title"] = str(c.get("name", c.get("title", ""))).strip()
        return self._kb.to_text(s)

    def embed_query(self, query: Union[str, dict, pd.Series]) -> np.ndarray:
        """Turn a text string or a campaign into a normalized query vector."""
        text = query if isinstance(query, str) else self.serialize(query)
        return self.embedder.encode([text])[0]

    def search(self, query: Union[str, dict, pd.Series], k: Optional[int] = None,
               filters: Optional[dict[str, Any]] = None,
               exclude_campaign_ids: Optional[Iterable[int]] = None) -> list[dict[str, Any]]:
        """Top-k similar campaigns, ranked by cosine, with self-exclusion + filters."""
        k = k or self.default_k
        qvec = self.embed_query(query)
        return self._search_vector(qvec, k, filters, exclude_campaign_ids)

    def similar_to_id(self, campaign_id: int, k: Optional[int] = None,
                      filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Neighbours of an existing KB campaign, excluding itself."""
        row = self.metadata_.index[self.metadata_["campaign_id"] == campaign_id]
        if len(row) == 0:
            raise KeyError(f"campaign_id {campaign_id} not in KB.")
        text = self.metadata_.loc[row[0], "text"]
        return self.search(text, k=k, filters=filters, exclude_campaign_ids=[campaign_id])

    # --------------------------------------------------------------- internals
    def _search_vector(self, qvec, k, filters, exclude_campaign_ids) -> list[dict[str, Any]]:
        exclude = {int(x) for x in (exclude_campaign_ids or [])}
        # Over-fetch a pool when we must filter/exclude (FAISS flat has no native filter).
        pool = k if not (filters or exclude) else min(
            max(k * self.pool_factor, self.min_pool), self.vector_db.n_vectors)
        sims, rows = self.vector_db.search(qvec, pool)
        sims, rows = sims[0], rows[0]

        out: list[dict[str, Any]] = []
        for sim, r in zip(sims, rows):
            if r < 0:
                continue
            rec = self.metadata_.iloc[int(r)]
            if int(rec["campaign_id"]) in exclude:      # self-exclusion
                continue
            if filters and not self._passes(rec, filters):
                continue
            out.append(self._neighbor(rec, float(sim), len(out) + 1))
            if len(out) >= k:
                break
        return out

    @staticmethod
    def _passes(rec: pd.Series, filters: dict[str, Any]) -> bool:
        for col, want in filters.items():
            val = rec[col]
            ok = (val in want) if isinstance(want, (list, tuple, set)) else (val == want)
            if not ok:
                return False
        return True

    @staticmethod
    def _neighbor(rec: pd.Series, similarity: float, rank: int) -> dict[str, Any]:
        d = {"rank": rank, "similarity": round(similarity, 4)}
        for col in _NEIGHBOR_COLS:
            v = rec[col]
            d[col] = (int(v) if col in ("campaign_id", "target") else
                      (float(v) if col in ("goal", "usd_goal_real") else
                       (int(v) if col == "launch_year" and pd.notna(v) else
                        (None if pd.isna(v) else v))))
        return d
