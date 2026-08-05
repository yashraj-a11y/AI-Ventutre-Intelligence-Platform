"""Day 14 · Phase E — Retrieval service (Day-6 VentureRetriever, faiss)."""
from __future__ import annotations
from typing import Any
from src.core.exceptions import ArtifactError
from src.services.base import campaign_of, summary


def retrieve(registry: Any, req: Any, k: int = 5) -> dict[str, Any]:
    """Semantically similar campaigns from the knowledge base."""
    if registry.retriever is None:                       # -> 503 artifact_unavailable
        raise ArtifactError("Semantic retrieval is unavailable (day14.api.load_retriever is false).")
    camp = campaign_of(req)
    res = registry.retriever.retrieve(camp, k=k)
    return {"query": summary(camp),
            "similar": res.get("similar", []),
            "similar_successful": res.get("similar_successful", []),
            "similar_failed": res.get("similar_failed", []),
            "honesty_note": res.get("honesty_note")}
