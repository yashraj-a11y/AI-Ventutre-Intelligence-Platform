"""Day 17 · Phase E — retrieval validation (honest proxy metrics, no fabricated labels).

All `heavy` (loads the SBERT model + 407 MB FAISS index). There are **no ground-truth "campaign X ~ Y"
similarity labels**, so we do NOT assert Precision@K against them. Instead we reuse the Day-15
`RetrievalEvaluator` and assert thresholds on the honest proxies — category-consistency lift over the
base rate, kNN-outcome ROC-AUC vs the real `target`, MRR — plus a self-retrieval sanity check (a stored
campaign retrieves *itself* at rank 0). Report → `artifacts/testing/retrieval_validation.json`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.retrieval.retrieval_evaluator import RetrievalEvaluator

_REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.heavy


@pytest.fixture(scope="module")
def evaluator() -> RetrievalEvaluator:
    """Build the retriever once (loads the 407 MB index) and share across this module's tests."""
    return RetrievalEvaluator()


def test_retrieval_proxy_metrics_meet_thresholds(evaluator):
    rep = evaluator.evaluate(n=300)                       # 300 held-out TEST queries, one FAISS batch
    cc5 = rep["category_consistency"]["k5"]
    kn5, kn10 = rep["knn_outcome"]["k5"], rep["knn_outcome"]["k10"]
    sim5 = rep["similarity"]["k5"]

    checks = {
        "category_lift@5>=3x": cc5["lift"] >= 3.0,
        "knn_auc@5>0.5": kn5["roc_auc"] > 0.5,
        "knn_auc@10>0.5": kn10["roc_auc"] > 0.5,
        "mrr@5>0.5": rep["mrr_category"]["k5"] > 0.5,
        "similarity_descending": sim5["rank1_mean"] > sim5["rankk_mean"],
        "knn_beats_majority@5": kn5["lift_over_majority"] > 0.0,
    }
    report = {"n_queries": rep["n_queries"], "checks": checks, "passed": all(checks.values()),
              "category_consistency": rep["category_consistency"], "knn_outcome": rep["knn_outcome"],
              "mrr_category": rep["mrr_category"], "similarity": rep["similarity"],
              "honesty_note": "proxy metrics vs real labels — no Precision@K (no similarity ground truth)",
              "created_utc": datetime.now(timezone.utc).isoformat()}
    out = _REPO / "artifacts/testing/retrieval_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))

    assert report["passed"], {k: v for k, v in checks.items() if not v}


def test_self_retrieval_ranks_itself_first(evaluator):
    """A stored campaign, searched by its own text, comes back at rank 0 with similarity ≈ 1.0."""
    engine = evaluator.engine
    rec = engine.metadata_.iloc[100]
    hits = engine.search(rec["text"], k=3)               # NOT similar_to_id (that excludes self)
    assert hits[0]["campaign_id"] == int(rec["campaign_id"])
    assert hits[0]["similarity"] >= 0.99                 # a vector is maximally similar to itself


def test_retrieval_is_deterministic(evaluator):
    engine = evaluator.engine
    text = engine.metadata_.iloc[100]["text"]
    a = [h["campaign_id"] for h in engine.search(text, k=5)]
    b = [h["campaign_id"] for h in engine.search(text, k=5)]
    assert a == b                                        # exact index -> stable neighbours
