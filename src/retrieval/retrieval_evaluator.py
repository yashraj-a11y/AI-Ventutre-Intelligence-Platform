"""Day 15 · Phase G — Retrieval Evaluation (honest proxy metrics).

Day 6 built the semantic retriever but never *measured* it. This does — honestly. There are **no
ground-truth "campaign X is truly similar to Y" relevance labels**, so textbook Precision@K against
such labels would be fabrication. Instead we evaluate against labels we *actually have*, on the
**held-out test split** (the KB is the train split):

- **category-consistency@K** — fraction of the top-K neighbors sharing the query's ``main_category``,
  reported as a **lift** over the base rate (random retrieval ≈ category prevalence). A semantic
  proxy: good retrieval clusters same-category campaigns.
- **kNN outcome-prediction** — do the K neighbors' funded/failed labels predict the query's **real**
  ``target``? The query's outcome IS ground truth, so accuracy / precision / recall / F1 / ROC-AUC of
  the neighbor-majority (or neighbor-funded-fraction) vs the true label is a *legitimate* measure of
  whether retrieval surfaces **outcome-relevant** evidence — vs the majority-class baseline.
- **MRR (category)** — mean reciprocal rank of the first same-category neighbor.
- **similarity distribution** — rank-1 / rank-K cosine stats (are neighbors meaningfully close?).

Batched (one encode + one FAISS search over all queries). Deterministic (fixed seed). No new deps.
Run with ``OMP_NUM_THREADS=1`` (faiss + SBERT).

Example
-------
>>> RetrievalEvaluator().evaluate()["knn_outcome"]["k5"]["roc_auc"]  # doctest: +SKIP
0.6...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.retrieval.venture_retriever import VentureRetriever
from src.utils.config import load_config
from src.utils.logger import get_logger

_CAPTION = ("Honest retrieval evaluation: no ground-truth similarity labels exist, so quality is "
            "measured by category-consistency (lift over base rate) and whether retrieved neighbors' "
            "outcomes predict the query's REAL held-out label. Test split queries vs train-split KB.")


class RetrievalEvaluator:
    """Evaluate the existing semantic retriever with honest proxy metrics on held-out queries.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    retriever:
        Optional pre-built Day-6 :class:`VentureRetriever` (shares the warm SBERT + FAISS stack).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 retriever: Optional[VentureRetriever] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.retrieval.evaluator")

        c = self.config.get("day15", {}).get("evaluation", {})
        self.k_values = list(c.get("k_values", [5, 10]))
        self.n_queries = int(c.get("n_queries", 1500))
        self.random_state = int(c.get("random_state", 15))
        self.out_path = Path(c.get("out_path", "artifacts/retrieval/retrieval_evaluation.json"))

        self.retriever = retriever or VentureRetriever(self.config, self.logger)
        self.engine = self.retriever.engine                 # SemanticSearchEngine (embedder + db + meta)

    # ------------------------------------------------------------------ public
    def evaluate(self, n: Optional[int] = None) -> dict[str, Any]:
        """Run the held-out proxy-metric evaluation and return the report dict."""
        n = n or self.n_queries
        max_k = max(self.k_values)
        engine = self.engine
        meta = engine.metadata_                             # aligned to index rows

        # --- held-out queries: sample from the TEST split (KB is the train split) ----------------
        kb = self.config.get("day6", {}).get("knowledge_base", {})
        df = pd.read_csv(kb.get("processed_path", "data/processed/kickstarter_processed.csv"))
        ti = np.load(Path(kb.get("split_dir", "artifacts/splits")) / "test_index.npy")
        queries = df.iloc[ti].sample(min(n, len(ti)), random_state=self.random_state).reset_index(drop=True)

        # --- batch: encode all queries once, one FAISS search ------------------------------------
        texts = [engine.serialize(r) for _, r in queries.iterrows()]
        qvecs = engine.embedder.encode(texts).astype("float32")     # (n, 384), normalized
        sims, rows = engine.vector_db.search(qvecs, k=max_k)         # (n, max_k)

        q_cat = queries["main_category"].to_numpy()
        q_target = queries["target"].to_numpy().astype(int)
        neigh_cat = meta["main_category"].to_numpy()[rows]          # (n, max_k)
        neigh_target = meta["target"].to_numpy().astype(int)[rows]  # (n, max_k)
        cat_match = (neigh_cat == q_cat[:, None])                   # (n, max_k) bool

        base_rate = float(q_target.mean())                         # test success rate
        base_cat_rate = float(np.mean([
            (queries["main_category"] == c).mean() for c in q_cat]))  # P(random neighbor same cat)
        majority_acc = max(base_rate, 1 - base_rate)               # trivial all-one-class baseline

        report: dict[str, Any] = {
            "n_queries": int(len(queries)), "k_values": self.k_values,
            "baselines": {"test_success_rate": round(base_rate, 4),
                          "base_category_rate": round(base_cat_rate, 4),
                          "majority_class_accuracy": round(majority_acc, 4)},
            "category_consistency": {}, "knn_outcome": {}, "mrr_category": {},
            "similarity": {}, "honesty_note": _CAPTION,
        }
        for k in self.k_values:
            key = f"k{k}"
            # category consistency + lift
            cc = float(cat_match[:, :k].mean())
            report["category_consistency"][key] = {
                "consistency": round(cc, 4), "base_rate": round(base_cat_rate, 4),
                "lift": round(cc / base_cat_rate, 2) if base_cat_rate else None}
            # kNN outcome prediction vs REAL target
            funded_frac = neigh_target[:, :k].mean(axis=1)          # neighbor funded fraction (score)
            pred = (funded_frac >= 0.5).astype(int)
            report["knn_outcome"][key] = {
                "accuracy": round(accuracy_score(q_target, pred), 4),
                "precision": round(precision_score(q_target, pred, zero_division=0), 4),
                "recall": round(recall_score(q_target, pred, zero_division=0), 4),
                "f1": round(f1_score(q_target, pred, zero_division=0), 4),
                "roc_auc": round(roc_auc_score(q_target, funded_frac), 4),
                "lift_over_majority": round(accuracy_score(q_target, pred) - majority_acc, 4)}
            # MRR (category) — first same-category neighbor
            has = cat_match[:, :k].any(axis=1)
            first = np.argmax(cat_match[:, :k], axis=1) + 1        # 1-indexed rank
            rr = np.where(has, 1.0 / first, 0.0)
            report["mrr_category"][key] = round(float(rr.mean()), 4)
            # similarity distribution
            report["similarity"][key] = {
                "rank1_mean": round(float(sims[:, 0].mean()), 4),
                "rankk_mean": round(float(sims[:, k - 1].mean()), 4),
                "p50_topk": round(float(np.percentile(sims[:, :k], 50)), 4)}

        self.logger.info("Retrieval eval on %d held-out queries: cat-consistency@5 %.2f (lift %.1fx), "
                         "kNN-AUC@5 %.3f.", len(queries),
                         report["category_consistency"]["k5"]["consistency"],
                         report["category_consistency"]["k5"]["lift"],
                         report["knn_outcome"]["k5"]["roc_auc"])
        return report

    def generate_report(self, n: Optional[int] = None) -> Path:
        """Run the evaluation and persist the JSON report."""
        report = self.evaluate(n)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(report, indent=2, default=str))
        self.logger.info("Saved retrieval evaluation -> %s", self.out_path)
        return self.out_path
