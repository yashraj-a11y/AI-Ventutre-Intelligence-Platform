"""Phase G — Retrieval Evidence Generator (Day-6, Architecture Layer 3).

Distill the Phase-F retrieval bundle into a compact **evidence pack**: the headline
numbers plus a founder-readable paragraph, all grounded in real retrieved outcomes.

    Historical evidence
    ───────────────────
    Retrieved 50 similar campaigns · 17 funded (34%)
    winners sought a small goal (median $4,000) · losers a modest goal (median $9,000)
    this campaign's $8,000 goal is above the winners' median
    Games category overall: 44% fund

Two things distinguish this from the draft:

- **Real patterns, not fabricated reasons.** The draft's "most common failure reason:
  weak retention" is invented. We report what the data actually shows: the *goal pattern*
  of similar winners vs losers (from real medians), and the dominant category/country of
  the neighbourhood — never founder / retention / market claims.
- **The retrieval-based prior sits next to the model.** The neighbours' funding rate is a
  k-NN estimate of P(fund); given the Day-4 calibrated probability it is compared —
  *corroborates* / *more optimistic* / *more cautious* — so retrieval becomes evidence for
  or against the prediction, not a competing black box.

Every pack carries an honesty caption: historical context, not proof.

Example
-------
>>> gen = EvidenceGenerator()
>>> pack = gen.generate(query, model_probability=0.28)
>>> print(pack["evidence_text"])
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from src.knowledge_base.venture_database import goal_band
from src.retrieval.venture_retriever import VentureRetriever, _json_safe
from src.utils.config import load_config
from src.utils.logger import get_logger

_HONESTY = ("Historical outcomes of similar past campaigns — context for, not proof of, "
            "this campaign's result. Based on campaign metadata and titles only (no "
            "founder / market / financial data).")


class EvidenceGenerator:
    """Turn a retrieval bundle into a structured + readable evidence pack.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    retriever:
        Optional pre-built :class:`VentureRetriever` (share a warm index/model).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 retriever: Optional[VentureRetriever] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.retrieval.evidence")

        c = self.config.get("day6", {}).get("evidence", {})
        self.n_examples = int(c.get("n_examples", 4))
        self.agree_tol = float(c.get("prior_model_agree_tol", 0.1))
        self.examples_path = Path(c.get("examples_path",
                                        "artifacts/retrieval/evidence_examples.json"))

        self.retriever = retriever or VentureRetriever(self.config, self.logger)

    # ------------------------------------------------------------------ public
    def generate(self, query: Union[dict, pd.Series],
                 model_probability: Optional[float] = None,
                 exclude_campaign_ids: Optional[list[int]] = None) -> dict[str, Any]:
        """Retrieve, then distill into an evidence pack (structured + readable)."""
        bundle = self.retriever.retrieve(query, exclude_campaign_ids=exclude_campaign_ids)
        return self._distill(bundle, model_probability)

    def generate_examples(self, n: Optional[int] = None, random_state: int = 5) -> Path:
        """Evidence packs for a spread of real campaigns -> deliverable JSON."""
        n = n or self.n_examples
        feat = ["name", "category", "main_category", "currency", "deadline",
                "goal", "launched", "country", "usd_goal_real", "name_clean"]
        processed = self.config.get("day6", {}).get("knowledge_base", {}).get(
            "processed_path", "data/processed/kickstarter_processed.csv")
        sample = pd.read_csv(processed).sample(n, random_state=random_state).reset_index(drop=True)
        packs = [self.generate(row.to_dict()) for _, row in sample[feat].iterrows()]
        self.examples_path.parent.mkdir(parents=True, exist_ok=True)
        self.examples_path.write_text(json.dumps(packs, indent=2, default=_json_safe))
        self.logger.info("Saved %d evidence packs -> %s", len(packs), self.examples_path)
        return self.examples_path

    # --------------------------------------------------------------- internals
    def _distill(self, bundle: dict[str, Any],
                 model_probability: Optional[float]) -> dict[str, Any]:
        fb = bundle["funding_benchmark"]
        cb = bundle["category_benchmark"]
        qa = bundle["query"]
        prior = fb.get("success_rate")

        succ_med = fb.get("successful", {}).get("median_goal_usd")
        fail_med = fb.get("failed", {}).get("median_goal_usd")
        funding_pattern = {
            "successful_median_goal_usd": succ_med,
            "failed_median_goal_usd": fail_med,
            "success_goal_band": goal_band(succ_med) if succ_med is not None else None,
            "failure_goal_band": goal_band(fail_med) if fail_med is not None else None,
            "query_goal_usd": fb.get("query_goal_usd"),
            "query_vs_successful_median": fb.get("query_goal_vs_successful_median"),
        }

        pack = {
            "n_retrieved": fb.get("n_neighbors", 0),
            "n_funded": fb.get("successful", {}).get("n", 0),
            "n_failed": fb.get("failed", {}).get("n", 0),
            "retrieval_prior": prior,                       # k-NN P(fund)
            "model_probability": (round(float(model_probability), 4)
                                  if model_probability is not None else None),
            "prior_vs_model": self._compare(prior, model_probability),
            "dominant_category": self._mode([n["category"] for n in bundle["similar"]]),
            "dominant_country": self._mode([n["country"] for n in bundle["similar"]]),
            "funding_pattern": funding_pattern,
            "category_benchmark": cb,
            "examples": {
                "successful": [self._example(n) for n in bundle["similar_successful"][:2]],
                "failed": [self._example(n) for n in bundle["similar_failed"][:2]],
            },
            "deferred": bundle["deferred_benchmarks"],
            "honesty_caption": _HONESTY,
        }
        pack["evidence_text"] = self._render(pack, qa, cb)
        return pack

    def _render(self, pack: dict[str, Any], qa: dict[str, Any],
                cb: dict[str, Any]) -> str:
        fp = pack["funding_pattern"]
        lines = [
            f"Retrieved {pack['n_retrieved']} similar past campaigns: "
            f"{pack['n_funded']} funded ({_pct(pack['retrieval_prior'])}) — a retrieval-based "
            f"estimate of this campaign's funding chance."
        ]
        if fp["successful_median_goal_usd"] is not None and fp["failed_median_goal_usd"] is not None:
            lines.append(
                f"Similar campaigns that funded typically sought {fp['success_goal_band']} "
                f"(median ${fp['successful_median_goal_usd']:,.0f}); those that failed sought "
                f"{fp['failure_goal_band']} (median ${fp['failed_median_goal_usd']:,.0f}). "
                f"This campaign's ${fp['query_goal_usd']:,.0f} goal is {fp['query_vs_successful_median']} "
                f"the funded median."
            )
        if cb.get("success_rate") is not None:
            lines.append(
                f"Across the {cb['main_category']} category overall, "
                f"{_pct(cb['success_rate'])} of campaigns fund.")
        if pack["model_probability"] is not None:
            lines.append(
                f"The prediction model estimates {_pct(pack['model_probability'])}; "
                f"retrieved evidence {pack['prior_vs_model']}.")
        lines.append(f"Note: {_HONESTY}")
        return " ".join(lines)

    def _compare(self, prior: Optional[float], model: Optional[float]) -> Optional[str]:
        if prior is None or model is None:
            return None
        diff = prior - model
        if abs(diff) <= self.agree_tol:
            return "corroborates the model"
        return ("is more optimistic than the model" if diff > 0
                else "is more cautious than the model")

    @staticmethod
    def _mode(values: list[Any]) -> Optional[Any]:
        vals = [v for v in values if v is not None]
        return Counter(vals).most_common(1)[0][0] if vals else None

    @staticmethod
    def _example(n: dict[str, Any]) -> dict[str, Any]:
        return {"title": n.get("title"), "similarity": n.get("similarity"),
                "outcome": n.get("outcome"), "reason": n.get("reason")}


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{round(100 * x)}%"
