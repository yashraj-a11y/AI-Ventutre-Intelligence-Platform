"""Phase I — Pattern Mining Pipeline (Day-7, Layer 6 entry point).

One object that runs the whole engine from a single call:

    train KB → build item frame + mine rules (ONCE)
             → cluster similar ventures (MiniBatchKMeans on Day-6 embeddings)
             → success / failure / funding / temporal patterns
             → rank (Bonferroni) → explain → store in the pattern repository

The design payoff of the shared core: ``build_item_frame`` and ``mine_rules`` run **once**,
and every miner/ranker/explainer consumes that single output — no phase re-scans the 265k
campaigns. Clustering is the honest **cluster-based discovery**: unsupervised grouping of the
Day-6 embeddings, then per-cluster success rate + dominant real features (which embedding
neighborhoods fund well?).

No faiss/torch/lightgbm here — mining is pandas/scipy over the metadata, and clustering is
sklearn over ``np.load``-ed embeddings — so none of the Day-6 OpenMP conflicts apply.

Example
-------
>>> pipe = PatternMiningPipeline()
>>> report = pipe.run()                 # mines, clusters, ranks, explains, stores everything
>>> report["multiple_testing"]["n_rejected_by_correction"]
67
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.pattern_mining.explanation_generator import PatternExplanationGenerator
from src.pattern_mining.failure_patterns import FailurePatternMiner
from src.pattern_mining.funding_patterns import FundingPatternAnalyzer
from src.pattern_mining.growth_patterns import TemporalPatternEngine
from src.pattern_mining.mining import build_item_frame, mine_rules
from src.pattern_mining.pattern_ranker import PatternRanker
from src.pattern_mining.repository import PatternRepository
from src.pattern_mining.success_patterns import SuccessPatternMiner
from src.utils.config import load_config
from src.utils.logger import get_logger


class PatternMiningPipeline:
    """Run the full pattern-mining engine (mine-once → cluster → rank → explain → store).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.patterns.pipeline")

        m = self.config.get("day7", {}).get("mining", {})
        self.min_support_count = int(m.get("min_support_count", 500))
        self.single_features = list(m.get("single_features", []))
        self.pair_features = list(m.get("pair_features", []))

        c = self.config.get("day7", {}).get("pipeline", {})
        self.n_clusters = int(c.get("n_clusters", 25))
        self.embeddings_path = Path(c.get("embeddings_path",
                                          "artifacts/embeddings/startup_embeddings.npy"))
        self.cluster_random_state = int(c.get("cluster_random_state", 42))
        self.top_explanations = int(c.get("top_explanations", 15))
        self.report_path = Path(c.get("report_path",
                                      "artifacts/patterns/pattern_mining_report.json"))

    # ------------------------------------------------------------------ public
    def run(self) -> dict[str, Any]:
        """Mine once, cluster, rank, explain, and store — writing every artifact."""
        # 1) shared core: build the item frame + mine every rule ONCE.
        frame, base = build_item_frame(self.config, self.logger)
        rules = mine_rules(frame, base, self.single_features,
                           self.pair_features, self.min_support_count)
        self.logger.info("Shared mine: %d campaigns, %d rules.", len(frame), len(rules))

        # 2) pattern views (all reuse frame/base/rules — no re-scan).
        succ = SuccessPatternMiner(self.config, self.logger); succ.mine(frame, base, rules); succ.save()
        fail = FailurePatternMiner(self.config, self.logger); fail.mine(frame, base, rules); fail.save()
        fund = FundingPatternAnalyzer(self.config, self.logger); fund.analyze(frame, base, rules); fund.save()
        temp = TemporalPatternEngine(self.config, self.logger); temp.analyze(frame, base); temp.save()

        # 3) rank (Bonferroni) + explain (reuse the shared ranking).
        ranker = PatternRanker(self.config, self.logger)
        ranked = ranker.rank(frame, base, rules); ranker.save()
        gen = PatternExplanationGenerator(self.config, self.logger)
        gen.generate_examples(ranked=ranked, base=base)

        # 4) cluster-based discovery (unsupervised, over Day-6 embeddings).
        clusters = self._cluster(frame, base)

        # 5) aggregate into the repository (reads the just-written artifacts).
        repo = PatternRepository(self.config, self.logger).build(); repo.save()

        report = {
            "architecture_layer": 6,
            "base_rate": round(base, 4),
            "n_campaigns": len(frame),
            "n_rules_mined": len(rules),
            "counts": repo.repo_["counts"],
            "multiple_testing": ranker.summary_,
            "clusters": clusters,
            "artifacts": {
                "success": str(succ.out_path), "failure": str(fail.out_path),
                "funding": str(fund.out_path), "temporal": str(temp.out_path),
                "ranked": str(ranker.out_path), "explanations": str(gen.out_path),
                "repository": str(repo.out_path),
            },
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2))
        self.logger.info("Pattern mining complete. Report -> %s", self.report_path)
        return report

    # --------------------------------------------------------------- internals
    def _cluster(self, frame: pd.DataFrame, base: float) -> list[dict[str, Any]]:
        """Cluster the Day-6 embeddings; profile each cluster by success rate + dominant items."""
        if not self.embeddings_path.exists():
            self.logger.warning("Embeddings missing (%s) — skipping clustering.", self.embeddings_path)
            return []
        from sklearn.cluster import MiniBatchKMeans

        emb = np.load(self.embeddings_path)          # (N, 384), normalized (Day-6)
        if len(emb) != len(frame):
            self.logger.warning("Embeddings (%d) != frame (%d) — skipping clustering.",
                                len(emb), len(frame))
            return []
        km = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.cluster_random_state,
                             n_init=3, batch_size=4096).fit(emb)
        f = frame.copy()
        f["cluster"] = km.labels_

        profiles = []
        for cid, sub in f.groupby("cluster", observed=True):
            profiles.append({
                "cluster": int(cid),
                "n": int(len(sub)),
                "success_rate": round(float(sub["target"].mean()), 4),
                "lift": round(float(sub["target"].mean() / base), 3),
                "dominant_main_category": self._mode(sub["main_category"]),
                "dominant_goal_band": self._mode(sub["goal_band"]),
                "dominant_country": self._mode(sub["country"]),
            })
        profiles.sort(key=lambda p: p["lift"], reverse=True)
        self.logger.info("Clustered into %d groups; best cluster lift %.2f.",
                         len(profiles), profiles[0]["lift"] if profiles else 0.0)
        return profiles

    @staticmethod
    def _mode(s: pd.Series) -> Optional[str]:
        vc = s.value_counts()
        return str(vc.index[0]) if len(vc) else None
