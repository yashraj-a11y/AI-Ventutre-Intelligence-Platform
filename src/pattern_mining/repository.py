"""Phase H — Pattern Knowledge Repository (Day-7, Layer 6).

Store every discovered pattern (success / failure / funding / temporal) + their readable
explanations in **one organized, provenance-tagged JSON repository** — an extension of the
Venture Knowledge Base that downstream layers (counterfactuals, recommendations, copilot)
query instead of re-mining.

**JSON, not pickle** (the Day-6 lesson): patterns are pure data — portable, diffable, and
honest. The repository records its **provenance** (source split, base rate, thresholds,
timestamp) and its **deferred capabilities**, so the whole pattern knowledge base carries its
own limits and reproducibility metadata.

:meth:`query` is the reuse surface: filter by pattern type, direction, and lift so a consumer
can ask "the strongest success patterns" or "failure patterns with lift below 0.3" without
touching the miners.

Example
-------
>>> repo = PatternRepository().build()
>>> repo.save()
>>> repo.query(direction="success", min_lift=1.7, top_k=3)          # strongest success rules
>>> PatternRepository().load().query(pattern_type="failure", max_lift=0.2)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.utils.config import load_config
from src.utils.logger import get_logger


class PatternRepository:
    """Aggregate, persist, and query the discovered pattern knowledge.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.patterns.repository")

        c = self.config.get("day7", {}).get("repository", {})
        self.version = str(c.get("version", "1.0.0"))
        self.sources = {k: Path(v) for k, v in c.get("sources", {}).items()}
        self.out_path = Path(c.get("out_path", "artifacts/patterns/pattern_repository.json"))

        self.repo_: dict[str, Any] = {}

    # ------------------------------------------------------------------ public
    def build(self, results: Optional[dict[str, Any]] = None) -> "PatternRepository":
        """Assemble the repository from miner result dicts (or the saved JSON sources)."""
        r = results or {k: self._read_json(p) for k, p in self.sources.items()}
        success = r.get("success", {}).get("patterns", [])
        failure = r.get("failure", {}).get("patterns", [])
        funding = r.get("funding", {})
        temporal = r.get("temporal", {})
        explanations = r.get("explanations", {}).get("explanations", [])

        base_rate = (r.get("success", {}).get("base_rate")
                     or funding.get("base_rate") or temporal.get("base_rate"))
        deferred = {**funding.get("deferred", {}), **temporal.get("deferred", {})}

        self.repo_ = {
            "version": self.version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "architecture_layer": 6,
            "note": ("Discovered venture patterns — historical associations from train "
                     "campaign metadata + titles, not causal drivers."),
            "provenance": {
                "source_split": "train",
                "base_rate": base_rate,
                "thresholds": r.get("success", {}).get("thresholds", {}),
            },
            "counts": {
                "success": len(success),
                "failure": len(failure),
                "funding_category_profiles": len(funding.get("category_profiles", [])),
                "funding_combos": len(funding.get("top_funding_combos", [])),
                "temporal_dimensions": sum(1 for k in ("year_trend", "month", "quarter",
                                                       "weekday", "weekend") if temporal.get(k)),
                "explanations": len(explanations),
            },
            "patterns": {
                "success": success,
                "failure": failure,
                "funding": {
                    "category_profiles": funding.get("category_profiles", []),
                    "top_funding_combos": funding.get("top_funding_combos", []),
                },
                "temporal": {k: temporal.get(k) for k in
                             ("year_trend", "month", "quarter", "weekday", "weekend", "highlights")},
                "explanations": explanations,
            },
            "deferred_capabilities": deferred,
        }
        self.logger.info("Built pattern repository v%s: %s.", self.version, self.repo_["counts"])
        return self

    def save(self) -> Path:
        """Persist the repository as JSON."""
        if not self.repo_:
            self.build()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(self.repo_, indent=2))
        self.logger.info("Saved pattern repository -> %s", self.out_path)
        return self.out_path

    def load(self) -> "PatternRepository":
        """Load a persisted repository."""
        self.repo_ = json.loads(self.out_path.read_text())
        self.logger.info("Loaded pattern repository v%s.", self.repo_.get("version"))
        return self

    def query(self, pattern_type: Optional[str] = None, direction: Optional[str] = None,
              min_lift: Optional[float] = None, max_lift: Optional[float] = None,
              top_k: Optional[int] = None) -> list[dict[str, Any]]:
        """Filter rule-patterns by type / direction / lift, ranked by |lift − 1|."""
        if not self.repo_:
            self.build()
        pats = self.repo_["patterns"]
        pool: list[dict[str, Any]] = []
        if pattern_type in (None, "success"):
            pool += pats["success"]
        if pattern_type in (None, "failure"):
            pool += pats["failure"]
        if pattern_type == "funding":
            pool += pats["funding"]["top_funding_combos"]

        out = []
        for p in pool:
            lift = p.get("lift")
            if lift is None:
                continue
            if direction and p.get("direction") != direction:
                continue
            if min_lift is not None and lift < min_lift:
                continue
            if max_lift is not None and lift > max_lift:
                continue
            out.append(p)
        out.sort(key=lambda p: abs(p["lift"] - 1.0), reverse=True)
        return out[:top_k] if top_k else out

    # --------------------------------------------------------------- internals
    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text()) if path.exists() else {}
