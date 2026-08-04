"""Phase I — Recommendation Engine Serialization (Day-11, Layer 11).

Like every engine since Day 4, the recommendation engine is a **configuration over existing
models** — it orchestrates the Day-4 model and the Day-8/9/10 engines, and adds its own policy
(recommendation categories, prioritization weights, confidence weights, investor bands). So
serialization captures that policy and pins it to its dependencies, following the content-hash-as-
identity pattern:

- **engine config** — the Day-11 tunables (aggregator / generator / prioritization / founder /
  investor / confidence / pipeline).
- **decision policy** — the prioritization + confidence weights, the confidence range, and the
  investor-band definition (how the engine turns intelligence into a ranked, banded decision).
- **dependency hashes** — SHA-256 of the Day-4 model + the Day-8/9/10 engine manifests, so the
  recommendation engine is pinned to the exact machinery it orchestrates.
- **artifact hashes** — the produced intelligence / recommendation / strategy / confidence outputs.
- **content_hash** — SHA-256 of the canonicalised config + policy = the engine's identity.

``verify()`` re-hashes everything (dependencies, artifacts, live config) to detect and localize
drift. **JSON, not pickle** — a policy over models, not a binary. Deferred (fabricated) items are
recorded, so the engine documents its own honesty boundary.

Example
-------
>>> ser = RecommendationSerializer().build()
>>> ser.save(); ser.verify()["all_ok"]
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

# The honesty boundary — items the recommendation engine refuses to fabricate.
_DEFERRED_ITEMS = [
    "customer_retention", "monthly_revenue", "revenue_growth", "unit_economics",
    "competitive_moat", "named_competitors", "founder_experience", "team_size",
    "pricing_strategy", "market_size_tam", "product_traction", "funding_rounds",
    "growth_trajectory", "product_strategy", "team_strategy", "financial_strategy",
    "technology_strategy", "go_to_market_strategy",
]
# The investor-band rule (documents how readiness + headroom become a recommendation).
_INVESTOR_BANDS = {
    "Strong Investment": "stage == Strong",
    "Proceed with Caution": "stage == Ready, or Fixable with optimized_p >= threshold + strong_margin",
    "Monitor Progress": "Fixable with borderline optimized_p (or a positive band downgraded by low confidence)",
    "Do Not Invest Yet": "Not ready (optimized_p < threshold)",
}


class RecommendationSerializer:
    """Persist + verify the Day-11 recommendation-engine definition (config + policy + hashes).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reco.serialization")

        d11 = self.config.get("day11", {})
        c = d11.get("serialization", {})
        self.version = str(c.get("version", "1.0.0"))
        self.manifest_path = Path(c.get("manifest_path",
                                        "artifacts/recommendation/recommendation_engine.json"))
        self.dependencies = {k: Path(v) for k, v in c.get("dependencies", {}).items()}
        self.artifacts = {k: Path(v) for k, v in c.get("artifacts", {}).items()}
        self._d11 = d11
        self.manifest_: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ public
    def build(self) -> "RecommendationSerializer":
        """Assemble the recommendation-engine manifest."""
        engine_config = self._engine_config()
        policy = self._policy()
        self.manifest_ = {
            "version": self.version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "architecture_layer": 11,
            "note": ("Recommendation engine = decision policy over the Day-4 model + Day-8/9/10 "
                     "engines. Deterministic, no LLM; aggregates real signals into one "
                     "evidence-backed venture decision. Model-simulated, not causal."),
            "engine_config": engine_config,
            "decision_policy": policy,
            "content_hash": self._content_hash(engine_config, policy),
            "dependencies": {name: self._file_entry(p) for name, p in self.dependencies.items()},
            "artifacts": {name: self._file_entry(p) for name, p in self.artifacts.items()},
            "deferred_items": _DEFERRED_ITEMS,
        }
        self.logger.info("Built recommendation engine manifest v%s (content %s…).",
                         self.version, self.manifest_["content_hash"][:12])
        return self

    def save(self) -> Path:
        """Persist the manifest JSON."""
        if self.manifest_ is None:
            self.build()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest_, indent=2))
        self.logger.info("Saved recommendation engine -> %s", self.manifest_path)
        return self.manifest_path

    def verify(self) -> dict[str, Any]:
        """Re-hash dependencies, artifacts, and the live config; detect + localize drift."""
        m = self.manifest_ if self.manifest_ is not None else json.loads(self.manifest_path.read_text())
        checks: dict[str, bool] = {}
        for group in ("dependencies", "artifacts"):
            for name, entry in m.get(group, {}).items():
                p = Path(entry["path"])
                cur = self._sha256_file(p) if p.exists() else None
                checks[f"{group}.{name}"] = (cur == entry["sha256"])
        checks["content_hash"] = (
            self._content_hash(self._engine_config(), self._policy()) == m["content_hash"])
        all_ok = all(checks.values())
        self.logger.info("Recommendation verify: %s (%d/%d checks pass).",
                         "OK" if all_ok else "DRIFT", sum(checks.values()), len(checks))
        return {"all_ok": all_ok, "checks": checks, "version": m["version"]}

    def describe(self) -> str:
        """One-line summary of the engine."""
        m = self.manifest_ or json.loads(self.manifest_path.read_text())
        return (f"Recommendation engine v{m['version']} (content {m['content_hash'][:12]}…): "
                f"layer {m['architecture_layer']}; {len(m['dependencies'])} deps hashed, "
                f"{len(m['artifacts'])} artifacts, {len(m['deferred_items'])} items deferred.")

    # --------------------------------------------------------------- internals
    def _engine_config(self) -> dict[str, Any]:
        return {k: self._d11.get(k) for k in
                ("aggregator", "generator", "prioritization", "founder",
                 "investor", "confidence", "pipeline")}

    def _policy(self) -> dict[str, Any]:
        prio = self._d11.get("prioritization", {})
        conf = self._d11.get("confidence", {})
        inv = self._d11.get("investor", {})
        return {
            "prioritization_weights": prio.get("weights"),
            "confidence_weights": conf.get("weights"),
            "confidence_range": [conf.get("floor"), conf.get("ceil")],
            "use_model_agreement": conf.get("use_model_agreement"),
            "investor_bands": _INVESTOR_BANDS,
            "low_confidence_downgrade": inv.get("low_confidence_downgrade"),
            "readiness_index": "round(100 * mean(funding_probability, optimized_probability))",
        }

    def _content_hash(self, engine_config, policy) -> str:
        blob = json.dumps({"engine_config": engine_config, "policy": policy},
                          sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    def _file_entry(self, path: Path) -> dict[str, Any]:
        return {"path": str(path), "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": self._sha256_file(path) if path.exists() else None}

    @staticmethod
    def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
