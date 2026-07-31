"""Phase I — Counterfactual Engine Serialization (Day-8, Layer 7).

The counterfactual engine is a **configuration over existing models** (the Day-4 calibrated
model + Day-7 patterns), not a new trained artifact. So serialization captures what defines it
and ties it to its dependencies — the Day-4/6 content-hash-as-identity pattern:

- **engine config** — the tunables (candidates / simulator / evidence / ranking / report).
- **recommendation templates** — mutable/immutable feature sets, tier weights, feasibility,
  and the per-feature action phrasing (how the engine turns a change into advice).
- **dependency hashes** — SHA-256 of the Day-4 calibrated model + Day-7 patterns the engine
  builds on, so the engine is pinned to the exact models it was defined against.
- **artifact hashes** — the produced recommendations / scenarios / report.
- **content_hash** — SHA-256 of the canonicalised config+templates = the engine's identity.

``verify()`` re-hashes everything (dependencies, artifacts, and the live config) to detect
drift. **JSON, not pickle** — a config, not a binary. Deferred (fabricated) features are
recorded, so the engine documents its own honesty boundary.

Example
-------
>>> ser = CounterfactualSerializer().build()
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

_DEFERRED_FEATURES = [
    "monthly_revenue", "revenue_growth", "customer_retention", "burn_rate", "team_size",
    "founder_experience", "market_size", "product_traction", "business_model",
    "cac_ltv_ratio", "funding_stage",
]
_ACTION_TEMPLATES = {
    "goal": "Lower your funding goal from $X to $Y",
    "campaign_duration": "Shorten your campaign from X to Y days",
    "launch_weekday": "Launch on Tuesday instead of X",
    "title": "Add a number to your title",
}


class CounterfactualSerializer:
    """Persist + verify the counterfactual engine definition (config + hashes).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.cf.serialization")

        d8 = self.config.get("day8", {})
        c = d8.get("serialization", {})
        self.version = str(c.get("version", "1.0.0"))
        self.manifest_path = Path(c.get("manifest_path",
                                        "artifacts/counterfactual/counterfactual_engine.json"))
        self.dependencies = {k: Path(v) for k, v in c.get("dependencies", {}).items()}
        self.artifacts = {k: Path(v) for k, v in c.get("artifacts", {}).items()}
        self._d8 = d8
        self.manifest_: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ public
    def build(self) -> "CounterfactualSerializer":
        """Assemble the engine manifest."""
        engine_config = self._engine_config()
        templates = self._templates()
        self.manifest_ = {
            "version": self.version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "architecture_layer": 7,
            "note": ("Counterfactual engine = configuration over the Day-4 model + Day-7 "
                     "patterns. Model-simulated what-ifs on mutable features; not causal."),
            "engine_config": engine_config,
            "recommendation_templates": templates,
            "content_hash": self._content_hash(engine_config, templates),
            "dependencies": {name: self._file_entry(p) for name, p in self.dependencies.items()},
            "artifacts": {name: self._file_entry(p) for name, p in self.artifacts.items()},
            "deferred_features": _DEFERRED_FEATURES,
        }
        self.logger.info("Built counterfactual engine manifest v%s (content %s…).",
                         self.version, self.manifest_["content_hash"][:12])
        return self

    def save(self) -> Path:
        """Persist the manifest JSON."""
        if self.manifest_ is None:
            self.build()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest_, indent=2))
        self.logger.info("Saved counterfactual engine -> %s", self.manifest_path)
        return self.manifest_path

    def verify(self) -> dict[str, Any]:
        """Re-hash dependencies, artifacts, and the live config; detect drift."""
        m = self.manifest_ if self.manifest_ is not None else json.loads(self.manifest_path.read_text())
        checks: dict[str, bool] = {}
        for group in ("dependencies", "artifacts"):
            for name, entry in m.get(group, {}).items():
                p = Path(entry["path"])
                cur = self._sha256_file(p) if p.exists() else None
                checks[f"{group}.{name}"] = (cur == entry["sha256"])
        # config drift: recompute the content hash from the CURRENT config.
        checks["content_hash"] = (
            self._content_hash(self._engine_config(), self._templates()) == m["content_hash"])
        all_ok = all(checks.values())
        self.logger.info("Counterfactual verify: %s (%d/%d checks pass).",
                         "OK" if all_ok else "DRIFT", sum(checks.values()), len(checks))
        return {"all_ok": all_ok, "checks": checks, "version": m["version"]}

    def describe(self) -> str:
        """One-line summary of the engine."""
        m = self.manifest_ or json.loads(self.manifest_path.read_text())
        t = m["recommendation_templates"]
        return (f"Counterfactual engine v{m['version']} (content {m['content_hash'][:12]}…): "
                f"mutable {t['mutable_features']}, immutable {t['immutable_features']}; "
                f"{len(m['dependencies'])} deps hashed, {len(m['deferred_features'])} features deferred.")

    # --------------------------------------------------------------- internals
    def _engine_config(self) -> dict[str, Any]:
        return {k: self._d8.get(k) for k in
                ("candidates", "simulator", "evidence", "ranking", "report")}

    def _templates(self) -> dict[str, Any]:
        cand = self._d8.get("candidates", {})
        rank = self._d8.get("ranking", {})
        return {
            "mutable_features": cand.get("mutable"),
            "immutable_features": cand.get("immutable"),
            "tier_weights": rank.get("tier_weights"),
            "feasibility": rank.get("feasibility"),
            "action_templates": _ACTION_TEMPLATES,
        }

    def _content_hash(self, engine_config, templates) -> str:
        blob = json.dumps({"engine_config": engine_config, "templates": templates},
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
