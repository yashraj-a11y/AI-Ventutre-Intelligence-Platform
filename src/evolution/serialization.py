"""Phase J — Evolution & Scenario Engine Serialization (Day-10, Layers 9/10).

Like the Day-8 counterfactual engine, the Day-10 planner is a **configuration over existing
models** — the Day-4 calibrated model and the Day-8 counterfactual engine — not a new trained
artifact. So serialization captures what *defines* the planner and pins it to its dependencies,
following the Day-4/6/8 content-hash-as-identity pattern:

- **engine config** — the Day-10 tunables (stage / roadmap / milestones / timeline / scenario /
  simulate / comparison + the two pipeline outputs).
- **readiness definition** — the stage boundaries (``strong_margin``, the four-way rule) and the
  canonical scenario apply-order, i.e. *how* the planner turns probabilities into stages and
  changes into scenarios.
- **dependency hashes** — SHA-256 of the Day-4 model + the Day-8 engine manifest, so the planner
  is pinned to the exact machinery it was defined against.
- **artifact hashes** — the produced evolution / scenario reports.
- **content_hash** — SHA-256 of the canonicalised config+definition = the planner's identity.

``verify()`` re-hashes everything (dependencies, artifacts, and the live config) to detect drift.
**JSON, not pickle** — a config, not a binary. Deferred (fabricated) features are recorded, so the
planner documents its own honesty boundary.

Example
-------
>>> ser = EvolutionSerializer().build()
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

# The honesty boundary — items the planner refuses to fabricate (aggregated from Phases B–H).
_DEFERRED_ITEMS = [
    "startup_lifecycle", "growth_milestones", "growth_timeline",
    "validate_customer_problem", "improve_retention", "grow_revenue",
    "expand_founding_team", "build_partnerships", "improve_unit_economics",
]
# The four-way funding-readiness rule (documents how probabilities become stages).
_READINESS_RULE = {
    "Strong": "base_p >= threshold + strong_margin",
    "Ready": "base_p >= threshold",
    "Fixable": "base_p < threshold <= optimized_p",
    "Not ready": "optimized_p < threshold",
}
# The canonical scenario apply-order (the verified fix for the deadline double-write).
_SCENARIO_APPLY_ORDER = ["goal", "shift launched to weekday",
                         "deadline = launched + duration", "title"]


class EvolutionSerializer:
    """Persist + verify the Day-10 evolution/scenario planner definition (config + hashes).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.evolution.serialization")

        d10 = self.config.get("day10", {})
        c = d10.get("serialization", {})
        self.version = str(c.get("version", "1.0.0"))
        self.manifest_path = Path(c.get("manifest_path",
                                        "artifacts/evolution/evolution_planner_engine.json"))
        self.dependencies = {k: Path(v) for k, v in c.get("dependencies", {}).items()}
        self.artifacts = {k: Path(v) for k, v in c.get("artifacts", {}).items()}
        self._d10 = d10
        self.manifest_: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ public
    def build(self) -> "EvolutionSerializer":
        """Assemble the planner manifest."""
        engine_config = self._engine_config()
        definition = self._definition()
        self.manifest_ = {
            "version": self.version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "architecture_layers": [9, 10],
            "note": ("Evolution/scenario planner = configuration over the Day-4 model + Day-8 "
                     "counterfactual engine. Funding-readiness stages + pre-launch prep + "
                     "model-simulated multi-variable scenarios; not a causal growth plan."),
            "engine_config": engine_config,
            "readiness_definition": definition,
            "content_hash": self._content_hash(engine_config, definition),
            "dependencies": {name: self._file_entry(p) for name, p in self.dependencies.items()},
            "artifacts": {name: self._file_entry(p) for name, p in self.artifacts.items()},
            "deferred_items": _DEFERRED_ITEMS,
        }
        self.logger.info("Built evolution planner manifest v%s (content %s…).",
                         self.version, self.manifest_["content_hash"][:12])
        return self

    def save(self) -> Path:
        """Persist the manifest JSON."""
        if self.manifest_ is None:
            self.build()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest_, indent=2))
        self.logger.info("Saved evolution planner -> %s", self.manifest_path)
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
            self._content_hash(self._engine_config(), self._definition()) == m["content_hash"])
        all_ok = all(checks.values())
        self.logger.info("Evolution planner verify: %s (%d/%d checks pass).",
                         "OK" if all_ok else "DRIFT", sum(checks.values()), len(checks))
        return {"all_ok": all_ok, "checks": checks, "version": m["version"]}

    def describe(self) -> str:
        """One-line summary of the planner."""
        m = self.manifest_ or json.loads(self.manifest_path.read_text())
        return (f"Evolution planner v{m['version']} (content {m['content_hash'][:12]}…): "
                f"layers {m['architecture_layers']}; {len(m['dependencies'])} deps hashed, "
                f"{len(m['artifacts'])} artifacts, {len(m['deferred_items'])} items deferred.")

    # --------------------------------------------------------------- internals
    def _engine_config(self) -> dict[str, Any]:
        return {k: self._d10.get(k) for k in
                ("stage", "roadmap", "milestones", "timeline",
                 "scenario", "simulate", "comparison", "evolution", "scenario_report")}

    def _definition(self) -> dict[str, Any]:
        return {
            "readiness_rule": _READINESS_RULE,
            "strong_margin": self._d10.get("stage", {}).get("strong_margin"),
            "scenario_apply_order": _SCENARIO_APPLY_ORDER,
            "duration_target": self._d10.get("scenario", {}).get("duration_target"),
            "lead_weeks": self._d10.get("timeline", {}).get("lead_weeks"),
        }

    def _content_hash(self, engine_config, definition) -> str:
        blob = json.dumps({"engine_config": engine_config, "definition": definition},
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
