"""Phase I — Reporting Engine Serialization (Day-13, Layer 13). The final module.

Like every engine since Day 4, the reporting system is a **configuration over existing outputs** —
deterministic report templates + a copilot over the Day-11 decision and Day-12 committee. So
serialization captures that policy and pins it to its dependencies, following the content-hash-as-
identity pattern:

- **engine config** — the Day-13 tunables (founder / investor / memo / executive / copilot / export /
  pipeline).
- **reporting policy** — the report types, export formats, copilot intents, and the NO-NEW-CLAIMS /
  provenance rule (how outputs become documents).
- **dependency hashes** — SHA-256 of the Day-4 model + the Day-11 recommendation engine + the Day-12
  agent system, pinning the whole platform.
- **artifact hashes** — the produced reports. **Report artifacts carry a ``generated_utc`` timestamp,
  so they are hashed on their CONTENT with the timestamp excluded** — regenerating a report (new
  timestamp, identical content) must not read as drift.
- **content_hash** — SHA-256 of the canonicalised config + policy = the engine's identity.

``verify()`` re-hashes everything (dependencies, content-of-artifacts, live config) to detect +
localize drift. **JSON, not pickle.** Deferred (fabricated) items are recorded.

Example
-------
>>> ser = ReportingSerializer().build()
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

# The honesty boundary — everything the reporting layer refuses to fabricate (deferred appendix).
_DEFERRED_ITEMS = [
    "founder_experience", "team_composition", "leadership", "advisory_board",
    "revenue", "mrr", "retention", "burn_rate", "runway", "cac_ltv", "unit_economics",
    "competitive_moat", "differentiation", "named_competitors", "market_size_tam",
    "expected_return", "irr", "valuation", "comparable_funding_rounds",
]
_REPORT_TYPES = ["Founder Intelligence Report", "Investor Due Diligence Report",
                 "Investment Memo", "Executive Summary"]
_COPILOT_INTENTS = ["probability", "improve", "similar", "why_fail", "risk",
                    "before_launch", "recommendation", "strengths", "deferred", "unknown"]


class ReportingSerializer:
    """Persist + verify the Day-13 reporting-engine definition (config + policy + hashes).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reporting.serialization")

        d13 = self.config.get("day13", {})
        c = d13.get("serialization", {})
        self.version = str(c.get("version", "1.0.0"))
        self.manifest_path = Path(c.get("manifest_path", "artifacts/reports/reporting_engine.json"))
        self.dependencies = {k: Path(v) for k, v in c.get("dependencies", {}).items()}
        self.artifacts = {k: Path(v) for k, v in c.get("artifacts", {}).items()}
        self._d13 = d13
        self.manifest_: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ public
    def build(self) -> "ReportingSerializer":
        """Assemble the reporting-engine manifest."""
        engine_config = self._engine_config()
        policy = self._policy()
        self.manifest_ = {
            "version": self.version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "architecture_layer": 13,
            "note": ("Reporting engine = deterministic report templates + a copilot over the Day-11 "
                     "decision + Day-12 committee. Presentation only (no new analysis); every "
                     "section names its source. No LLM. Model-simulated, not causal."),
            "engine_config": engine_config,
            "reporting_policy": policy,
            "content_hash": self._content_hash(engine_config, policy),
            "dependencies": {name: self._file_entry(p, content=False)
                             for name, p in self.dependencies.items()},
            "artifacts": {name: self._file_entry(p, content=True)     # content hash (ts-excluded)
                          for name, p in self.artifacts.items()},
            "deferred_items": _DEFERRED_ITEMS,
        }
        self.logger.info("Built reporting engine manifest v%s (content %s…).",
                         self.version, self.manifest_["content_hash"][:12])
        return self

    def save(self) -> Path:
        """Persist the manifest JSON."""
        if self.manifest_ is None:
            self.build()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest_, indent=2))
        self.logger.info("Saved reporting engine -> %s", self.manifest_path)
        return self.manifest_path

    def verify(self) -> dict[str, Any]:
        """Re-hash dependencies (file), artifacts (content, ts-excluded), and the live config."""
        m = self.manifest_ if self.manifest_ is not None else json.loads(self.manifest_path.read_text())
        checks: dict[str, bool] = {}
        for name, entry in m.get("dependencies", {}).items():
            p = Path(entry["path"])
            cur = self._sha256_file(p) if p.exists() else None
            checks[f"dependencies.{name}"] = (cur == entry["sha256"])
        for name, entry in m.get("artifacts", {}).items():
            cur = self._artifact_content_hash(Path(entry["path"]))
            checks[f"artifacts.{name}"] = (cur == entry["sha256"])
        checks["content_hash"] = (
            self._content_hash(self._engine_config(), self._policy()) == m["content_hash"])
        all_ok = all(checks.values())
        self.logger.info("Reporting verify: %s (%d/%d checks pass).",
                         "OK" if all_ok else "DRIFT", sum(checks.values()), len(checks))
        return {"all_ok": all_ok, "checks": checks, "version": m["version"]}

    def describe(self) -> str:
        """One-line summary of the reporting engine."""
        m = self.manifest_ or json.loads(self.manifest_path.read_text())
        return (f"Reporting engine v{m['version']} (content {m['content_hash'][:12]}…): "
                f"layer {m['architecture_layer']}; {len(m['reporting_policy']['report_types'])} report "
                f"types, {len(m['dependencies'])} deps hashed, {len(m['artifacts'])} artifacts "
                f"(content, ts-excluded), {len(m['deferred_items'])} items deferred.")

    # --------------------------------------------------------------- internals
    def _engine_config(self) -> dict[str, Any]:
        return {k: self._d13.get(k) for k in
                ("founder", "investor", "memo", "executive", "copilot", "export", "pipeline")}

    def _policy(self) -> dict[str, Any]:
        return {
            "report_types": _REPORT_TYPES,
            "export_formats": self._d13.get("export", {}).get("formats"),
            "copilot_intents": _COPILOT_INTENTS,
            "rule": ("NO NEW CLAIMS — presentation-only reformatting of the Day 4-12 outputs; every "
                     "section carries a source provenance tag; deferred items appended, not fabricated."),
            "no_llm": True,
        }

    def _content_hash(self, engine_config, policy) -> str:
        blob = json.dumps({"engine_config": engine_config, "policy": policy},
                          sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    def _file_entry(self, path: Path, content: bool) -> dict[str, Any]:
        h = (self._artifact_content_hash(path) if content else
             (self._sha256_file(path) if path.exists() else None))
        return {"path": str(path), "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "hash_mode": "content(ts-excluded)" if content else "file", "sha256": h}

    def _artifact_content_hash(self, path: Path) -> Optional[str]:
        """Hash a report artifact's CONTENT with ``generated_utc`` timestamps stripped."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except Exception:
            return self._sha256_file(path)
        blob = json.dumps(self._strip_ts(data), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    def _strip_ts(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._strip_ts(v) for k, v in obj.items() if k != "generated_utc"}
        if isinstance(obj, list):
            return [self._strip_ts(x) for x in obj]
        return obj

    @staticmethod
    def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
