"""Phase J — Multi-Agent System Serialization (Day-12, Layer 12).

Like every engine since Day 4, the multi-agent system is a **configuration over existing models** —
six deterministic agents + a consensus rule over the Day-11 intelligence. Serialization captures
that policy and pins it to its dependencies, following the content-hash-as-identity pattern:

- **engine config** — the Day-12 tunables (the six agents' cutoffs + orchestrator + consensus).
- **committee policy** — the consensus weights (incl. the C-7 down-weight), the band cutoffs, the
  risk-veto rule, and the confidence-not-inflated-by-count note (how stances become a decision).
- **dependency hashes** — SHA-256 of the Day-4 model + Day-8 counterfactual + Day-11 recommendation
  engine, pinning the whole orchestrated stack.
- **artifact hashes** — the six agent analyses + the collected reports + the committee decision.
- **content_hash** — SHA-256 of the canonicalised config + policy = the system's identity.

``verify()`` re-hashes everything (dependencies, artifacts, live config) to detect + localize drift.
**JSON, not pickle** — a policy over models, not a binary. Deferred (fabricated) items are recorded,
so the committee documents its own honesty boundary.

Example
-------
>>> ser = AgentSystemSerializer().build()
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

# The honesty boundary — everything the committee refuses to fabricate (aggregated across agents).
_DEFERRED_ITEMS = [
    "founder_experience", "team_composition", "leadership", "advisory_board",
    "burn_rate", "runway", "revenue", "cac_ltv", "unit_economics", "profitability",
    "market_size_tam", "industry_cagr", "competitive_moat", "differentiation",
    "named_competitors", "network_effects", "regulatory_risk", "legal_risk",
    "operational_risk", "expected_return", "irr", "valuation",
]
# The six agents' honest roles (documents what each perspective actually reads).
_AGENT_ROLES = {
    "MarketAgent": "category attractiveness & market timing (not TAM)",
    "FounderAgent": "creator's controllable campaign craft (not founder background)",
    "FinancialAgent": "funding viability = goal realism + odds (not P&L)",
    "CompetitionAgent": "cohort competition & positioning (not moat)",
    "RiskAgent": "risk & objections; carries the veto (not legal/regulatory)",
    "InvestmentAgent": "confidence-weighted synthesis of the five specialists",
}


class AgentSystemSerializer:
    """Persist + verify the Day-12 multi-agent system definition (config + policy + hashes).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.agents.serialization")

        d12 = self.config.get("day12", {})
        c = d12.get("serialization", {})
        self.version = str(c.get("version", "1.0.0"))
        self.manifest_path = Path(c.get("manifest_path", "artifacts/agents/agent_system_engine.json"))
        self.dependencies = {k: Path(v) for k, v in c.get("dependencies", {}).items()}
        self.artifacts = {k: Path(v) for k, v in c.get("artifacts", {}).items()}
        self._d12 = d12
        self.manifest_: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ public
    def build(self) -> "AgentSystemSerializer":
        """Assemble the multi-agent system manifest."""
        engine_config = self._engine_config()
        policy = self._policy()
        self.manifest_ = {
            "version": self.version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "architecture_layer": 12,
            "note": ("Multi-agent system = six deterministic specialist agents + a consensus rule "
                     "over the Day-11 intelligence. Correlated views of one signal set (not "
                     "independent experts); no LLM. Model-simulated, not causal."),
            "agent_roles": _AGENT_ROLES,
            "engine_config": engine_config,
            "committee_policy": policy,
            "content_hash": self._content_hash(engine_config, policy),
            "dependencies": {name: self._file_entry(p) for name, p in self.dependencies.items()},
            "artifacts": {name: self._file_entry(p) for name, p in self.artifacts.items()},
            "deferred_items": _DEFERRED_ITEMS,
        }
        self.logger.info("Built multi-agent system manifest v%s (content %s…).",
                         self.version, self.manifest_["content_hash"][:12])
        return self

    def save(self) -> Path:
        """Persist the manifest JSON."""
        if self.manifest_ is None:
            self.build()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest_, indent=2))
        self.logger.info("Saved multi-agent system -> %s", self.manifest_path)
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
        self.logger.info("Multi-agent verify: %s (%d/%d checks pass).",
                         "OK" if all_ok else "DRIFT", sum(checks.values()), len(checks))
        return {"all_ok": all_ok, "checks": checks, "version": m["version"]}

    def describe(self) -> str:
        """One-line summary of the system."""
        m = self.manifest_ or json.loads(self.manifest_path.read_text())
        return (f"Multi-agent system v{m['version']} (content {m['content_hash'][:12]}…): "
                f"layer {m['architecture_layer']}; {len(m['agent_roles'])} agents, "
                f"{len(m['dependencies'])} deps hashed, {len(m['artifacts'])} artifacts, "
                f"{len(m['deferred_items'])} items deferred.")

    # --------------------------------------------------------------- internals
    def _engine_config(self) -> dict[str, Any]:
        return {k: self._d12.get(k) for k in
                ("market", "founder", "financial", "competition", "risk",
                 "investment", "orchestrator", "consensus")}

    def _policy(self) -> dict[str, Any]:
        cons = self._d12.get("consensus", {})
        return {
            "consensus_weights": cons.get("weights"),
            "band_cutoffs": {"strong_at": cons.get("strong_at"), "proceed_at": cons.get("proceed_at"),
                             "monitor_at": cons.get("monitor_at")},
            "risk_veto": {"veto_confidence": cons.get("veto_confidence"),
                          "rule": "RiskAgent negative at >= veto_confidence caps a positive committee at Monitor Progress"},
            "committee_score": "sum(weight * confidence * stance) / sum(weight * confidence), stance in {-1,0,+1}",
            "confidence_note": "NOT inflated by agent count (correlated views); CompetitionAgent down-weighted for overlap",
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
