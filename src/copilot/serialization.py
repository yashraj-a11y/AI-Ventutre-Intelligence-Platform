"""Phase I — Venture Copilot Serialization (Day-9, Layer 8).

The copilot, like the counterfactual engine, is a **configuration over prior layers** (Day-4
model, Day-5 SHAP, Day-7 patterns, Day-8 counterfactuals) — not a new trained artifact. So we
serialize what defines it and pin its dependencies (Day-4/6/8 content-hash pattern):

- **engine config** — the tunables (objections / risk / swot / market / competition /
  recommendations / pipeline).
- **reasoning templates** — the objection *dimensions*, risk *categories*, SWOT *quadrant
  sources*, and the ``deterministic`` / ``no_llm`` flags (how the copilot reasons).
- **dependency hashes** — Day-4 model, Day-5 base model, Day-7 patterns, Day-8 engine.
- **content_hash** — SHA-256 of the canonicalised config+templates = the copilot's identity.
- **deferred_dimensions** — the fabricated analyses (named competitors, market size, founder,
  financials) the copilot will never produce.

``verify()`` re-hashes dependencies, artifacts, and the live config. **JSON, not pickle.**

Example
-------
>>> ser = CopilotSerializer().build()
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

_TEMPLATES = {
    "deterministic": True,
    "no_llm": True,
    "objection_dimensions": ["funding_odds", "goal_size", "category_fit", "timing", "cohort"],
    "risk_categories": ["funding_risk", "financial_goal_risk", "category_fit_risk",
                        "timing_risk", "cohort_risk"],
    "swot_sources": {"strengths": "Day-5 SHAP+", "weaknesses": "Day-5 SHAP-",
                     "opportunities": "Day-8 counterfactual levers", "threats": "Day-7 headwinds"},
    "market_reframe": "category/segment analysis (own data)",
    "competition_reframe": "category competition (own data)",
    "recommendation_source": "Day-8 counterfactual + strategic framing",
}
_DEFERRED_DIMENSIONS = [
    "named_competitors", "market_size_tam_sam_som", "market_cagr", "founder_team_quality",
    "traction", "unit_economics_burn_retention", "business_model", "moat_defensibility",
    "regulatory_risk", "technology_risk",
]


class CopilotSerializer:
    """Persist + verify the venture-copilot definition (config + templates + dependency hashes).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.copilot.serialization")

        d9 = self.config.get("day9", {})
        c = d9.get("serialization", {})
        self.version = str(c.get("version", "1.0.0"))
        self.manifest_path = Path(c.get("manifest_path", "artifacts/copilot/venture_copilot.json"))
        self.dependencies = {k: Path(v) for k, v in c.get("dependencies", {}).items()}
        self.artifacts = {k: Path(v) for k, v in c.get("artifacts", {}).items()}
        self._d9 = d9
        self.manifest_: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ public
    def build(self) -> "CopilotSerializer":
        """Assemble the copilot manifest."""
        engine_config = self._engine_config()
        self.manifest_ = {
            "version": self.version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "architecture_layer": 8,
            "note": ("Venture Copilot = deterministic templates over the Day-4 model + Day-5 "
                     "SHAP + Day-7 patterns + Day-8 counterfactuals. No LLM. Model predictions "
                     "+ historical associations reframed as VC-style reasoning, not ground truth."),
            "engine_config": engine_config,
            "reasoning_templates": _TEMPLATES,
            "content_hash": self._content_hash(engine_config, _TEMPLATES),
            "dependencies": {name: self._file_entry(p) for name, p in self.dependencies.items()},
            "artifacts": {name: self._file_entry(p) for name, p in self.artifacts.items()},
            "deferred_dimensions": _DEFERRED_DIMENSIONS,
        }
        self.logger.info("Built copilot manifest v%s (content %s…).",
                         self.version, self.manifest_["content_hash"][:12])
        return self

    def save(self) -> Path:
        """Persist the manifest JSON."""
        if self.manifest_ is None:
            self.build()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest_, indent=2))
        self.logger.info("Saved venture copilot -> %s", self.manifest_path)
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
        checks["content_hash"] = (self._content_hash(self._engine_config(), _TEMPLATES)
                                  == m["content_hash"])
        all_ok = all(checks.values())
        self.logger.info("Copilot verify: %s (%d/%d checks pass).",
                         "OK" if all_ok else "DRIFT", sum(checks.values()), len(checks))
        return {"all_ok": all_ok, "checks": checks, "version": m["version"]}

    def describe(self) -> str:
        """One-line summary of the copilot."""
        m = self.manifest_ or json.loads(self.manifest_path.read_text())
        t = m["reasoning_templates"]
        return (f"Venture Copilot v{m['version']} (content {m['content_hash'][:12]}…, "
                f"deterministic={t['deterministic']}, no_llm={t['no_llm']}): "
                f"{len(t['objection_dimensions'])} objection dims, {len(t['risk_categories'])} risk "
                f"categories; {len(m['dependencies'])} deps hashed, "
                f"{len(m['deferred_dimensions'])} dimensions deferred.")

    # --------------------------------------------------------------- internals
    def _engine_config(self) -> dict[str, Any]:
        return {k: self._d9.get(k) for k in
                ("objections", "risk", "swot", "market", "competition",
                 "recommendations", "copilot_pipeline")}

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
