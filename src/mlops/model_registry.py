"""Day 16 · Phase E — Model registry (file-backed source of truth for "what's deployed & why").

A folder of ``.joblib`` files can't answer "which model is in production, trained on what, scoring
how much?" The registry can. It's a single JSON file (``artifacts/registry/model_registry.json``)
holding one entry per **(name, version)**, where ``name`` is a *deployment slot* (e.g. the funding
classifier) and each version carries its metrics, real hyperparameters, artifact + feature-pipeline +
dataset **content hashes**, owner, and a **status** moving through the lifecycle:

    Development → Staging → Production → Archived

The core invariant: **at most one Production version per slot.** Promoting a version to Production
auto-archives the incumbent — so the registry can never claim two live models, and every transition
is recorded in a ``status_history``. The mandatory Logistic-Regression **baseline** is registered too
(``role="baseline"``, Archived — retained for comparison, not deployed), so every future candidate has
a floor to beat.

No server, no external registry service — just a versioned JSON file, consistent with the platform's
content-hash discipline.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.mlops.version_manager import sha256_file as _sha256_file   # canonical content-hash helper (Phase F)
from src.utils.config import load_config
from src.utils.logger import get_logger

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATUSES = ("Development", "Staging", "Production", "Archived")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    """A versioned, file-backed model registry with a Dev→Staging→Production→Archived lifecycle."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 path: Optional[str] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.mlops.registry")
        p = path or self.config.get("day16", {}).get("registry", {}).get(
            "path", "artifacts/registry/model_registry.json")
        self.path = _REPO_ROOT / p
        self._data = self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"entries": [], "updated_utc": None}

    def save(self) -> Path:
        self._data["updated_utc"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, default=str))
        return self.path

    # ------------------------------------------------------------------ helpers
    def _find(self, name: str, version: str) -> Optional[dict[str, Any]]:
        return next((e for e in self._data["entries"]
                     if e["name"] == name and e["version"] == version), None)

    @staticmethod
    def _hash_ref(path: Optional[str]) -> Optional[dict[str, Any]]:
        if not path:
            return None
        p = Path(path); p = p if p.is_absolute() else _REPO_ROOT / p
        rel = str(p.relative_to(_REPO_ROOT)) if p.is_relative_to(_REPO_ROOT) else str(p)
        return {"path": rel, "sha256": _sha256_file(p)}

    # ------------------------------------------------------------------ public: register
    def register(self, *, name: str, version: str, model_type: str,
                 metrics: dict[str, float], params: Optional[dict[str, Any]] = None,
                 artifact_path: Optional[str] = None, feature_pipeline_path: Optional[str] = None,
                 dataset_path: Optional[str] = None, owner: str = "avip",
                 status: str = "Development", role: Optional[str] = None,
                 references: Optional[dict[str, Any]] = None, note: Optional[str] = None,
                 extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Add or update a (name, version) entry with its metrics + content-hash provenance."""
        if status not in _STATUSES:
            raise ValueError(f"status must be one of {_STATUSES}, got {status!r}")

        existing = self._find(name, version)
        entry = existing or {"name": name, "version": version, "created_utc": _now(),
                             "status_history": []}
        entry.update({
            "model_type": model_type, "role": role, "owner": owner,
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "params": params or {},
            "artifact": self._hash_ref(artifact_path),
            "feature_pipeline": self._hash_ref(feature_pipeline_path),
            "dataset_version": (self._hash_ref(dataset_path) or {}).get("sha256"),
            "references": references or {}, "extra": extra or {},
            "registered_utc": _now(),
        })
        if existing is None:
            entry["status"] = status
            entry["status_history"].append({"status": status, "at": _now(), "note": note or "registered"})
            self._data["entries"].append(entry)
        else:                                              # keep lifecycle status; use set_status to move
            entry.setdefault("status", status)
        self.save()
        self.logger.info("Registered %s v%s (%s, status=%s, role=%s) PR-AUC=%.4f",
                         name, version, model_type, entry["status"], role,
                         entry["metrics"].get("pr_auc", float("nan")))
        return entry

    # ------------------------------------------------------------------ public: lifecycle
    def set_status(self, name: str, version: str, status: str, note: Optional[str] = None) -> dict[str, Any]:
        """Transition a version's status; promoting to Production auto-archives the incumbent."""
        if status not in _STATUSES:
            raise ValueError(f"status must be one of {_STATUSES}, got {status!r}")
        entry = self._find(name, version)
        if entry is None:
            raise KeyError(f"no registry entry for {name} v{version}")

        if status == "Production":                         # enforce single-Production-per-slot invariant
            for other in self._data["entries"]:
                if (other["name"] == name and other is not entry
                        and other.get("status") == "Production"):
                    other["status"] = "Archived"
                    other["status_history"].append(
                        {"status": "Archived", "at": _now(),
                         "note": f"auto-archived: superseded by v{version}"})
                    self.logger.info("Auto-archived %s v%s (superseded by v%s)",
                                     name, other["version"], version)

        entry["status"] = status
        entry["status_history"].append({"status": status, "at": _now(), "note": note or "manual"})
        self.save()
        self.logger.info("%s v%s -> %s", name, version, status)
        return entry

    def promote(self, name: str, version: str, to: str = "Production",
                note: Optional[str] = None) -> dict[str, Any]:
        """Alias for :meth:`set_status` (reads naturally as a promotion)."""
        return self.set_status(name, version, to, note=note)

    # ------------------------------------------------------------------ public: query
    def get(self, name: str, version: Optional[str] = None) -> Optional[dict[str, Any]]:
        """A specific version, or the most recently registered version of ``name``."""
        if version is not None:
            return self._find(name, version)
        entries = [e for e in self._data["entries"] if e["name"] == name]
        return max(entries, key=lambda e: e["registered_utc"], default=None)

    def production(self, name: str) -> Optional[dict[str, Any]]:
        """The current Production version for a slot (the model actually served)."""
        return next((e for e in self._data["entries"]
                     if e["name"] == name and e.get("status") == "Production"), None)

    def list(self, status: Optional[str] = None, name: Optional[str] = None) -> list[dict[str, Any]]:
        """All entries, optionally filtered by status and/or slot name."""
        return [e for e in self._data["entries"]
                if (status is None or e.get("status") == status)
                and (name is None or e["name"] == name)]

    def history(self, name: str, version: str) -> list[dict[str, Any]]:
        e = self._find(name, version)
        return list(e["status_history"]) if e else []

    def describe(self) -> dict[str, Any]:
        """Counts by status + the current Production version per slot."""
        by_status: dict[str, int] = {}
        for e in self._data["entries"]:
            by_status[e.get("status", "?")] = by_status.get(e.get("status", "?"), 0) + 1
        prod = {e["name"]: f"v{e['version']}" for e in self._data["entries"]
                if e.get("status") == "Production"}
        return {"n_entries": len(self._data["entries"]), "by_status": by_status,
                "production": prod, "path": str(self.path.relative_to(_REPO_ROOT))}
