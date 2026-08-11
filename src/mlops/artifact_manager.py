"""Day 16 · Phase F — Artifact manager (content-hash registry: copy small, reference large).

Every artifact that shapes a prediction should be versioned — preprocessing pipeline, TF-IDF
vectorizer, encoders/scalers, feature pipeline, vector DB, prediction models, SHAP explainers. But
versioning ≠ blindly copying: the FAISS index is **407 MB** and the embeddings another **407 MB**.
Copying those per run is wasteful and pointless (Day-16 C-4). So this manager decides by size:

- **small** (≤ ``max_copy_mb``): copied into a managed store ``artifacts/store/<category>/<name>`` and
  indexed with mode ``copied`` (or ``in_place`` if it already lives under the store);
- **large** (> ``max_copy_mb``): **referenced** by ``sha256`` + path, mode ``referenced`` — bytes are
  never duplicated.

Either way an entry lands in ``artifacts/registry/artifact_index.json`` with its content hash and
size, so any run can point at *exactly* the bytes it used. Categories: models, pipelines, vectorizers,
explainers, reports. Hashing is delegated to :mod:`~src.mlops.version_manager` (the canonical home).
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.mlops.version_manager import file_reference, sha256_file
from src.utils.config import load_config
from src.utils.logger import get_logger

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CATEGORIES = ("models", "pipelines", "vectorizers", "explainers", "reports")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactManager:
    """Content-hash artifact registry — copies small artifacts, references large ones by sha256."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.mlops.artifacts")

        a = self.config.get("day16", {}).get("artifacts", {})
        self.index_path = _REPO_ROOT / a.get("index_path", "artifacts/registry/artifact_index.json")
        self.max_copy_bytes = int(float(a.get("max_copy_mb", 50)) * (1 << 20))
        self.store_root = _REPO_ROOT / a.get("store_root", "artifacts/store")
        self._data = self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> dict[str, Any]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {"artifacts": [], "updated_utc": None}

    def save(self) -> Path:
        self._data["updated_utc"] = _now()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self._data, indent=2, default=str))
        return self.index_path

    # ------------------------------------------------------------------ helpers
    def _find(self, name: str, category: str) -> Optional[dict[str, Any]]:
        return next((e for e in self._data["artifacts"]
                     if e["name"] == name and e["category"] == category), None)

    @staticmethod
    def _rel(p: Path) -> str:
        return str(p.relative_to(_REPO_ROOT)) if p.is_relative_to(_REPO_ROOT) else str(p)

    # ------------------------------------------------------------------ public
    def register(self, path: str, category: str, name: Optional[str] = None,
                 force_reference: bool = False, tags: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Register an artifact: copy it (small) or reference it by hash (large / ``force_reference``)."""
        if category not in _CATEGORIES:
            raise ValueError(f"category must be one of {_CATEGORIES}, got {category!r}")
        src = Path(path); src = src if src.is_absolute() else _REPO_ROOT / src
        if not src.exists():
            raise FileNotFoundError(src)
        name = name or src.name
        size = src.stat().st_size
        large = force_reference or size > self.max_copy_bytes

        if large:                                            # reference by content hash — never copy
            ref = file_reference(src)
            entry = {"mode": "referenced", "path": ref["path"], "sha256": ref["sha256"]}
        else:
            dest = self.store_root / category / name
            already_in_store = src.resolve().is_relative_to(self.store_root.resolve()) \
                if src.resolve().exists() else False
            if already_in_store:                             # already managed — index in place
                entry = {"mode": "in_place", "path": self._rel(src), "sha256": sha256_file(src)}
            else:                                            # copy into the managed store
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                entry = {"mode": "copied", "path": self._rel(dest), "sha256": sha256_file(dest),
                         "source": self._rel(src)}

        record = self._find(name, category) or {"name": name, "category": category,
                                                "created_utc": _now()}
        record.update({**entry, "size_bytes": size, "tags": tags or {}, "registered_utc": _now()})
        if record not in self._data["artifacts"] and self._find(name, category) is None:
            self._data["artifacts"].append(record)
        self.save()
        self.logger.info("Artifact '%s' [%s] %s (%.1f MB, sha %s)", name, category, entry["mode"],
                         size / (1 << 20), (entry["sha256"] or "na")[:12])
        return record

    def get(self, name: str, category: str) -> Optional[dict[str, Any]]:
        return self._find(name, category)

    def list(self, category: Optional[str] = None, mode: Optional[str] = None) -> list[dict[str, Any]]:
        return [e for e in self._data["artifacts"]
                if (category is None or e["category"] == category)
                and (mode is None or e["mode"] == mode)]

    def verify(self, name: str, category: str) -> bool:
        """Re-hash the on-disk file and confirm it still matches the recorded sha256."""
        e = self._find(name, category)
        if e is None:
            raise KeyError(f"no artifact {category}/{name}")
        current = sha256_file(_REPO_ROOT / e["path"])
        ok = current == e["sha256"]
        if not ok:
            self.logger.warning("Artifact %s/%s hash drift: recorded %s vs current %s",
                                category, name, (e["sha256"] or "na")[:12], (current or "na")[:12])
        return ok

    def describe(self) -> dict[str, Any]:
        """Counts by mode + bytes saved by referencing instead of copying."""
        by_mode: dict[str, int] = {}
        referenced_bytes = 0
        for e in self._data["artifacts"]:
            by_mode[e["mode"]] = by_mode.get(e["mode"], 0) + 1
            if e["mode"] == "referenced":
                referenced_bytes += e.get("size_bytes") or 0
        return {"n_artifacts": len(self._data["artifacts"]), "by_mode": by_mode,
                "bytes_saved_by_reference": referenced_bytes,
                "mb_saved_by_reference": round(referenced_bytes / (1 << 20), 1),
                "index": self._rel(self.index_path)}
