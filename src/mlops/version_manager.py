"""Day 16 · Phase F — Version manager (content-hash helpers + reproducibility fingerprint).

By Phase F, two modules (the experiment tracker and the model registry) both hash files and pin
provenance. This module is their **canonical home** — one place that owns:

- **content hashing** — ``sha256_file`` (streamed, so a 50 MB CSV or 407 MB index doesn't load into
  memory) and ``file_reference`` (``{path, sha256, size_bytes}``), the "reference don't copy" primitive
  (Day-16 C-4);
- **the reproducibility fingerprint** — the six dimensions that re-derive a run (dataset hash, split
  hash, seed, git commit + dirty flag, library versions), plus a short deterministic ``version_string``
  usable as a run id.

Consolidating here means the tracker and registry share one implementation instead of drifting copies.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Optional

from src.utils.config import load_config
from src.utils.logger import get_logger

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Libraries whose version can change a fitted model or a metric.
_PINNED_LIBS = ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "mlflow")


# ---------------------------------------------------------------------- content hashing
def sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    """Streamed SHA-256 of a file (chunked so large artifacts don't load into memory). None if absent."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _rel(p: Path) -> str:
    p = p if p.is_absolute() else _REPO_ROOT / p
    return str(p.relative_to(_REPO_ROOT)) if p.is_relative_to(_REPO_ROOT) else str(p)


def file_reference(path: Path) -> dict[str, Any]:
    """A content-hash reference to a file: ``{path, sha256, size_bytes}`` — never copies bytes."""
    p = Path(path); p = p if p.is_absolute() else _REPO_ROOT / p
    return {"path": _rel(p), "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size if p.exists() else None}


# ---------------------------------------------------------------------- environment
def git_info() -> dict[str, Any]:
    """Current commit + dirty flag; degrades gracefully outside a git repo."""
    def _run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(["git", *args], cwd=_REPO_ROOT,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def lib_versions() -> dict[str, str]:
    """Pin the libraries whose version can change a model or metric."""
    out = {"python": platform.python_version()}
    for mod in _PINNED_LIBS:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "n/a"
    return out


class VersionManager:
    """Owns the reproducibility fingerprint (dataset+split+seed+git+env) and a version string."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.mlops.version")

        kb = self.config.get("day6", {}).get("knowledge_base", {})
        self.dataset_path = _REPO_ROOT / kb.get("processed_path", "data/processed/kickstarter_processed.csv")
        self.split_dir = _REPO_ROOT / "artifacts/splits"
        self.seed = int(self.config.get("day16", {}).get("random_state", 16))

    def fingerprint(self) -> dict[str, Any]:
        """The six dimensions that must be pinned to re-derive a run (Day-16 IMP-1)."""
        return {
            "dataset": {"path": _rel(self.dataset_path), "sha256": sha256_file(self.dataset_path)},
            "split": {"train_sha256": sha256_file(self.split_dir / "train_index.npy"),
                      "test_sha256": sha256_file(self.split_dir / "test_index.npy")},
            "seed": self.seed,
            "git": git_info(),
            "libraries": lib_versions(),
        }

    def version_string(self, fingerprint: Optional[dict[str, Any]] = None) -> str:
        """Short, deterministic id derived from the stable parts of the fingerprint (excludes dirty)."""
        fp = fingerprint or self.fingerprint()
        stable = {"dataset": fp["dataset"]["sha256"], "split": fp["split"],
                  "seed": fp["seed"], "git": fp["git"]["commit"], "libraries": fp["libraries"]}
        blob = json.dumps(stable, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]
