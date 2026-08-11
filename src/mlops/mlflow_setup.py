"""Day 16 · Phase B — MLflow setup (LOCAL backend, no server).

Experiment tracking needs somewhere to *put* runs. The draft said "working MLflow server"; the honest
choice for a local, reproducible platform is a **local, serverless backend** — no service to run.
``mlflow ui`` is an *optional* local viewer, not a deployment.

**Backend note (honest deviation from the frozen `file:./mlruns`).** MLflow 3.15 put the plain
file store into *maintenance mode* — it refuses ``file:./mlruns`` unless you opt into a deprecated
backend. So we use the supported, still-local **SQLite** backend (``sqlite:///mlflow.db``) with a
local artifact directory (``mlartifacts/``). Same intent (local, serverless, reproducible,
git-ignored), supported mechanism.

This module does two things:

- resolves the configured backend + artifact URIs to **absolute, repo-rooted** paths, so every run
  lands in the *same* store no matter which directory the process was launched from;
- provides a ``run(...)`` context manager (and a module-level :func:`mlflow_run`) that opens a run
  under the configured experiment and closes it cleanly on exit.

Example
-------
>>> with mlflow_run("smoke") as active:            # doctest: +SKIP
...     mlflow.log_param("model", "lightgbm")
...     mlflow.log_metric("pr_auc", 0.688)
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import mlflow

from src.utils.config import load_config
from src.utils.logger import get_logger

_REPO_ROOT = Path(__file__).resolve().parents[2]


class MLflowSetup:
    """Configure a local file-backed MLflow tracking store + experiment.

    Parameters
    ----------
    config, logger:
        Standard config + logger. Reads ``day16.mlflow.{tracking_uri, experiment_name}``.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.mlops.mlflow")

        c = self.config.get("day16", {}).get("mlflow", {})
        self.tracking_uri = self._resolve_tracking_uri(c.get("tracking_uri", "sqlite:///mlflow.db"))
        self.artifact_uri = self._resolve_file_uri(c.get("artifact_location", "./mlartifacts"))
        self.experiment_name = c.get("experiment_name", "avip-venture-intelligence")
        self.experiment_id: Optional[str] = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _abs_repo_path(p: str) -> Path:
        """Resolve a possibly-relative path against the repo root (so cwd doesn't matter)."""
        q = Path(p)
        return q if q.is_absolute() else (_REPO_ROOT / p.lstrip("./")).resolve()

    def _resolve_tracking_uri(self, raw: str) -> str:
        """Repo-root a ``sqlite:///`` (or ``file:``) backend URI; pass http(s) through unchanged."""
        if raw.startswith("sqlite:///"):
            return f"sqlite:///{self._abs_repo_path(raw[len('sqlite:///'):])}"
        if raw.startswith("file:"):
            p = self._abs_repo_path(raw[len("file:"):]); p.mkdir(parents=True, exist_ok=True)
            return p.as_uri()
        return raw                                       # e.g. http://... (remote server, not used)

    def _resolve_file_uri(self, raw: str) -> str:
        """Repo-root a local artifact directory and return it as a ``file://`` URI."""
        p = self._abs_repo_path(raw[len("file:"):] if raw.startswith("file:") else raw)
        p.mkdir(parents=True, exist_ok=True)
        return p.as_uri()

    def setup(self) -> "MLflowSetup":
        """Point MLflow at the local backend and ensure the experiment exists (idempotent)."""
        mlflow.set_tracking_uri(self.tracking_uri)
        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:                                  # create with an explicit LOCAL artifact root
            self.experiment_id = mlflow.create_experiment(self.experiment_name,
                                                          artifact_location=self.artifact_uri)
        else:
            self.experiment_id = exp.experiment_id
        mlflow.set_experiment(self.experiment_name)
        self.logger.info("MLflow ready: tracking_uri=%s artifacts=%s experiment=%r (id %s)",
                         self.tracking_uri, self.artifact_uri, self.experiment_name,
                         self.experiment_id)
        return self

    # ------------------------------------------------------------------ run context
    @contextmanager
    def run(self, run_name: Optional[str] = None,
            tags: Optional[dict[str, Any]] = None) -> Iterator[Any]:
        """Open an MLflow run under the configured experiment; auto-close on exit."""
        self.setup()                                     # idempotent; safe to call per run
        with mlflow.start_run(run_name=run_name, tags=tags) as active:
            self.logger.info("MLflow run '%s' started (run_id %s)", run_name, active.info.run_id)
            yield active
        self.logger.info("MLflow run '%s' ended.", run_name)

    def describe(self) -> dict[str, Any]:
        """Small dict describing where runs are stored (for logging / notebooks)."""
        self.setup()
        return {"tracking_uri": self.tracking_uri, "artifact_uri": self.artifact_uri,
                "experiment_name": self.experiment_name, "experiment_id": self.experiment_id}


@contextmanager
def mlflow_run(run_name: Optional[str] = None, config: Optional[dict[str, Any]] = None,
               tags: Optional[dict[str, Any]] = None) -> Iterator[Any]:
    """Module-level convenience: ``with mlflow_run("name") as active: mlflow.log_param(...)``."""
    with MLflowSetup(config).run(run_name=run_name, tags=tags) as active:
        yield active
