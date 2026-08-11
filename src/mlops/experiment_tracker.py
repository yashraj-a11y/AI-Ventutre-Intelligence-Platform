"""Day 16 · Phase C — Experiment tracking (log one run as data).

Wrap a single training run so it becomes *reproducible evidence*, not a forgotten side effect. For one
run we log, to the local MLflow store:

- **params** — the model's *real* hyperparameters (LogReg: ``C``/``penalty``; LightGBM: ``num_leaves``
  etc.). No invented "batch size" for models that don't have one.
- **metrics** — the six we actually compute on the **held-out test split** (accuracy, precision,
  recall, F1, ROC-AUC, PR-AUC).
- **artifacts** — small files uploaded (the model, the report); large ones (FAISS index, etc.) are
  logged by **content-hash reference**, never copied (Day-16 C-4).
- **reproducibility fingerprint** — dataset hash + split hash + seed + git SHA + library versions, so
  a run can be *re-derived*, not just remembered.

Every run is also mirrored to ``artifacts/reports/experiment_<name>.json`` (metrics-as-data), so the
record survives independently of the MLflow store.

Example
-------
>>> ExperimentTracker().track(                      # doctest: +SKIP
...     run_name="lightgbm-incumbent",
...     params={"num_leaves": 127, "n_estimators": 300},
...     metrics={"pr_auc": 0.688, "roc_auc": 0.777})["run_id"]
'...'
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import mlflow

from src.mlops.mlflow_setup import MLflowSetup
from src.mlops.version_manager import VersionManager, sha256_file as _sha256_file  # canonical (Phase F)
from src.utils.config import load_config
from src.utils.logger import get_logger

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PathLike = Union[str, Path]


class ExperimentTracker:
    """Log a training run's params, metrics, artifacts, and reproducibility fingerprint.

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    setup:
        Optional pre-built :class:`~src.mlops.mlflow_setup.MLflowSetup` (shares the tracking store).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None,
                 setup: Optional[MLflowSetup] = None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.mlops.tracker")
        self.setup = setup or MLflowSetup(self.config, self.logger)

        self.reports_dir = _REPO_ROOT / "artifacts/reports"
        self.versions = VersionManager(self.config, self.logger)   # canonical fingerprint (Phase F)

    # ------------------------------------------------------------------ fingerprint
    def reproducibility_fingerprint(self) -> dict[str, Any]:
        """The six dimensions that must be pinned to re-derive a run (delegates to VersionManager)."""
        return self.versions.fingerprint()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _flat_params(params: dict[str, Any]) -> dict[str, str]:
        """MLflow params must be scalar-ish; stringify and drop Nones."""
        return {k: str(v) for k, v in params.items() if v is not None}

    def _reference(self, path: _PathLike) -> dict[str, Any]:
        """Content-hash REFERENCE for a large artifact (logged, never uploaded) — Day-16 C-4."""
        p = Path(path)
        p = p if p.is_absolute() else _REPO_ROOT / p
        return {"path": str(p.relative_to(_REPO_ROOT)) if p.is_relative_to(_REPO_ROOT) else str(p),
                "sha256": _sha256_file(p)}

    # ------------------------------------------------------------------ public
    def track(self, *, run_name: str, params: dict[str, Any], metrics: dict[str, float],
              model_type: Optional[str] = None, artifacts: Optional[list[_PathLike]] = None,
              references: Optional[dict[str, _PathLike]] = None,
              tags: Optional[dict[str, Any]] = None,
              extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Log one run and mirror it to ``artifacts/reports/experiment_<run_name>.json``.

        ``artifacts`` are uploaded (small: model, report); ``references`` are large artifacts logged by
        sha256 reference (e.g. the FAISS index) — never copied.
        """
        fp = self.reproducibility_fingerprint()
        refs = {name: self._reference(p) for name, p in (references or {}).items()}
        report: dict[str, Any] = {
            "run_name": run_name, "model_type": model_type,
            "params": params, "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "artifacts": [str(a) for a in (artifacts or [])], "references": refs,
            "reproducibility": fp, "extra": extra or {},
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

        with self.setup.run(run_name=run_name, tags=tags) as active:
            run_id = active.info.run_id
            mlflow.log_params(self._flat_params(params))
            mlflow.log_param("random_state", fp["seed"])
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            # fingerprint + references travel as tags (queryable, tiny) ...
            mlflow.set_tags({
                "model_type": model_type or "unknown",
                "dataset_sha256": (fp["dataset"]["sha256"] or "na")[:12],
                "split_train_sha256": (fp["split"]["train_sha256"] or "na")[:12],
                "git_commit": (fp["git"]["commit"] or "na")[:12],
                "git_dirty": str(fp["git"]["dirty"]),
                "python": fp["libraries"]["python"],
                **{f"ref.{n}": (r["sha256"] or "na")[:12] for n, r in refs.items()},
            })
            # ... and the full fingerprint is uploaded as its own artifact (auditable in detail)
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            fp_path = self.reports_dir / f"fingerprint_{self._safe(run_name)}.json"
            fp_path.write_text(json.dumps(fp, indent=2, default=str))
            mlflow.log_artifact(str(fp_path))
            for a in (artifacts or []):                  # small artifacts: uploaded
                ap = Path(a); ap = ap if ap.is_absolute() else _REPO_ROOT / ap
                if ap.exists():
                    mlflow.log_artifact(str(ap))
                else:
                    self.logger.warning("artifact not found, skipping upload: %s", ap)
            # metrics-as-data mirror (survives independently of the MLflow store)
            report["run_id"] = run_id
            out = self.reports_dir / f"experiment_{self._safe(run_name)}.json"
            out.write_text(json.dumps(report, indent=2, default=str))
            mlflow.log_artifact(str(out))

        self.logger.info("Tracked run '%s' (run_id %s): %s; mirror -> %s", run_name, run_id,
                         {k: report["metrics"].get(k) for k in ("f1", "pr_auc", "roc_auc")}, out)
        return report

    @staticmethod
    def _safe(name: str) -> str:
        """Filesystem-safe run name for the mirrored report."""
        return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "run"
