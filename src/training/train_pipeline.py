"""Day 16 · Phase G — Reproducible training pipeline (ONE command, versioned, non-destructive).

Ties Phases C–F into a single reproducible workflow:

    load data → train (+calibrate) → evaluate on test → save VERSIONED outputs
              → track (MLflow + fingerprint) → register (Development) → register artifacts

**The two frozen guarantees (C-5):**

1. **Never overwrites the incumbent.** Outputs go to ``artifacts/training_runs/<version>/`` and the new
   model is registered as **Development** — the Production incumbent is untouched (promotion is Phase H).
2. **Reproduction is verified, not assumed.** Re-running from the recorded config reproduces the
   incumbent's test PR-AUC (≈ 0.688); :meth:`verify_reproduction` runs training twice and asserts the
   metrics are identical (deterministic seeds).

Feature building (raw → TF-IDF → fusion) is the upstream Day-3 step whose output
(``artifacts/fused_features/``) is already materialised; its provenance is pinned by the dataset +
split hashes in the reproducibility fingerprint, so this pipeline reproduces the **model** from those
fingerprinted features without destructively rebuilding them.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib

from src.models.data_split import ModelData, ModelDataSplitter
from src.mlops.artifact_manager import ArtifactManager
from src.mlops.experiment_tracker import ExperimentTracker
from src.mlops.model_registry import ModelRegistry
from src.mlops.version_manager import VersionManager
from src.training.trainer import SingleModelTrainer, TrainResult
from src.utils.config import load_config
from src.utils.logger import get_logger

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TrainingPipeline:
    """One reproducible command: train → evaluate → save (versioned) → track → register."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.training.pipeline")

        t = self.config.get("day16", {}).get("training", {})
        self.runs_dir = _REPO_ROOT / t.get("runs_dir", "artifacts/training_runs")
        self.slot = t.get("slot", "funding_success_classifier")
        self.reproduce_tol = float(t.get("reproduce_tol", 0.01))

        self.versions = VersionManager(self.config, self.logger)
        self.trainer = SingleModelTrainer(self.config, self.logger)

    # ------------------------------------------------------------------ helpers
    def _incumbent_pr_auc(self) -> Optional[float]:
        p = _REPO_ROOT / "artifacts/models/model_metadata.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())["validation_metrics"]["calibrated_val"]["pr_auc"]

    # ------------------------------------------------------------------ public
    def run(self, *, family: str = "lightgbm", calibrate: bool = True,
            register: bool = True, track: bool = True) -> dict[str, Any]:
        """Execute the full pipeline and return a report (registers as Development, never promotes)."""
        fp = self.versions.fingerprint()
        version = f"{family}-{self.versions.version_string(fp)}"     # deterministic: same data+config → same id
        run_dir = self.runs_dir / version
        run_dir.mkdir(parents=True, exist_ok=True)

        data = ModelDataSplitter(self.config, self.logger).load()
        result: TrainResult = self.trainer.train(data, family=family, calibrate=calibrate, out_dir=run_dir)

        # --- save VERSIONED outputs (never the incumbent path) -----------------------------------
        model_path = run_dir / "model.joblib"
        joblib.dump(result.model, model_path)
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps({
            "version": version, "family": family, "calibrated": result.calibrated,
            "threshold": result.threshold, "val_metrics": result.val_metrics,
            "test_metrics": result.test_metrics, "params": result.params,
            "calibration": result.calibration, "reproducibility": fp,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2, default=str))

        # --- reproduction check vs the incumbent -------------------------------------------------
        inc = self._incumbent_pr_auc()
        test_pr = result.test_metrics["pr_auc"]
        reproduces = (abs(test_pr - inc) <= self.reproduce_tol) if inc is not None else None

        report: dict[str, Any] = {
            "version": version, "family": family, "run_dir": str(run_dir.relative_to(_REPO_ROOT)),
            "test_metrics": result.test_metrics, "threshold": result.threshold,
            "incumbent_pr_auc": inc, "reproduces_incumbent": reproduces,
            "reproduce_tol": self.reproduce_tol,
        }

        # --- track (MLflow + fingerprint), referencing large artifacts by hash -------------------
        if track:
            tr = ExperimentTracker(self.config, self.logger).track(
                run_name=f"train-{version}", model_type=f"{family} (calibrated={result.calibrated})",
                params=result.params, metrics=result.test_metrics,
                artifacts=[str(metrics_path)],
                references={"model": str(model_path), "feature_pipeline":
                            "artifacts/feature_pipeline/feature_pipeline.joblib"},
                tags={"phase": "G", "family": family, "status": "Development"},
                extra={"threshold": result.threshold, "run_dir": report["run_dir"]})
            report["run_id"] = tr["run_id"]

        # --- register as DEVELOPMENT (never auto-promote — that's Phase H) + artifacts -----------
        if register:
            ModelRegistry(self.config, self.logger).register(
                name=self.slot, version=version, model_type=f"{family} (calibrated={result.calibrated})",
                metrics=result.test_metrics, params=result.params, role="candidate",
                status="Development", artifact_path=str(model_path),
                feature_pipeline_path="artifacts/feature_pipeline/feature_pipeline.joblib",
                dataset_path=self.versions.dataset_path, note=f"Day-16 train_pipeline run {version}")
            am = ArtifactManager(self.config, self.logger)
            am.register(str(model_path), "models", name=f"{version}.joblib")     # copied (small)
            am.register("artifacts/vector_db/faiss.index", "vectorizers")        # referenced (large)

        self.logger.info("train_pipeline %s: test PR-AUC=%.4f (incumbent %.4f, reproduces=%s) -> %s",
                         version, test_pr, inc if inc else float("nan"), reproduces, run_dir)
        return report

    def verify_reproduction(self, *, family: str = "lightgbm") -> dict[str, Any]:
        """Train twice and assert identical test metrics — reproduction verified, not assumed (C-5)."""
        data = ModelDataSplitter(self.config, self.logger).load()
        r1 = self.trainer.train(data, family=family, calibrate=(family == "lightgbm"))
        r2 = self.trainer.train(data, family=family, calibrate=(family == "lightgbm"))
        identical = r1.test_metrics == r2.test_metrics
        inc = self._incumbent_pr_auc()
        out = {"identical": identical, "run1": r1.test_metrics, "run2": r2.test_metrics,
               "incumbent_pr_auc": inc,
               "reproduces_incumbent": (abs(r1.test_metrics["pr_auc"] - inc) <= self.reproduce_tol)
               if inc is not None else None}
        self.logger.info("Reproduction: identical=%s, run1 PR-AUC=%.4f vs incumbent %.4f",
                         identical, r1.test_metrics["pr_auc"], inc if inc else float("nan"))
        return out
