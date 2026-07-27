"""Phase I — Model Serialization & Versioning (Day-4).

A deployed model is more than its weights. To reproduce a prediction months from
now you also need: the exact **preprocessing** (the Day-3 feature pipeline), the
**decision threshold**, the **hyperparameters**, the **metrics** it was accepted
on, and the **data** it was trained on. :class:`ModelSerializer` bundles all of
that into a single self-describing ``model_metadata.json`` and content-hashes
every artifact so integrity/version can be verified later.

The metadata captures:
- **version** + creation timestamp,
- **model type** + calibration method + tuned **hyperparameters**,
- **validation metrics** + the **decision threshold**,
- a **pointer + SHA-256** to the feature pipeline (model and preprocessing are
  versioned *together* — a model is meaningless without its transforms),
- a **pointer + SHA-256** to the training dataset (provenance),
- library versions (reproducibility).

Content hashing (not timestamps) is the version identity: if the model bytes
change, the hash changes, so you can always tell whether two deployments run the
same artifact.

Example
-------
>>> meta = ModelSerializer().serialize()
>>> meta["decision_threshold"], meta["feature_pipeline"]["sha256"][:8]
(0.329, 'a1b2c3d4')
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.utils.config import load_config
from src.utils.logger import get_logger


class ModelSerializer:
    """Assemble + persist the self-describing model bundle (metadata + hashes)."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.models.serialization")

        d4 = self.config.get("day4", {})
        s = d4.get("serialization", {})
        self.version: str = str(s.get("version", "1.0.0"))
        self.model_dir = Path(s.get("model_dir", "artifacts/models"))
        self.metadata_path = Path(s.get("metadata_path", "artifacts/models/model_metadata.json"))
        self.feature_pipeline_path = Path(
            s.get("feature_pipeline_path", "artifacts/feature_pipeline/feature_pipeline.joblib")
        )
        self.dataset_path = Path(s.get("dataset_path", "data/processed/kickstarter_processed.csv"))

        # Upstream artifacts to fold into the bundle.
        self.calibrated_model_path = Path(
            d4.get("calibration", {}).get("model_path", "artifacts/models/calibrated_model.joblib")
        )
        self.best_config_path = Path(
            d4.get("tuning", {}).get("best_config_path", "artifacts/models/best_config.json")
        )
        self.calibration_json = Path(
            d4.get("calibration", {}).get("plot_path", "")
        ).parent / "calibration.json"

    # ------------------------------------------------------------------ public
    def serialize(self) -> dict[str, Any]:
        """Build and write model_metadata.json bundling model + pipeline + provenance."""
        best_config = self._read_json(self.best_config_path)
        calibration = self._read_json(self.calibration_json)

        metadata = {
            "version": self.version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "task": "binary classification — Kickstarter funding success",
            "model": {
                "type": "LightGBM (tuned) + CalibratedClassifierCV",
                "calibration_method": calibration.get("best_method"),
                "hyperparameters": best_config.get("best_params", {}),
                "artifact": str(self.calibrated_model_path),
                "sha256": self._sha256(self.calibrated_model_path),
            },
            "decision_threshold": self._threshold(calibration),
            "validation_metrics": self._metrics(best_config, calibration),
            "feature_pipeline": {
                "artifact": str(self.feature_pipeline_path),
                "sha256": self._sha256(self.feature_pipeline_path),
                "note": "model MUST be served with this exact pipeline (train/serve parity)",
            },
            "dataset": {
                "artifact": str(self.dataset_path),
                "sha256": self._sha256(self.dataset_path),
            },
            "environment": self._environment(),
        }

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        self.logger.info(
            "Serialized model bundle v%s -> %s (model sha %s).",
            self.version, self.metadata_path, metadata["model"]["sha256"][:8],
        )
        return metadata

    def verify(self) -> dict[str, bool]:
        """Re-hash the referenced artifacts and confirm they match the metadata."""
        meta = self._read_json(self.metadata_path)
        checks = {
            "model": self._sha256(Path(meta["model"]["artifact"])) == meta["model"]["sha256"],
            "feature_pipeline": self._sha256(Path(meta["feature_pipeline"]["artifact"]))
            == meta["feature_pipeline"]["sha256"],
            "dataset": self._sha256(Path(meta["dataset"]["artifact"])) == meta["dataset"]["sha256"],
        }
        self.logger.info("Bundle verification: %s", checks)
        return checks

    # --------------------------------------------------------------- internals
    @staticmethod
    def _sha256(path: Path) -> Optional[str]:
        if not path.exists():
            return None
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def _threshold(calibration: dict[str, Any]) -> Optional[float]:
        return calibration.get("threshold", {}).get("f1_optimal")

    @staticmethod
    def _metrics(best_config: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
        out = {"tuned_cv_pr_auc": best_config.get("best_cv_score")}
        results = calibration.get("results", {})
        best = calibration.get("best_method")
        if best and best in results:
            out["calibrated_val"] = {k: v for k, v in results[best].items() if not k.startswith("_")}
        return out

    @staticmethod
    def _environment() -> dict[str, str]:
        import sklearn
        env = {"python": platform.python_version(), "sklearn": sklearn.__version__}
        try:
            import lightgbm
            env["lightgbm"] = lightgbm.__version__
        except ImportError:
            pass
        return env
