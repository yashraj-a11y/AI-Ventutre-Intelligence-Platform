"""Phase H — Pipeline Serialization.

A *fitted* pipeline is part of the model: it carries learned state (the missing-value
fills) plus the exact transformation logic. To guarantee that training today and
inference months later apply **identical** transformations, we persist three artifacts:

- ``preprocessing_pipeline.joblib`` — the fitted pipeline object (joblib: fast, good
  for numpy/sklearn-style objects).
- ``feature_schema.json`` — the output contract: expected features, dtypes, dropped
  columns, target column, validation rules. Lets future inference data be checked.
- ``metadata.json`` — provenance: pipeline version, creation time, **dataset version
  (SHA-256 of the raw CSV)**, training shape, target definition, strategies.

Example
-------
>>> ser = PipelineSerializer()
>>> ser.save(pipeline, processed_df, raw_path="data/raw/ks-projects-201801.csv")
>>> pipe = ser.load()
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

# The business-rule contract, embedded in the schema for downstream reference.
_VALIDATION_RULES = [
    {"rule": "V1", "check": "goal > 0", "action": "drop"},
    {"rule": "V2", "check": "duration (days) > 0", "action": "flag"},
    {"rule": "V3", "check": "duration <= 90", "action": "flag"},
    {"rule": "V4", "check": "launched < deadline", "action": "drop"},
    {"rule": "V5", "check": "launched != 1970 epoch", "action": "drop"},
    {"rule": "V6", "check": "state in valid set", "action": "drop"},
    {"rule": "V7", "check": "country != 'N,0\"'", "action": "drop (quarantine)"},
    {"rule": "V8", "check": "required business fields present", "action": "flag"},
]


class PipelineSerializer:
    """Persist and restore the fitted preprocessing pipeline + schema + metadata."""

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        artifacts_dir: Optional[str] = None,
        logger=None,
    ) -> None:
        self.logger = logger or get_logger("avip.preprocessing.serialization")
        self.cfg = config if config is not None else load_config()
        self.artifacts_dir = Path(artifacts_dir or "artifacts/pipelines")
        self.pipeline_path = self.artifacts_dir / "preprocessing_pipeline.joblib"
        self.schema_path = self.artifacts_dir / "feature_schema.json"
        self.metadata_path = self.artifacts_dir / "metadata.json"

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def dataset_sha256(path: str) -> str:
        """SHA-256 of the raw dataset file — the frozen 'dataset version' convention."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    def build_schema(self, processed_df: pd.DataFrame) -> dict[str, Any]:
        cols = self.cfg.get("columns", {})
        dropped = list(cols.get("identifier", [])) + list(cols.get("leakage_drop", []))
        return {
            "version": self.cfg.get("version", "unknown"),
            "n_features": processed_df.shape[1],
            "features": list(processed_df.columns),
            "dtypes": {c: str(t) for c, t in processed_df.dtypes.items()},
            "target_column": self.cfg.get("target", {}).get("target_column", "target"),
            "dropped_columns": dropped,
            "validation_rules": _VALIDATION_RULES,
        }

    def build_metadata(
        self, processed_df: pd.DataFrame, raw_path: Optional[str]
    ) -> dict[str, Any]:
        tgt = self.cfg.get("target", {})
        meta = {
            "pipeline_version": self.cfg.get("version", "unknown"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "name": self.cfg.get("dataset", {}).get("name"),
                "raw_path": raw_path,
                "sha256": self.dataset_sha256(raw_path) if raw_path else None,
            },
            "training_shape": list(processed_df.shape),
            "target_definition": {
                "source_column": tgt.get("source_column"),
                "positive_state": tgt.get("positive_state"),
                "encoding": "successful -> 1, failed -> 0",
                "decided_states": tgt.get("decided_states"),
            },
            "missing_strategy": self.cfg.get("missing", {}),
            "dropped_features": (
                list(self.cfg.get("columns", {}).get("identifier", []))
                + list(self.cfg.get("columns", {}).get("leakage_drop", []))
            ),
        }
        return meta

    # ------------------------------------------------------------------- save
    def save(
        self,
        pipeline: Any,
        processed_df: pd.DataFrame,
        raw_path: Optional[str] = None,
    ) -> dict[str, str]:
        """Persist the pipeline, schema, and metadata; return the written paths."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, self.pipeline_path)
        schema = self.build_schema(processed_df)
        metadata = self.build_metadata(processed_df, raw_path)
        self.schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        self.logger.info(
            "Serialized pipeline -> %s (+ schema, metadata).", self.pipeline_path
        )
        return {
            "pipeline": str(self.pipeline_path),
            "schema": str(self.schema_path),
            "metadata": str(self.metadata_path),
        }

    # ------------------------------------------------------------------- load
    def load(self) -> Any:
        """Load the fitted pipeline object."""
        if not self.pipeline_path.is_file():
            raise FileNotFoundError(f"No pipeline at {self.pipeline_path}")
        return joblib.load(self.pipeline_path)

    def load_schema(self) -> dict[str, Any]:
        return json.loads(self.schema_path.read_text(encoding="utf-8"))

    def load_metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))
