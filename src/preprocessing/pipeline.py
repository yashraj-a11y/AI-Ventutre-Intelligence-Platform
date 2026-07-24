"""Phase G — Target Definition & Pipeline Integration.

Two pieces:

- :class:`TargetBuilder` — filters to *decided* campaigns (``successful`` / ``failed``)
  and builds the binary ``target`` column (successful -> 1, failed -> 0). This is a
  row-changing Stage-1 operation.

- :class:`PreprocessingPipeline` — the Stage-1 **orchestrator**. It runs every handler
  in the frozen order with one call, times each stage, records how many rows each
  stage removed, and collects the validation + integrity sub-reports.

Frozen Stage-1 order::

    DatasetValidator          (validate; raise on failure)
      -> ColumnCleaner
      -> TextCleaner
      -> CategoryNormalizer
      -> ColumnRemover        (drop id + leakage)
      -> TargetBuilder        (filter decided, build target)
      -> MissingValueHandler
      -> DuplicateRemover
      -> DataTypeHandler
      -> DataIntegrityValidator

This is NOT an ``sklearn.Pipeline`` — several stages remove rows, which sklearn
forbids. It is a purpose-built orchestrator for the row-changing cleaning stage.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd

from src.preprocessing.cleaning import (
    CategoryNormalizer,
    ColumnCleaner,
    ColumnRemover,
    TextCleaner,
)
from src.preprocessing.datatype import DataTypeHandler
from src.preprocessing.duplicates import DuplicateRemover
from src.preprocessing.integrity import DataIntegrityValidator
from src.preprocessing.missing import MissingValueHandler
from src.preprocessing.validation import DatasetValidator
from src.utils.config import load_config
from src.utils.logger import get_logger


class TargetBuilder:
    """Filter to decided outcomes and build the binary target (row-changing)."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.logger = logger or get_logger("avip.preprocessing.pipeline")
        cfg = config if config is not None else load_config()
        tgt = cfg.get("target", {})
        self.source_column = tgt.get("source_column", "state")
        self.positive_state = tgt.get("positive_state", "successful")
        self.decided_states = list(tgt.get("decided_states", ["successful", "failed"]))
        self.target_column = tgt.get("target_column", "target")

    def fit(self, X: pd.DataFrame, y=None) -> "TargetBuilder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.source_column not in X.columns:
            self.logger.warning(
                "TargetBuilder: source column '%s' absent; skipping.", self.source_column
            )
            return X.copy()

        before = len(X)
        mask = X[self.source_column].isin(self.decided_states)
        out = X.loc[mask].copy()
        out[self.target_column] = (
            (out[self.source_column] == self.positive_state).astype(int)
        )
        out = out.reset_index(drop=True)

        pos = int(out[self.target_column].sum())
        self.logger.info(
            "TargetBuilder: kept %d/%d decided rows; target balance: %d positive "
            "(%.1f%%) / %d negative.",
            len(out), before, pos, 100 * pos / len(out) if len(out) else 0.0,
            len(out) - pos,
        )
        return out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class PreprocessingPipeline:
    """Stage-1 orchestrator: runs all cleaning handlers in the frozen order."""

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        mode: str = "train",
        logger=None,
    ) -> None:
        self.logger = logger or get_logger("avip.preprocessing.pipeline")
        cfg = config if config is not None else load_config()
        self.mode = mode

        self.validator = DatasetValidator(config=cfg, logger=self.logger)
        self.integrity = DataIntegrityValidator(config=cfg, mode=mode, logger=self.logger)
        self.steps: list[tuple[str, Any]] = [
            ("column_cleaner", ColumnCleaner(logger=self.logger)),
            ("text_cleaner", TextCleaner(logger=self.logger)),
            ("category_normalizer", CategoryNormalizer(config=cfg, logger=self.logger)),
            ("column_remover", ColumnRemover(config=cfg, logger=self.logger)),
            ("target_builder", TargetBuilder(config=cfg, logger=self.logger)),
            ("missing", MissingValueHandler(config=cfg, logger=self.logger)),
            ("duplicates", DuplicateRemover(logger=self.logger)),
            ("datatype", DataTypeHandler(config=cfg, logger=self.logger)),
            ("integrity", self.integrity),
        ]
        self.report_: dict[str, Any] = {}

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Run the full Stage-1 pipeline, fitting stateful handlers on ``X``."""
        t_start = time.perf_counter()

        # 1. Validate up front; a failing dataset never enters the pipeline.
        val = self.validator.validate(X, raise_on_fail=True)
        self.report_["validation"] = val.to_dict()

        out = X
        step_reports: dict[str, Any] = {}
        for name, handler in self.steps:
            t0 = time.perf_counter()
            rows_in = len(out)
            out = handler.fit_transform(out)
            step_reports[name] = {
                "rows_in": rows_in,
                "rows_out": len(out),
                "rows_removed": rows_in - len(out),
                "seconds": round(time.perf_counter() - t0, 3),
            }

        step_reports["integrity"]["report"] = self.integrity.report()
        self.report_["steps"] = step_reports
        self.report_["final_shape"] = list(out.shape)
        self.report_["total_seconds"] = round(time.perf_counter() - t_start, 3)

        self.logger.info(
            "PreprocessingPipeline done: %d -> %d rows, %d columns in %.2fs.",
            len(X), len(out), out.shape[1], self.report_["total_seconds"],
        )
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted pipeline to new data (reuses learned statistics)."""
        self.validator.validate(X, raise_on_fail=True)
        out = X
        for _, handler in self.steps:
            out = handler.transform(out)
        return out

    def get_report(self) -> dict[str, Any]:
        return self.report_
