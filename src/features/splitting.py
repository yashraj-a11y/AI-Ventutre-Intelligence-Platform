"""Phase 0 — Train/Test Split (Day-3 Stage-2 leakage foundation).

Everything in Day 3 that *learns* — one-hot vocabularies, scaler mean/std,
TF-IDF idf weights, feature-selection thresholds — must be fit on **training
data only**. If any of those statistics see the test rows, information leaks
backward from the evaluation set into the model and the reported metrics become
optimistic: you measure how well the pipeline memorised the test set, not how
well it generalises.

:class:`DataSplitter` therefore runs *first*, before any transformer is fit. It

- separates the label (``target``) from the feature frame and drops the leak
  column ``state`` — ``state`` *is* the target (``successful`` -> 1), so leaving
  it in features would let any model score a trivial, meaningless 100%;
- performs a **stratified** split so the ~40/60 class balance is preserved in
  both partitions (a random split can drift the minority rate, biasing both
  training and evaluation);
- persists the split indices + provenance so the exact same partition is
  reproducible on any later run and every phase splits identically.

The splitter never transforms feature values; it only partitions rows.

Example
-------
>>> from src.features.splitting import DataSplitter
>>> import pandas as pd
>>> df = pd.read_csv("data/processed/kickstarter_processed.csv")
>>> result = DataSplitter().split(df)          # persists artifacts/splits/*
>>> result.X_train.shape, result.y_train.mean()
((265171, 10), 0.404...)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import load_config
from src.utils.logger import get_logger


class DataSplitError(Exception):
    """Raised when the input frame cannot be split (missing/degenerate target)."""


@dataclass
class SplitResult:
    """Outcome of a train/test split: the four partitions + reproducible indices.

    ``X_*`` exclude the target and any leak columns; ``y_*`` are the label Series.
    ``train_index`` / ``test_index`` are the original dataframe index labels,
    persisted so the partition can be reconstructed exactly.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_index: np.ndarray
    test_index: np.ndarray
    feature_columns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Compact, log/JSON-friendly view of the split."""
        return {
            "n_train": int(len(self.X_train)),
            "n_test": int(len(self.X_test)),
            "n_features": int(self.X_train.shape[1]),
            "train_pos_rate": round(float(self.y_train.mean()), 6),
            "test_pos_rate": round(float(self.y_test.mean()), 6),
        }


class DataSplitter:
    """Stratified train/test splitter — the first, fit-free step of Day 3.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.features.splitting")

        split_cfg = self.config.get("day3", {}).get("split", {})
        self.test_size: float = float(split_cfg.get("test_size", 0.2))
        self.random_state: int = int(split_cfg.get("random_state", 42))
        self.stratify: bool = bool(split_cfg.get("stratify", True))
        self.target_column: str = split_cfg.get("target_column", "target")
        self.leak_columns: list[str] = list(split_cfg.get("leak_columns", []))
        self.artifacts_dir = Path(split_cfg.get("artifacts_dir", "artifacts/splits"))

    # ------------------------------------------------------------------ public
    def split(self, df: pd.DataFrame, save: bool = True) -> SplitResult:
        """Split ``df`` into stratified train/test partitions.

        Parameters
        ----------
        df:
            The Day-2 processed dataframe (not modified).
        save:
            If ``True``, persist indices + metadata under ``artifacts_dir``.
        """
        self._validate(df)

        y = df[self.target_column]
        # Drop the label AND any leak columns from the feature matrix.
        drop_cols = [self.target_column] + [c for c in self.leak_columns if c in df.columns]
        X = df.drop(columns=drop_cols)

        stratify_on = y if self.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_on,
        )

        result = SplitResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_index=X_train.index.to_numpy(),
            test_index=X_test.index.to_numpy(),
            feature_columns=list(X.columns),
            metadata=self._build_metadata(df, X, y_train, y_test, drop_cols),
        )

        self._log(result)
        if save:
            self._save(result)
        return result

    # --------------------------------------------------------------- internals
    def _validate(self, df: pd.DataFrame) -> None:
        if df is None or df.shape[0] == 0:
            raise DataSplitError("Cannot split an empty dataframe.")
        if self.target_column not in df.columns:
            raise DataSplitError(
                f"Target column '{self.target_column}' is absent from the dataframe."
            )
        target = df[self.target_column]
        if target.isna().any():
            raise DataSplitError(
                f"Target '{self.target_column}' contains {int(target.isna().sum())} "
                "NaN value(s); resolve before splitting."
            )
        n_classes = target.nunique(dropna=True)
        if n_classes < 2:
            raise DataSplitError(
                f"Target '{self.target_column}' has {n_classes} class(es); "
                "a stratified split needs at least 2."
            )

    def _build_metadata(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        drop_cols: list[str],
    ) -> dict[str, Any]:
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_version": self.config.get("version", "unknown"),
            "n_total": int(len(df)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "test_size": self.test_size,
            "random_state": self.random_state,
            "stratify": self.stratify,
            "target_column": self.target_column,
            "dropped_from_features": drop_cols,
            "feature_columns": list(X.columns),
            "class_balance": {
                "overall": _rate(df[self.target_column]),
                "train": _rate(y_train),
                "test": _rate(y_test),
            },
        }

    def _log(self, result: SplitResult) -> None:
        s = result.summary()
        self.logger.info(
            "Split complete — %d train / %d test rows, %d features; "
            "pos-rate train=%.4f test=%.4f.",
            s["n_train"],
            s["n_test"],
            s["n_features"],
            s["train_pos_rate"],
            s["test_pos_rate"],
        )

    def _save(self, result: SplitResult) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.artifacts_dir / "train_index.npy", result.train_index)
        np.save(self.artifacts_dir / "test_index.npy", result.test_index)
        meta_path = self.artifacts_dir / "split_metadata.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(result.metadata, fh, indent=2)
        self.logger.info("Persisted split indices + metadata to %s", self.artifacts_dir)

    # ----------------------------------------------------------- reproduce API
    def load_split(self, df: pd.DataFrame) -> SplitResult:
        """Reconstruct a previously-persisted split by re-applying saved indices.

        Guarantees any later phase partitions ``df`` identically to the run that
        produced the artifacts, without re-invoking the RNG.
        """
        train_index = np.load(self.artifacts_dir / "train_index.npy", allow_pickle=True)
        test_index = np.load(self.artifacts_dir / "test_index.npy", allow_pickle=True)
        with (self.artifacts_dir / "split_metadata.json").open(encoding="utf-8") as fh:
            metadata = json.load(fh)

        drop_cols = metadata.get("dropped_from_features", [self.target_column])
        y = df[self.target_column]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        return SplitResult(
            X_train=X.loc[train_index],
            X_test=X.loc[test_index],
            y_train=y.loc[train_index],
            y_test=y.loc[test_index],
            train_index=train_index,
            test_index=test_index,
            feature_columns=list(X.columns),
            metadata=metadata,
        )


def _rate(y: pd.Series) -> dict[str, Any]:
    """Class distribution as {count, positive_rate} for logging/metadata."""
    counts = y.value_counts(dropna=False).to_dict()
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "positive_rate": round(float(pd.to_numeric(y, errors="coerce").mean()), 6),
    }
