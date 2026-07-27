"""Phase B — Train / Validation / Test Strategy (Day-4).

Day 3 already produced a leakage-free, stratified 80/20 **train/test** split
(persisted indices, and the feature pipeline was fit on that train). Day 4 must
**not re-split** — doing so would create a fresh test set the pipeline never saw
fit-on-train for, and would let the final holdout drift.

Instead :class:`ModelDataSplitter`:

- loads the persisted Day-3 matrices (``X_train_selected`` / ``X_test_selected``
  + labels),
- carves a **validation set out of the training portion** (stratified) so we can
  select models / tune / calibrate *without ever touching test*,
- exposes a **stratified k-fold** splitter over the fit portion for CV.

Roles (never blurred):
- **fit**   — what models are trained on.
- **val**   — model selection, hyperparameter search, calibration.
- **test**  — the Day-3 holdout; scored **exactly once**, at the end (Phase J).

Stratification preserves the 40.38% base rate in every partition, so metrics on
each are comparable.

Example
-------
>>> data = ModelDataSplitter().load()
>>> data.X_fit.shape, data.X_val.shape, data.X_test.shape
((212137, 865), (53034, 865), (66293, 865))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utils.config import load_config
from src.utils.logger import get_logger


@dataclass
class ModelData:
    """The fit/val/test partitions + labels, ready for modeling."""

    X_fit: sp.csr_matrix
    y_fit: np.ndarray
    X_val: sp.csr_matrix
    y_val: np.ndarray
    X_test: sp.csr_matrix
    y_test: np.ndarray

    def summary(self) -> dict[str, Any]:
        def rate(y):
            return round(float(np.mean(y)), 4)

        return {
            "fit": {"n": self.X_fit.shape[0], "pos_rate": rate(self.y_fit)},
            "val": {"n": self.X_val.shape[0], "pos_rate": rate(self.y_val)},
            "test": {"n": self.X_test.shape[0], "pos_rate": rate(self.y_test)},
            "n_features": self.X_fit.shape[1],
        }


class ModelDataSplitter:
    """Load Day-3 matrices and produce fit/val/test partitions + a CV splitter.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.models.data_split")

        s = self.config.get("day4", {}).get("split", {})
        self.fused_dir = Path(s.get("fused_dir", "artifacts/fused_features"))
        self.val_size: float = float(s.get("val_size", 0.2))
        self.n_splits: int = int(s.get("n_splits", 5))
        self.random_state: int = int(s.get("random_state", 42))

    # ------------------------------------------------------------------ public
    def load(self) -> ModelData:
        """Load persisted train/test, carve a stratified validation set from train."""
        X_train = sp.load_npz(self.fused_dir / "X_train_selected.npz").tocsr()
        X_test = sp.load_npz(self.fused_dir / "X_test_selected.npz").tocsr()
        y_train = np.load(self.fused_dir / "y_train.npy")
        y_test = np.load(self.fused_dir / "y_test.npy")

        # Carve validation FROM train only (test is never re-partitioned).
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.val_size,
            stratify=y_train,          # preserve the 40/60 base rate
            random_state=self.random_state,
        )
        data = ModelData(
            X_fit=X_fit.tocsr(),
            y_fit=y_fit,
            X_val=X_val.tocsr(),
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
        self.logger.info(
            "Model data: fit %d / val %d / test %d rows, %d features; "
            "pos-rate fit=%.4f val=%.4f test=%.4f.",
            data.X_fit.shape[0], data.X_val.shape[0], data.X_test.shape[0],
            data.X_fit.shape[1],
            data.y_fit.mean(), data.y_val.mean(), data.y_test.mean(),
        )
        return data

    def cv(self) -> StratifiedKFold:
        """Stratified k-fold over the fit portion (for tuning / calibration)."""
        return StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
