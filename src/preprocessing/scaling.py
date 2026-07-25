"""Phase E — Feature Scaling (Day-3 Stage-2).

The final step of the numeric branch (``log -> cap -> scale``). After log1p and
winsorizing, ``goal`` lives on ~[4.6, 12.6] while an engineered feature like
``launch_month`` lives on [1, 12] and ``goal_per_day`` somewhere else entirely.
Gradient- and distance-based models (logistic regression, SVM, k-NN) treat a
feature's raw magnitude as its importance, so unscaled inputs let the
widest-range feature dominate. :class:`ScalingHandler` puts every numeric on a
common scale — by default zero mean / unit variance (``StandardScaler``).

**Scaling is model-dependent — hence a toggle.** Tree models (Day 4) split on
thresholds and are invariant to any monotonic rescaling, so they need no
scaling; logistic regression needs it badly. ``enabled: false`` makes the
handler a pass-through so the same pipeline serves both model families.

**Fit on train only.** The scaler learns location/scale (mean/std, or
median/IQR) from the training data; applying train statistics to the test set is
the whole point — the test set is scaled *as if* it were future data, and its
post-scaling mean is therefore only *approximately* 0, not exactly.

Available methods
-----------------
- ``standard``  (default): ``(x - mean) / std``            -> mean 0, var 1
- ``robust``    : ``(x - median) / IQR``                   -> outlier-resistant
- ``minmax``    : ``(x - min) / (max - min)``              -> [0, 1]
- ``maxabs``    : ``x / max(|x|)``                         -> [-1, 1], sparse-safe

Example
-------
>>> sc = ScalingHandler().fit(X_train_num)
>>> Xs = sc.transform(X_train_num)          # columns now ~mean 0 / var 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from src.utils.config import load_config
from src.utils.logger import get_logger

_SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
}


class ScalingHandler:
    """Fit a numeric scaler on train; apply it to any partition (or pass through).

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.preprocessing.scaling")

        sc = self.config.get("day3", {}).get("scaling", {})
        self.method: str = sc.get("method", "standard")
        self.enabled: bool = bool(sc.get("enabled", True))
        self.columns: list[str] = list(
            sc.get("columns", ["goal", "usd_goal_real", "goal_per_day"])
        )

        if self.method not in _SCALERS:
            raise ValueError(
                f"Unknown scaling method '{self.method}'; choose from {list(_SCALERS)}."
            )

        # Fitted state.
        self._cols_present: list[str] = []
        self._scaler = None
        self.metadata_: dict[str, Any] = {}
        self._fitted = False

    # -------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y: Any = None) -> "ScalingHandler":
        """Learn location/scale per column from **train** (no-op if disabled)."""
        self._cols_present = [c for c in self.columns if c in X.columns]

        if not self.enabled or not self._cols_present:
            self._fitted = True
            self.logger.info(
                "Scaling disabled or no columns present — pass-through."
                if not self.enabled
                else "No configured numeric columns present — pass-through."
            )
            return self

        self._scaler = _SCALERS[self.method]()
        block = X[self._cols_present].apply(pd.to_numeric, errors="coerce").astype(float)
        self._scaler.fit(block)
        self.metadata_ = self._describe_scaler()

        self._fitted = True
        self.logger.info(
            "Scaler fitted (%s) on %s.", self.method, self._cols_present
        )
        return self

    # -------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with numeric columns scaled (or unchanged if disabled)."""
        self._check_fitted()
        if not self.enabled or self._scaler is None or not self._cols_present:
            return X.copy()

        out = X.copy()
        block = out[self._cols_present].apply(pd.to_numeric, errors="coerce").astype(float)
        out[self._cols_present] = self._scaler.transform(block)
        return out

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path = "artifacts/scalers/scaler.joblib") -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        self.logger.info("Saved fitted scaler to %s", path)
        return path

    @staticmethod
    def load(path: str | Path = "artifacts/scalers/scaler.joblib") -> "ScalingHandler":
        return joblib.load(Path(path))

    # --------------------------------------------------------------- internals
    def _describe_scaler(self) -> dict[str, Any]:
        """Expose the learned parameters per column for the audit trail."""
        s = self._scaler
        meta: dict[str, Any] = {"method": self.method}
        # Each scaler exposes center/scale under different attribute names.
        if hasattr(s, "mean_"):
            meta["center"] = dict(zip(self._cols_present, s.mean_.tolist()))
        elif hasattr(s, "center_"):
            meta["center"] = dict(zip(self._cols_present, s.center_.tolist()))
        elif hasattr(s, "data_min_"):
            meta["center"] = dict(zip(self._cols_present, s.data_min_.tolist()))
        if hasattr(s, "scale_"):
            meta["scale"] = dict(zip(self._cols_present, s.scale_.tolist()))
        return meta

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("ScalingHandler must be fit() before transform().")
