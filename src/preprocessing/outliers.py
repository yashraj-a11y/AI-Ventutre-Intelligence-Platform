"""Phase D — Outlier Treatment (Day-3 Stage-2).

**Cap, never drop.** Stage 2 is row-preserving: a campaign with an extreme goal
is still a real campaign whose success we must predict, so we *winsorize* the
value to a learned fence rather than delete the row. (Impossible values — goal
<= 0, bad dates — were already dropped in Day-2 integrity; what remains here are
*legitimate but extreme* values.)

:class:`OutlierHandler` learns per-column fences on **train** and clips every
partition to them. Because the fences are quantiles of the training data, this
is a genuine fit-on-train step — computing them on the test set would leak the
test distribution into the transform (unlike the parameter-free log1p).

Runs *after* Phase C in the numeric branch (``log -> cap -> scale``). Capping
log-transformed values is deliberate: log already compresses the heavy right
tail, so the fences act on a near-symmetric variable and touch very few rows —
capping mops up the residual tail rather than doing the heavy lifting alone.

Detection methods
-----------------
- ``iqr`` (default): fences at ``[Q1 - k*IQR, Q3 + k*IQR]`` (Tukey, k=1.5).
- ``percentile``: fences at ``[q_low, q_high]`` quantiles (e.g. 1st/99th).

Example
-------
>>> capper = OutlierHandler().fit(X_train_logged)
>>> Xc = capper.transform(X_train_logged)     # values clipped, rows preserved
>>> capper.metadata_["goal"]["pct_capped"]
0.3...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger


class OutlierHandler:
    """Learn per-column fences on train; clip (winsorize) all partitions to them.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.preprocessing.outliers")

        oc = self.config.get("day3", {}).get("outliers", {})
        self.method: str = oc.get("method", "iqr")  # iqr | percentile
        self.iqr_multiplier: float = float(oc.get("iqr_multiplier", 1.5))
        self.lower_quantile: float = float(oc.get("lower_quantile", 0.01))
        self.upper_quantile: float = float(oc.get("upper_quantile", 0.99))
        self.columns: list[str] = list(
            oc.get("columns", ["goal", "usd_goal_real", "goal_per_day"])
        )

        # Fitted state.
        self._cols_present: list[str] = []
        self.bounds_: dict[str, tuple[float, float]] = {}
        self.metadata_: dict[str, dict[str, Any]] = {}
        self._fitted = False

    # -------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y: Any = None) -> "OutlierHandler":
        """Learn ``(lower, upper)`` fences per column from the **training** data."""
        self._cols_present = [c for c in self.columns if c in X.columns]
        self.bounds_ = {}
        self.metadata_ = {}

        for c in self._cols_present:
            s = pd.to_numeric(X[c], errors="coerce").astype(float)
            lower, upper = self._fences(s)
            self.bounds_[c] = (lower, upper)

            n_low = int((s < lower).sum())
            n_high = int((s > upper).sum())
            self.metadata_[c] = {
                "method": self.method,
                "lower": lower,
                "upper": upper,
                "n_capped_low": n_low,
                "n_capped_high": n_high,
                "pct_capped": round(100.0 * (n_low + n_high) / len(s), 4),
            }

        self._fitted = True
        self.logger.info(
            "Outlier fences fitted (%s) on %s.", self.method, self._cols_present
        )
        for c, m in self.metadata_.items():
            self.logger.info(
                "  %s: [%.4f, %.4f] — %.3f%% of train rows capped.",
                c, m["lower"], m["upper"], m["pct_capped"],
            )
        return self

    # -------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with configured columns clipped to fences (no rows removed)."""
        self._check_fitted()
        out = X.copy()
        for c in self._cols_present:
            lower, upper = self.bounds_[c]
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float).clip(lower, upper)
        return out

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # --------------------------------------------------------------- internals
    def _fences(self, s: pd.Series) -> tuple[float, float]:
        if self.method == "percentile":
            return float(s.quantile(self.lower_quantile)), float(
                s.quantile(self.upper_quantile)
            )
        # Default: Tukey IQR fences.
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        return q1 - self.iqr_multiplier * iqr, q3 + self.iqr_multiplier * iqr

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path = "artifacts/outliers/outlier_handler.joblib") -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        self.logger.info("Saved fitted outlier handler to %s", path)
        return path

    @staticmethod
    def load(
        path: str | Path = "artifacts/outliers/outlier_handler.joblib",
    ) -> "OutlierHandler":
        return joblib.load(Path(path))

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("OutlierHandler must be fit() before transform().")
