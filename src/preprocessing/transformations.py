"""Phase C — Numerical Transformation (Day-3 Stage-2).

Phase A found the monetary numerics are near log-normal: ``goal`` has skewness
73 and kurtosis ~5900. A logistic model fed that raw sees a variable where
99.9% of rows are crushed near zero and a handful of 10^8 outliers dominate the
gradient. :class:`TransformationHandler` applies ``log1p`` to the skewed
numerics, pulling skew to ~0 and giving each feature a usable, roughly
symmetric spread.

Columns transformed (in place, row-preserving): ``goal``, ``usd_goal_real`` and
— when present, i.e. after Phase F has built it — ``goal_per_day``. All
log-transform logic lives here (``goal_log`` was removed from Phase F to avoid a
duplicated responsibility).

**log1p vs the fit/transform contract.** ``log1p(x) = log(1 + x)`` is a *fixed*
function — it learns nothing from data, so applying it to the full set could not
leak. We still expose ``fit``/``transform`` for interface consistency and
because the optional ``yeo-johnson`` method *does* learn a per-column power
parameter (lambda), which must be estimated on **train only**. That is the real
reason the method sits behind a fitted handler.

Example
-------
>>> tr = TransformationHandler().fit(X_train)
>>> Xt = tr.transform(X_train)          # goal/usd_goal_real now log1p-scaled
>>> tr.metadata_["goal"]["skew_after"]
-0.12...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger


class TransformationHandler:
    """Apply skew-reducing transforms to numeric columns, fit on train.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.preprocessing.transformations")

        tr_cfg = self.config.get("day3", {}).get("transform", {})
        self.method: str = tr_cfg.get("method", "log1p")  # log1p | yeo-johnson
        self.columns: list[str] = list(
            tr_cfg.get("columns", ["goal", "usd_goal_real", "goal_per_day"])
        )

        # Fitted state.
        self._cols_present: list[str] = []
        self._power_transformers: dict[str, Any] = {}  # yeo-johnson only
        self.metadata_: dict[str, dict[str, Any]] = {}
        self._fitted = False

    # -------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y: Any = None) -> "TransformationHandler":
        """Record before/after skew per column; learn lambda if yeo-johnson."""
        self._cols_present = [c for c in self.columns if c in X.columns]
        self.metadata_ = {}
        self._power_transformers = {}

        for c in self._cols_present:
            s = pd.to_numeric(X[c], errors="coerce").astype(float)
            self._warn_if_negative(c, s)
            before = float(s.skew())

            if self.method == "yeo-johnson":
                from sklearn.preprocessing import PowerTransformer

                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                transformed = pt.fit_transform(s.to_numpy().reshape(-1, 1)).ravel()
                self._power_transformers[c] = pt
                lam = float(pt.lambdas_[0])
            else:  # log1p (default): fixed, parameter-free
                transformed = np.log1p(s.clip(lower=0).to_numpy())
                lam = None

            self.metadata_[c] = {
                "method": self.method,
                "skew_before": before,
                "skew_after": float(pd.Series(transformed).skew()),
                "lambda": lam,
            }

        self._fitted = True
        self.logger.info(
            "Transform fitted (%s) on %s.", self.method, self._cols_present
        )
        for c, m in self.metadata_.items():
            self.logger.info(
                "  %s: skew %.2f -> %.2f", c, m["skew_before"], m["skew_after"]
            )
        return self

    # -------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with the configured numerics transformed in place."""
        self._check_fitted()
        out = X.copy()
        for c in self._cols_present:
            s = pd.to_numeric(out[c], errors="coerce").astype(float)
            if self.method == "yeo-johnson":
                out[c] = self._power_transformers[c].transform(
                    s.to_numpy().reshape(-1, 1)
                ).ravel()
            else:
                out[c] = np.log1p(s.clip(lower=0).to_numpy())
        return out

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # -------------------------------------------------------------- persistence
    def save(
        self, path: str | Path = "artifacts/transformers/transformer.joblib"
    ) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        self.logger.info("Saved fitted transformer to %s", path)
        return path

    @staticmethod
    def load(
        path: str | Path = "artifacts/transformers/transformer.joblib",
    ) -> "TransformationHandler":
        return joblib.load(Path(path))

    # --------------------------------------------------------------- internals
    def _warn_if_negative(self, col: str, s: pd.Series) -> None:
        n_neg = int((s < 0).sum())
        if n_neg and self.method == "log1p":
            self.logger.warning(
                "%s has %d negative value(s); clipping to 0 before log1p.", col, n_neg
            )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TransformationHandler must be fit() before transform().")
