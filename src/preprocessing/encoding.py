"""Phase B — Categorical Encoding (Day-3 Stage-2).

:class:`EncodingHandler` turns the four categorical columns into a numeric,
**sparse** block ready for multi-modal fusion. The encoding policy is
cardinality-driven (decided in Phase A):

- **One-hot** the low-cardinality columns ``main_category`` (15), ``currency``
  (14), ``country`` (22) — each level becomes its own 0/1 indicator.
- **Frequency-encode** the high-cardinality column ``category`` (159 levels):
  each level maps to the *fraction of training rows* it covers, collapsing 159
  levels into a single numeric column instead of 159 sparse ones.

All categorical→numeric logic lives here (Option A): one-hot and frequency
share the same fit-on-train / handle-unseen machinery, so there is one home for
"how categories become numbers".

**Fit-on-train discipline.** ``fit`` learns the one-hot vocabulary and the
frequency maps from the training set only. At ``transform`` time a category not
seen in training is handled gracefully: one-hot emits an all-zero row for that
column (``handle_unknown="ignore"``); frequency emits 0.0 (an unseen level had
zero training mass). No test statistic ever enters a fitted parameter.

Output is a :class:`scipy.sparse.csr_matrix` — a dense one-hot block for 331k
rows would be almost entirely zeros and waste memory; sparse stores only the
non-zeros.

Example
-------
>>> enc = EncodingHandler().fit(X_train)
>>> X_train_enc = enc.transform(X_train)      # scipy.sparse.csr_matrix
>>> enc.get_feature_names_out()[:3]
['main_category_Art', 'main_category_Comics', 'main_category_Crafts']
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

from src.utils.config import load_config
from src.utils.logger import get_logger


class EncodingHandler:
    """Fit/transform categorical columns into a sparse numeric block.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.preprocessing.encoding")

        enc_cfg = self.config.get("day3", {}).get("encoding", {})
        self.onehot_cols: list[str] = list(enc_cfg.get("low_cardinality_onehot", []))
        self.freq_cols: list[str] = list(enc_cfg.get("frequency_encode", []))

        # Fitted state (populated by ``fit``).
        self._ohe: Optional[OneHotEncoder] = None
        self._freq_maps: dict[str, dict[Any, float]] = {}
        self._feature_names: list[str] = []
        self._fitted = False

    # -------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y: Any = None) -> "EncodingHandler":
        """Learn the one-hot vocabulary and frequency maps from **train** only."""
        onehot_present = [c for c in self.onehot_cols if c in X.columns]
        freq_present = [c for c in self.freq_cols if c in X.columns]

        # One-hot: unseen categories at transform time -> all-zero (ignore).
        if onehot_present:
            self._ohe = OneHotEncoder(
                handle_unknown="ignore", sparse_output=True, dtype=np.float64
            )
            self._ohe.fit(X[onehot_present].astype("string").fillna("__missing__"))

        # Frequency: level -> fraction of training rows (normalize=True).
        self._freq_maps = {}
        for c in freq_present:
            self._freq_maps[c] = (
                X[c].value_counts(normalize=True, dropna=True).to_dict()
            )

        self._onehot_present = onehot_present
        self._freq_present = freq_present
        self._feature_names = self._build_feature_names()
        self._fitted = True
        self.logger.info(
            "Encoder fitted — one-hot %s, frequency %s -> %d output features.",
            onehot_present,
            freq_present,
            len(self._feature_names),
        )
        return self

    # -------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> sp.csr_matrix:
        """Encode ``X`` into a sparse block using the fitted state."""
        self._check_fitted()
        blocks: list[sp.spmatrix] = []

        if self._onehot_present:
            ohe_block = self._ohe.transform(
                X[self._onehot_present].astype("string").fillna("__missing__")
            )
            blocks.append(ohe_block)

        for c in self._freq_present:
            mapped = X[c].map(self._freq_maps[c]).fillna(0.0).to_numpy(dtype=np.float64)
            blocks.append(sp.csr_matrix(mapped.reshape(-1, 1)))

        if not blocks:
            return sp.csr_matrix((len(X), 0))
        return sp.hstack(blocks, format="csr")

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> sp.csr_matrix:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------ names
    def get_feature_names_out(self) -> list[str]:
        self._check_fitted()
        return list(self._feature_names)

    def _build_feature_names(self) -> list[str]:
        names: list[str] = []
        if self._onehot_present:
            names.extend(self._ohe.get_feature_names_out(self._onehot_present).tolist())
        names.extend(f"{c}_freq" for c in self._freq_present)
        return names

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path = "artifacts/encoders/encoder.joblib") -> Path:
        """Persist the fitted encoder (part of the model — must travel with it)."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        self.logger.info("Saved fitted encoder to %s", path)
        return path

    @staticmethod
    def load(path: str | Path = "artifacts/encoders/encoder.joblib") -> "EncodingHandler":
        return joblib.load(Path(path))

    # --------------------------------------------------------------- internals
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("EncodingHandler must be fit() before transform().")
