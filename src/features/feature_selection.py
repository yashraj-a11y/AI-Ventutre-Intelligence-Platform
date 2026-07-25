"""Phase K — Feature Selection (Day-3 Stage-2).

Trims the fused 866-column matrix using three complementary criteria, all fit on
**train**:

1. **Variance** — drop (near-)constant columns (``VarianceThreshold``). A column
   that never varies carries no information and only adds noise/compute.
2. **Redundancy** — drop one of a highly-correlated pair. Phase A found
   ``goal`` ↔ ``usd_goal_real`` at Pearson 0.957; keeping both gives a linear
   model two copies of one signal (multicollinearity). We keep whichever has the
   higher mutual information with the target and drop the other.
3. **Relevance** — rank the survivors by **mutual information** with the target
   (optionally keep only the top-``k``). MI captures any dependency, linear or
   not, and unlike correlation works for the 0/1 one-hot and sparse TF-IDF
   columns too.

The selector learns a boolean **support mask** over the fused columns on train
and applies it to any partition — so train and test stay column-aligned, and the
test set never influences which features are kept (that would leak).

Example
-------
>>> sel = FeatureSelector().fit(X_train_fused, y_train, feature_names)
>>> X_train_sel = sel.transform(X_train_fused)
>>> sel.selected_names_[:3]
['app', 'art', 'book']
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from src.utils.config import load_config
from src.utils.logger import get_logger


@dataclass
class SelectionResult:
    """What survived selection and why each column was dropped."""

    n_input: int
    n_selected: int
    dropped_variance: list[str] = field(default_factory=list)
    dropped_correlated: list[str] = field(default_factory=list)
    dropped_low_mi: list[str] = field(default_factory=list)
    top_features: list[tuple[str, float]] = field(default_factory=list)


class FeatureSelector:
    """Variance + redundancy + mutual-information selection over the fused matrix.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.features.feature_selection")

        s = self.config.get("day3", {}).get("selection", {})
        self.variance_threshold: float = float(s.get("variance_threshold", 0.0))
        self.correlated_pairs: list[list[str]] = list(
            s.get("correlated_pairs", [["goal", "usd_goal_real"]])
        )
        self.corr_threshold: float = float(s.get("corr_threshold", 0.9))
        self.top_k: Optional[int] = s.get("top_k", None)
        self.mi_subsample: int = int(s.get("mi_subsample", 40000))
        self.random_state: int = int(s.get("random_state", 42))

        self.support_: Optional[np.ndarray] = None  # boolean mask over input cols
        self.selected_names_: list[str] = []
        self.result_: Optional[SelectionResult] = None
        self._fitted = False

    # -------------------------------------------------------------------- fit
    def fit(self, X: sp.spmatrix, y, feature_names: list[str]) -> "FeatureSelector":
        """Learn the support mask on **train** (X fused, y target, names aligned)."""
        X = sp.csr_matrix(X)
        n = X.shape[1]
        names = list(feature_names)
        keep = np.ones(n, dtype=bool)

        # 1. Variance ------------------------------------------------------
        vt = VarianceThreshold(threshold=self.variance_threshold).fit(X)
        var_mask = vt.get_support()
        dropped_var = [names[i] for i in range(n) if not var_mask[i]]
        keep &= var_mask

        # 2. Redundancy: drop the lower-MI member of each correlated pair --
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        dropped_corr: list[str] = []
        for pair in self.correlated_pairs:
            idxs = [name_to_idx[p] for p in pair if p in name_to_idx]
            if len(idxs) < 2 or not all(keep[i] for i in idxs):
                continue
            r = self._pearson(X, idxs[0], idxs[1])
            if abs(r) >= self.corr_threshold:
                mi = self._mi(X, y, idxs)
                drop_idx = idxs[int(np.argmin(mi))]
                keep[drop_idx] = False
                dropped_corr.append(names[drop_idx])
                self.logger.info(
                    "Correlated pair %s (r=%.3f): dropping '%s' (lower MI).",
                    pair, r, names[drop_idx],
                )

        # 3. Relevance: rank survivors by MI; optionally keep top-k --------
        surviving = np.where(keep)[0]
        mi_scores = self._mi(X, y, surviving.tolist())
        ranked = sorted(zip(surviving, mi_scores), key=lambda t: t[1], reverse=True)
        dropped_low_mi: list[str] = []
        if self.top_k is not None and self.top_k < len(ranked):
            for idx, _ in ranked[self.top_k:]:
                keep[idx] = False
                dropped_low_mi.append(names[idx])

        self.support_ = keep
        self.selected_names_ = [names[i] for i in range(n) if keep[i]]
        self.result_ = SelectionResult(
            n_input=n,
            n_selected=int(keep.sum()),
            dropped_variance=dropped_var,
            dropped_correlated=dropped_corr,
            dropped_low_mi=dropped_low_mi,
            top_features=[(names[i], float(s)) for i, s in ranked[:15]],
        )
        self._fitted = True
        self.logger.info(
            "Selection: %d -> %d features (variance -%d, correlated -%d, low-MI -%d).",
            n, int(keep.sum()), len(dropped_var), len(dropped_corr), len(dropped_low_mi),
        )
        return self

    # -------------------------------------------------------------- transform
    def transform(self, X: sp.spmatrix) -> sp.csr_matrix:
        self._check_fitted()
        return sp.csr_matrix(X)[:, self.support_]

    def fit_transform(self, X, y, feature_names) -> sp.csr_matrix:
        return self.fit(X, y, feature_names).transform(X)

    # --------------------------------------------------------------- internals
    def _pearson(self, X: sp.csr_matrix, i: int, j: int) -> float:
        a = np.asarray(X[:, i].todense()).ravel()
        b = np.asarray(X[:, j].todense()).ravel()
        return float(np.corrcoef(a, b)[0, 1])

    def _mi(self, X: sp.csr_matrix, y, cols: list[int]) -> np.ndarray:
        """Mutual information of given columns with y, on a subsample for speed."""
        y = np.asarray(y)
        n = X.shape[0]
        if self.mi_subsample and n > self.mi_subsample:
            rng = np.random.default_rng(self.random_state)
            rows = rng.choice(n, self.mi_subsample, replace=False)
        else:
            rows = np.arange(n)
        Xs = X[rows][:, cols]
        ys = y[rows]
        return mutual_info_classif(
            Xs, ys, discrete_features="auto", random_state=self.random_state
        )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("FeatureSelector must be fit() before transform().")
