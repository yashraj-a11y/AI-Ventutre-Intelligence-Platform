"""Phase J — Multi-Modal Fusion (Day-3 Stage-2, Week 5).

Combines the three feature blocks — **text** (TF-IDF, Phase H), **categorical**
(one-hot + frequency, Phase B) and **numeric** (engineered + log/cap/scaled,
Phases C–F) — into a single sparse matrix the model consumes:

    X_fused = scipy.sparse.hstack([X_text, X_categorical, X_numeric])

"Multi-modal" here is honest, not hype: it fuses *weak* title text with a
*tabular* block, and the tabular side (goal, duration, category, timing) carries
most of the signal — the titles are ~6 words and thin. Fusion's value is giving
the model both modalities in one aligned matrix.

Two things make this correct: (1) every block has the **same rows in the same
order** (all derive from the Phase-0 split), so ``hstack`` aligns by position;
(2) the dense numeric block is **promoted to sparse** first, because
``scipy.sparse.hstack`` needs sparse inputs and NumPy can't mix dense + sparse.

:class:`FeatureFusion` records a ``block_slices`` map (which column range belongs
to which block) so a fused feature index is always traceable back to its origin,
and persists the matrix as ``.npz``.

Example
-------
>>> fused = FeatureFusion().fuse({
...     "text":        (X_tfidf,  tfidf_names),
...     "categorical": (X_cat,    cat_names),
...     "numeric":     (X_num_df, num_names),
... })
>>> fused.X.shape
(265171, 866)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils.config import load_config
from src.utils.logger import get_logger

Block = Union[np.ndarray, pd.DataFrame, sp.spmatrix]


@dataclass
class FusedMatrix:
    """A fused sparse feature matrix plus the provenance to interpret it."""

    X: sp.csr_matrix
    feature_names: list[str] = field(default_factory=list)
    block_slices: dict[str, tuple[int, int]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "shape": list(self.X.shape),
            "nnz": int(self.X.nnz),
            "density": round(self.X.nnz / (self.X.shape[0] * self.X.shape[1]), 6),
            "n_features": len(self.feature_names),
            "blocks": {k: list(v) for k, v in self.block_slices.items()},
        }

    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path, self.X)
        return path


class FeatureFusion:
    """Horizontally stack the text/categorical/numeric blocks into one CSR matrix.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.features.fusion")

        fc = self.config.get("day3", {}).get("fusion", {})
        # Order matters only for column layout / block_slices, not correctness.
        self.order: list[str] = list(fc.get("order", ["text", "categorical", "numeric"]))

    # ------------------------------------------------------------------ public
    def fuse(self, blocks: dict[str, tuple[Block, list[str]]]) -> FusedMatrix:
        """Fuse ``{block_name: (matrix, feature_names)}`` into one CSR matrix."""
        ordered = [name for name in self.order if name in blocks]
        # Include any blocks not mentioned in `order` (defensive), appended last.
        ordered += [name for name in blocks if name not in ordered]

        self._validate_rows(blocks, ordered)

        mats: list[sp.csr_matrix] = []
        names: list[str] = []
        slices: dict[str, tuple[int, int]] = {}
        col = 0
        for name in ordered:
            block, block_names = blocks[name]
            csr = self._to_sparse(block)
            mats.append(csr)
            start, end = col, col + csr.shape[1]
            slices[name] = (start, end)
            names.extend(self._names(block_names, csr.shape[1], name))
            col = end

        X = sp.hstack(mats, format="csr")
        result = FusedMatrix(X=X, feature_names=names, block_slices=slices)

        self.logger.info(
            "Fused %d blocks -> %s, nnz %d, density %.3f%%.",
            len(mats),
            X.shape,
            X.nnz,
            100 * X.nnz / (X.shape[0] * X.shape[1]),
        )
        for name, (s, e) in slices.items():
            self.logger.info("  block '%s': columns [%d:%d] (%d)", name, s, e, e - s)
        return result

    def save_feature_names(
        self, names: list[str], path: str | Path = "artifacts/fused_features/feature_names.json"
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(names, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load_npz(path: str | Path) -> sp.csr_matrix:
        return sp.load_npz(Path(path)).tocsr()

    # --------------------------------------------------------------- internals
    def _validate_rows(self, blocks, ordered) -> None:
        n_rows = {name: blocks[name][0].shape[0] for name in ordered}
        distinct = set(n_rows.values())
        if len(distinct) > 1:
            raise ValueError(f"Blocks have mismatched row counts: {n_rows}")

    @staticmethod
    def _to_sparse(block: Block) -> sp.csr_matrix:
        if sp.issparse(block):
            return block.tocsr()
        arr = block.to_numpy() if isinstance(block, pd.DataFrame) else np.asarray(block)
        return sp.csr_matrix(arr.astype(np.float64))

    @staticmethod
    def _names(block_names: list[str], n_cols: int, block: str) -> list[str]:
        if block_names and len(block_names) == n_cols:
            return list(block_names)
        # Fallback: synthesize names if none/mismatched were supplied.
        return [f"{block}_{i}" for i in range(n_cols)]
