"""Phase I — Sparse vs Dense Analysis (Day-3 Stage-2, Week 4).

Day 3 produces three feature blocks with very different densities:

- **numeric** (Phase C/D/E/F): ~14 columns, essentially 100% dense.
- **categorical** (Phase B): 52 columns, ~8% dense (one-hot + one freq col).
- **text / TF-IDF** (Phase H): 800 columns, ~0.3% dense.

:class:`SparseDenseAnalyzer` measures each block's memory as **dense** vs
**sparse**, then reasons about *fusion*: the text block is 99.7% zeros, so
materialising the combined matrix densely is orders of magnitude larger than
storing it sparse. Crucially, ``numpy.concatenate`` / ``numpy.hstack`` cannot
combine a dense ``ndarray`` with a ``scipy.sparse`` matrix — they are different
storage types — which is *why* Phase J must use ``scipy.sparse.hstack`` (after
promoting the small dense numeric block to sparse).

This phase produces a report + a fusion-strategy note; it fits nothing.

Example
-------
>>> report = SparseDenseAnalyzer().analyze({"numeric": num, "categorical": cat, "text": tfidf})
>>> report.render_markdown("docs/day3_phaseI_sparse_analysis.md")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils.config import load_config
from src.utils.logger import get_logger

Block = Union[np.ndarray, pd.DataFrame, sp.spmatrix]

_FLOAT_BYTES = 8   # float64 value
_INDEX_BYTES = 4   # int32 index in CSR


@dataclass
class BlockStats:
    name: str
    n_rows: int
    n_cols: int
    nnz: int
    density: float
    dense_mb: float
    sparse_mb: float

    @property
    def savings(self) -> float:
        return self.dense_mb / self.sparse_mb if self.sparse_mb else float("inf")


@dataclass
class SparseReport:
    blocks: list[BlockStats] = field(default_factory=list)
    n_rows: int = 0
    total_cols: int = 0
    total_nnz: int = 0
    combined_density: float = 0.0
    combined_dense_mb: float = 0.0
    combined_sparse_mb: float = 0.0
    fusion_format: str = "csr"
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def combined_savings(self) -> float:
        return (
            self.combined_dense_mb / self.combined_sparse_mb
            if self.combined_sparse_mb
            else float("inf")
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["combined_savings"] = self.combined_savings
        for b, bd in zip(self.blocks, d["blocks"]):
            bd["savings"] = b.savings
        return d

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def render_markdown(self, path: str | Path) -> str:
        md = _render(self)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        return md


class SparseDenseAnalyzer:
    """Measure per-block density/memory and derive the fusion strategy."""

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.features.sparse_analysis")

    def analyze(self, blocks: dict[str, Block]) -> SparseReport:
        report = SparseReport()
        n_rows = None

        for name, block in blocks.items():
            stats = self._block_stats(name, block)
            report.blocks.append(stats)
            n_rows = stats.n_rows if n_rows is None else n_rows
            report.total_cols += stats.n_cols
            report.total_nnz += stats.nnz

        report.n_rows = n_rows or 0
        cells = report.n_rows * report.total_cols
        report.combined_density = (report.total_nnz / cells) if cells else 0.0
        report.combined_dense_mb = cells * _FLOAT_BYTES / 1e6
        report.combined_sparse_mb = (
            report.total_nnz * (_FLOAT_BYTES + _INDEX_BYTES)
            + (report.n_rows + 1) * _INDEX_BYTES
        ) / 1e6

        self.logger.info(
            "Sparse analysis — %d features, combined density %.3f%%, "
            "dense %.1f MB vs sparse %.1f MB (%.0fx).",
            report.total_cols,
            report.combined_density * 100,
            report.combined_dense_mb,
            report.combined_sparse_mb,
            report.combined_savings,
        )
        return report

    # --------------------------------------------------------------- internals
    def _block_stats(self, name: str, block: Block) -> BlockStats:
        if sp.issparse(block):
            n_rows, n_cols = block.shape
            nnz = int(block.nnz)
            sparse_mb = (
                block.data.nbytes + getattr(block, "indices", np.array([])).nbytes
                + getattr(block, "indptr", np.array([])).nbytes
            ) / 1e6
        else:
            arr = block.to_numpy() if isinstance(block, pd.DataFrame) else np.asarray(block)
            n_rows, n_cols = arr.shape
            nnz = int(np.count_nonzero(arr))
            # A dense block stored sparsely would cost nnz*(val+idx) — reported for contrast.
            sparse_mb = nnz * (_FLOAT_BYTES + _INDEX_BYTES) / 1e6

        dense_mb = n_rows * n_cols * _FLOAT_BYTES / 1e6
        density = nnz / (n_rows * n_cols) if n_rows * n_cols else 0.0
        return BlockStats(
            name=name,
            n_rows=n_rows,
            n_cols=n_cols,
            nnz=nnz,
            density=density,
            dense_mb=dense_mb,
            sparse_mb=sparse_mb,
        )


def _render(r: SparseReport) -> str:
    lines = ["# Day 3 — Phase I: Sparse vs Dense Analysis\n"]
    lines.append(f"_Generated {r.created_utc} · rows: {r.n_rows:,}_\n")

    lines.append("## Per-block memory\n")
    lines.append("| block | cols | density | dense | sparse | savings |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for b in r.blocks:
        lines.append(
            f"| {b.name} | {b.n_cols} | {b.density*100:.3f}% | "
            f"{b.dense_mb:.1f} MB | {b.sparse_mb:.1f} MB | {b.savings:.0f}× |"
        )
    lines.append(
        f"| **combined** | **{r.total_cols}** | **{r.combined_density*100:.3f}%** | "
        f"**{r.combined_dense_mb:.1f} MB** | **{r.combined_sparse_mb:.1f} MB** | "
        f"**{r.combined_savings:.0f}×** |"
    )

    lines.append("\n## Fusion strategy\n")
    lines.append(
        "- The **text** block is ~99.7% zeros; densifying the combined matrix "
        f"would cost **{r.combined_dense_mb:.0f} MB** vs **{r.combined_sparse_mb:.0f} MB** "
        f"sparse (**{r.combined_savings:.0f}×**).\n"
        "- `numpy.hstack` / `numpy.concatenate` **cannot** mix a dense `ndarray` "
        "with a `scipy.sparse` matrix — different storage types.\n"
        "- **Strategy (Phase J):** promote the small dense numeric block to sparse "
        "(`scipy.sparse.csr_matrix`), then `scipy.sparse.hstack([text, categorical, "
        "numeric])` → one CSR matrix.\n"
        "- Do **not** StandardScale the sparse blocks (centering fills the zeros); "
        "scaling stays on the dense numeric block only."
    )
    return "\n".join(lines) + "\n"
