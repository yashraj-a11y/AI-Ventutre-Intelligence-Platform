"""Phase D — Duplicate Handling.

Detects, reports, and removes duplicate records. Three single-responsibility
classes:

- :class:`DuplicateDetector` — *find* duplicates (exact rows + duplicate IDs), no changes.
- :class:`DuplicateRemover`  — *remove* exact duplicate rows, keeping the first.
- :class:`DuplicateReporter` — *summarize* duplicates into a JSON-serializable report.

Important — this is the first **row-changing** handler. Removing rows means the
output has fewer rows than the input, which **violates the sklearn ``transform``
contract** (transform must preserve row count and cannot touch ``y``). Therefore
:class:`DuplicateRemover` belongs to **Stage 1** (the custom dataset-cleaning
orchestrator), NOT an ``sklearn.Pipeline``. It exposes ``fit_transform`` for a
uniform call signature, but it is not a true feature transformer.

Current dataset: 0 duplicate rows, 0 duplicate IDs — this handler is a safeguard
for future ingests. (ID uniqueness is also validated earlier in Phase A, before
``ColumnRemover`` drops the ``id`` column.)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.preprocessing.cleaning import normalize_column_name
from src.utils.logger import get_logger


def _find_id_column(df: pd.DataFrame) -> Optional[str]:
    """Return the identifier column (normalized name 'id') if present."""
    for col in df.columns:
        if normalize_column_name(col) == "id":
            return col
    return None


class DuplicateDetector:
    """Detect duplicate rows and duplicate identifiers (read-only)."""

    def __init__(self, logger=None) -> None:
        self.logger = logger or get_logger("avip.preprocessing.duplicates")

    def detect(self, X: pd.DataFrame) -> dict[str, object]:
        """Return duplicate statistics without modifying the data."""
        n_rows = len(X)
        dup_row_mask = X.duplicated(keep="first")
        n_dup_rows = int(dup_row_mask.sum())

        id_col = _find_id_column(X)
        if id_col is not None:
            n_dup_ids = int(X[id_col].duplicated(keep="first").sum())
        else:
            n_dup_ids = None  # id already dropped (e.g. after Phase B)

        stats = {
            "total_rows": n_rows,
            "duplicate_rows": n_dup_rows,
            "duplicate_row_pct": round(100 * n_dup_rows / n_rows, 4) if n_rows else 0.0,
            "duplicate_ids": n_dup_ids,
            "id_column_checked": id_col,
        }
        self.logger.info("DuplicateDetector: %s", stats)
        return stats

    def duplicate_row_mask(self, X: pd.DataFrame) -> pd.Series:
        """Boolean mask marking rows that are exact duplicates of an earlier row."""
        return X.duplicated(keep="first")


class DuplicateRemover:
    """Remove exact duplicate rows, keeping the first occurrence (row-changing)."""

    def __init__(self, logger=None) -> None:
        self.logger = logger or get_logger("avip.preprocessing.duplicates")
        self.n_removed_: int = 0

    def fit(self, X: pd.DataFrame, y=None) -> "DuplicateRemover":
        return self  # stateless; removal decided per-call

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        before = len(X)
        out = X.drop_duplicates(keep="first").reset_index(drop=True)
        self.n_removed_ = before - len(out)
        self.logger.info(
            "DuplicateRemover: removed %d exact duplicate row(s) (%d -> %d).",
            self.n_removed_,
            before,
            len(out),
        )
        return out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class DuplicateReporter:
    """Summarize duplicate findings into a JSON-serializable report."""

    def __init__(self, logger=None) -> None:
        self.logger = logger or get_logger("avip.preprocessing.duplicates")

    def report(
        self, X: pd.DataFrame, rows_removed: Optional[int] = None
    ) -> dict[str, object]:
        stats = DuplicateDetector(logger=self.logger).detect(X)
        stats["rows_removed"] = rows_removed
        return stats
