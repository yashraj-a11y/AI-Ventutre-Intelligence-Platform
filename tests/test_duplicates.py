"""Tests for Phase D — duplicate handlers."""

import pandas as pd

from src.preprocessing.duplicates import (
    DuplicateDetector,
    DuplicateRemover,
    DuplicateReporter,
)


def test_detect_finds_injected_duplicates(raw_df):
    demo = pd.concat([raw_df, raw_df.head(2)], ignore_index=True)  # +2 dup rows
    stats = DuplicateDetector().detect(demo)
    assert stats["duplicate_rows"] == 2
    assert stats["duplicate_ids"] == 2


def test_remover_keeps_first(raw_df):
    demo = pd.concat([raw_df, raw_df.head(2)], ignore_index=True)
    remover = DuplicateRemover()
    out = remover.transform(demo)
    assert len(out) == len(raw_df)      # duplicates gone
    assert remover.n_removed_ == 2


def test_detector_graceful_without_id(raw_df):
    # after the id column is gone, id-check returns None (no crash)
    stats = DuplicateDetector().detect(raw_df.drop(columns=["ID"]))
    assert stats["duplicate_ids"] is None


def test_reporter_includes_removed(raw_df):
    rep = DuplicateReporter().report(raw_df, rows_removed=3)
    assert rep["rows_removed"] == 3
