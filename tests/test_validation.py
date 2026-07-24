"""Tests for Phase A — DatasetValidator."""

import pandas as pd
import pytest

from src.preprocessing.validation import DatasetValidator, DataValidationError


def test_valid_dataset_passes(raw_df):
    report = DatasetValidator().validate(raw_df)
    assert report.passed
    assert report.n_rows == 10
    assert report.n_columns == 15


def test_missing_column_is_critical(raw_df):
    report = DatasetValidator().validate(raw_df.drop(columns=["goal"]))
    assert not report.passed
    assert any(c.name == "required_columns" and c.status == "CRITICAL" for c in report.checks)


def test_empty_dataset_is_critical():
    report = DatasetValidator().validate(pd.DataFrame())
    assert not report.passed


def test_duplicate_ids_error(raw_df):
    dup = raw_df.copy()
    dup.loc[1, "ID"] = 1  # make two rows share ID=1
    report = DatasetValidator().validate(dup)
    assert not report.passed
    assert any(c.name == "id_uniqueness" and c.status == "ERROR" for c in report.checks)


def test_invalid_state_error(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "state"] = "not_a_state"
    report = DatasetValidator().validate(bad)
    assert not report.passed


def test_assert_valid_raises(raw_df):
    with pytest.raises(DataValidationError):
        DatasetValidator().assert_valid(raw_df.drop(columns=["state"]))
