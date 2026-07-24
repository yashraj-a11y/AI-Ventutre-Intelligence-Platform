"""Tests for Phase E — DataTypeHandler."""

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from src.preprocessing.cleaning import ColumnCleaner, ColumnRemover
from src.preprocessing.datatype import DataTypeHandler


def _prep(raw_df):
    df = ColumnCleaner().fit_transform(raw_df)
    return ColumnRemover().fit_transform(df)


def test_dates_become_datetime(raw_df):
    out = DataTypeHandler().fit_transform(_prep(raw_df))
    assert is_datetime64_any_dtype(out["launched"])
    assert is_datetime64_any_dtype(out["deadline"])
    # date math now works
    assert (out["deadline"] - out["launched"]).dt.days.iloc[0] == 31


def test_categoricals_become_category(raw_df):
    out = DataTypeHandler().fit_transform(_prep(raw_df))
    assert str(out["country"].dtype) == "category"
    assert str(out["main_category"].dtype) == "category"


def test_bad_date_coerced_to_nat(raw_df):
    df = _prep(raw_df)
    df.loc[0, "launched"] = "not-a-date"
    out = DataTypeHandler().fit_transform(df)
    assert pd.isna(out.loc[0, "launched"])   # coerced, not crashed
