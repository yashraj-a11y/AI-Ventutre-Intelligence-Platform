"""Tests for Phase C — MissingValueHandler (incl. the fit/transform discipline)."""

import numpy as np
import pandas as pd

from src.preprocessing.missing import MissingValueHandler


def _df(goal, name):
    return pd.DataFrame({"goal": goal, "name": name})


def test_fit_learns_median_transform_fills():
    mvh = MissingValueHandler(numeric_strategy="median")
    df = _df([1.0, 2.0, 3.0, np.nan], ["a", "b", "c", np.nan])
    out = mvh.fit_transform(df)
    assert mvh.fill_values_["goal"] == 2.0           # median of 1,2,3
    assert out["goal"].isna().sum() == 0
    assert out.loc[3, "name"] == ""                  # text filled with ""


def test_fit_on_train_transform_on_test_uses_train_stat():
    mvh = MissingValueHandler(numeric_strategy="median")
    train = _df([10.0, 20.0, 30.0], ["x", "y", "z"])
    test = _df([np.nan], [np.nan])
    mvh.fit(train)                                    # learns median 20 on TRAIN
    out = mvh.transform(test)
    assert out.loc[0, "goal"] == 20.0                 # TRAIN statistic applied to TEST


def test_transform_before_fit_is_noop():
    mvh = MissingValueHandler()
    df = _df([np.nan], ["a"])
    out = mvh.transform(df)                            # no fit() called
    assert out["goal"].isna().sum() == 1              # unchanged (safe no-op)
