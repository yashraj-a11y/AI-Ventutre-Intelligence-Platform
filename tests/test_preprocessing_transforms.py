"""Day 17 · Phase B — unit tests for Day-3 preprocessing transforms.

Fast, fixture-based (no real data / models / index) — CI-safe. Each test targets one behaviour:
expected output, invalid input, an edge case, or an exception. Covers the encoding, scaling,
log-transform, and outlier handlers that the original suite left untested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.preprocessing.encoding import EncodingHandler
from src.preprocessing.outliers import OutlierHandler
from src.preprocessing.scaling import ScalingHandler
from src.preprocessing.transformations import TransformationHandler


def _num_df() -> pd.DataFrame:
    return pd.DataFrame({"goal": [100., 200., 300., 400., 500.],
                         "usd_goal_real": [1e3, 2e3, 3e3, 4e3, 5e3]})


# ----------------------------------------------------------------- EncodingHandler
def test_encoding_fit_transform_shape(raw_df, base_config):
    enc = EncodingHandler(base_config).fit(raw_df)
    X = enc.transform(raw_df)
    assert sp.issparse(X)
    assert X.shape[0] == len(raw_df)
    assert X.shape[1] == len(enc.get_feature_names_out()) > 0


def test_encoding_transform_before_fit_raises(raw_df, base_config):
    with pytest.raises(Exception):
        EncodingHandler(base_config).transform(raw_df)


def test_encoding_unseen_category_does_not_crash(raw_df, base_config):
    enc = EncodingHandler(base_config).fit(raw_df)
    novel = raw_df.copy()
    novel["country"] = "ZZ"                       # a country unseen at fit time
    X = enc.transform(novel)                       # handle_unknown -> ignored, no crash
    assert X.shape[0] == len(novel)


# ----------------------------------------------------------------- ScalingHandler
def test_scaling_standardizes_to_zero_mean(base_config):
    out = ScalingHandler(base_config).fit_transform(_num_df())
    assert abs(float(out["usd_goal_real"].mean())) < 1e-9   # StandardScaler centers
    assert out.shape == _num_df().shape


def test_scaling_missing_columns_ok(base_config):
    out = ScalingHandler(base_config).fit_transform(pd.DataFrame({"goal": [1., 2., 3.]}))
    assert len(out) == 3                            # absent configured columns skipped, not crash


def test_scaling_disabled_is_passthrough(base_config):
    base_config["day3"]["scaling"]["enabled"] = False
    df = _num_df()
    pd.testing.assert_frame_equal(ScalingHandler(base_config).fit_transform(df), df)


# ----------------------------------------------------------------- TransformationHandler
def test_transformation_applies_log1p(base_config):
    base_config["day3"]["transform"]["method"] = "log1p"
    df = pd.DataFrame({"goal": [0., 9., 99.], "usd_goal_real": [1., 3., 7.]})
    out = TransformationHandler(base_config).fit_transform(df)
    assert np.allclose(out["goal"].to_numpy(), np.log1p([0., 9., 99.]))   # log1p(0)=0


def test_transformation_is_rank_preserving(base_config):
    df = pd.DataFrame({"goal": [5., 1., 100., 20.], "usd_goal_real": [1., 1., 1., 1.]})
    out = TransformationHandler(base_config).fit_transform(df)
    assert list(np.argsort(out["goal"].to_numpy())) == list(np.argsort(df["goal"].to_numpy()))


# ----------------------------------------------------------------- OutlierHandler
def test_outlier_caps_extreme_value(base_config):
    df = pd.DataFrame({"goal": [1., 2., 3., 4., 5., 6., 7., 8., 9., 1000.]})
    out = OutlierHandler(base_config).fit_transform(df)
    assert float(out["goal"].max()) < 1000.0       # extreme clipped down
    assert float(out["goal"].max()) >= 9.0          # ...but not below the normal range


def test_outlier_leaves_normal_data_unchanged(base_config):
    df = pd.DataFrame({"goal": [10., 11., 12., 13., 14.]})
    out = OutlierHandler(base_config).fit_transform(df)
    assert np.allclose(out["goal"].to_numpy(), df["goal"].to_numpy())
