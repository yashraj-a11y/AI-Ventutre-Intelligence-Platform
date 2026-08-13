"""Day 17 · Phase B — unit tests for Day-3 feature-engineering modules.

Fast, fixture-based — CI-safe. Covers the feature builder, text preprocessor, TF-IDF vectorizer, and
feature selector. (TF-IDF asserts the REAL cap of 800 features, not the draft's fabricated 2,500.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.features.feature_builder import FeatureBuilder
from src.features.feature_selection import FeatureSelector
from src.features.text_preprocessing import TextPreprocessor
from src.features.tfidf_vectorizer import TfidfVectorizerHandler


# ----------------------------------------------------------------- FeatureBuilder
def test_feature_builder_is_additive_and_row_preserving(raw_df, base_config):
    out = FeatureBuilder(base_config).fit_transform(raw_df)
    assert len(out) == len(raw_df)                          # row-preserving
    assert set(raw_df.columns).issubset(out.columns)        # additive (originals kept)
    for c in ["campaign_duration", "goal_per_day", "title_length", "word_count", "launch_year"]:
        assert c in out.columns


def test_feature_builder_handles_nan_title(raw_df, base_config):
    out = FeatureBuilder(base_config).fit_transform(raw_df)   # r9 has name=NaN
    assert out["title_length"].notna().all()                 # no NaN leaks into title stats


# ----------------------------------------------------------------- TextPreprocessor
def test_text_lowercases_and_removes_stopwords(base_config):
    tp = TextPreprocessor(base_config).fit()
    out = tp.process_text("The Amazing Widget Machine")
    assert out == out.lower()
    assert "the" not in out.split()                          # 'the' is a stopword
    assert "widget" in out.split()


def test_text_empty_and_nan_return_empty(base_config):
    tp = TextPreprocessor(base_config).fit()
    assert tp.process_text("") == ""
    assert tp.process_text(np.nan) == ""


def test_text_transform_adds_output_column(base_config):
    tp = TextPreprocessor(base_config).fit()
    out = tp.transform(pd.DataFrame({"name_clean": ["Cool Project", "Another Idea"]}))
    assert "name_processed" in out.columns
    assert len(out) == 2


# ----------------------------------------------------------------- TfidfVectorizerHandler
def _text_df() -> pd.DataFrame:
    return pd.DataFrame({"name_processed": ["cool project game", "project game fun", "game fun cool"]})


def test_tfidf_fit_transform_respects_max_features(base_config):
    tf = TfidfVectorizerHandler(base_config).fit(_text_df())
    X = tf.transform(_text_df())
    assert sp.issparse(X)
    assert X.shape[0] == 3
    assert 0 < X.shape[1] <= 800                            # real cap is 800, not 2500
    assert X.shape[1] == len(tf.get_feature_names_out())


def test_tfidf_transform_before_fit_raises(base_config):
    with pytest.raises(Exception):
        TfidfVectorizerHandler(base_config).transform(_text_df())


# ----------------------------------------------------------------- FeatureSelector
def test_feature_selector_drops_zero_variance(base_config):
    # col a varies, col b is all-zero, col c is constant -> b & c have zero variance
    X = sp.csr_matrix(np.array([[1., 0., 5.], [2., 0., 5.], [3., 0., 5.], [4., 0., 5.]]))
    y = np.array([0, 1, 0, 1])
    fs = FeatureSelector(base_config).fit(X, y, ["a", "b", "c"])
    Xs = fs.transform(X)
    assert fs.result_.n_input == 3
    assert Xs.shape[1] == fs.result_.n_selected <= 3
    assert "b" in fs.result_.dropped_variance               # zero-variance column removed
