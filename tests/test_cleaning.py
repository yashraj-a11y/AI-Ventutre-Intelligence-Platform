"""Tests for Phase B — cleaning handlers."""

from src.preprocessing.cleaning import (
    CategoryNormalizer,
    ColumnCleaner,
    ColumnRemover,
    TextCleaner,
    normalize_column_name,
)


def test_normalize_column_name():
    assert normalize_column_name("usd pledged") == "usd_pledged"
    assert normalize_column_name(" ID ") == "id"


def test_column_cleaner_renames(raw_df):
    out = ColumnCleaner().fit_transform(raw_df)
    assert "usd_pledged" in out.columns
    assert "id" in out.columns
    assert "usd pledged" not in out.columns


def test_text_cleaner_preserves_raw_and_strips_marker(raw_df):
    out = TextCleaner().fit_transform(raw_df)
    assert "name_clean" in out.columns
    assert "name" in out.columns  # raw preserved
    # row 8 title "Cool Thing (Canceled)" -> marker removed, lowercased
    cleaned = out.loc[7, "name_clean"]
    assert "canceled" not in cleaned
    assert cleaned == "cool thing"
    assert out.loc[7, "name"] == "Cool Thing (Canceled)"  # unchanged


def test_column_remover_drops_leakage_keeps_goal(raw_df):
    df = ColumnCleaner().fit_transform(raw_df)
    out = ColumnRemover().fit_transform(df)
    for leaked in ["id", "pledged", "backers", "usd_pledged", "usd_pledged_real"]:
        assert leaked not in out.columns
    assert "usd_goal_real" in out.columns  # kept: known at launch


def test_category_normalizer_strips_whitespace(raw_df):
    df = raw_df.copy()
    df.loc[0, "country"] = "  US  "
    out = CategoryNormalizer().fit_transform(df)
    assert out.loc[0, "country"] == "US"
