"""Tests for Phase G — the full Stage-1 PreprocessingPipeline (integration)."""

import pandas as pd

from src.preprocessing.pipeline import PreprocessingPipeline, TargetBuilder


def test_target_builder_filters_and_encodes(raw_df):
    out = TargetBuilder().fit_transform(raw_df)
    assert "target" in out.columns
    assert set(out["target"].unique()) <= {0, 1}
    assert "canceled" not in set(out["state"])       # undecided removed


def _pipe(base_config, tmp_path):
    base_config["dataset"]["quarantine_path"] = str(tmp_path / "q.csv")
    return PreprocessingPipeline(config=base_config, mode="train")


def test_pipeline_output_is_leakage_free_and_targeted(base_config, tmp_path, raw_df):
    out = _pipe(base_config, tmp_path).fit_transform(raw_df)
    # leakage + identifier columns gone
    for leaked in ["id", "pledged", "backers", "usd_pledged", "usd_pledged_real"]:
        assert leaked not in out.columns
    # target present and binary
    assert "target" in out.columns
    assert set(out["target"].unique()) <= {0, 1}
    # kept columns
    assert {"name", "name_clean", "usd_goal_real"} <= set(out.columns)


def test_pipeline_drops_expected_rows(base_config, tmp_path, raw_df):
    # 10 raw -> drop r3 (canceled) -> 9 -> drop r4/r5/r6 (epoch/corrupt/zero-goal) -> 6
    out = _pipe(base_config, tmp_path).fit_transform(raw_df)
    assert len(out) == 6
    kept = set(out["name"])
    assert "Same Day" in kept                         # sub-day flagged, kept
    assert "Epoch Bug" not in kept


def test_pipeline_is_reproducible(base_config, tmp_path, raw_df):
    pipe = _pipe(base_config, tmp_path)
    first = pipe.fit_transform(raw_df)
    second = pipe.transform(raw_df)                   # reuse fitted state
    assert first.shape == second.shape
    assert list(first.columns) == list(second.columns)


def test_pipeline_report_populated(base_config, tmp_path, raw_df):
    pipe = _pipe(base_config, tmp_path)
    pipe.fit_transform(raw_df)
    report = pipe.get_report()
    assert "validation" in report
    assert "steps" in report
    assert report["steps"]["target_builder"]["rows_removed"] == 1  # the canceled row
