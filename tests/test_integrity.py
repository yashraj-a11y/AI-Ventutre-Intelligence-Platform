"""Tests for Phase F — DataIntegrityValidator."""

import pandas as pd

from src.preprocessing.integrity import DataIntegrityValidator


def _prep(raw_df):
    """Minimal typed frame integrity needs (datetimes)."""
    df = raw_df.copy()
    df["launched"] = pd.to_datetime(df["launched"], errors="coerce")
    df["deadline"] = pd.to_datetime(df["deadline"], errors="coerce")
    df["goal"] = df["goal"].astype(float)
    return df


def test_epoch_and_corrupt_country_dropped(base_config, tmp_path, raw_df):
    base_config["dataset"]["quarantine_path"] = str(tmp_path / "q.csv")
    iv = DataIntegrityValidator(config=base_config, mode="train")
    out = iv.transform(_prep(raw_df))
    rep = iv.report()
    assert rep["violations_per_rule"]["V5"] == 1     # r4 epoch 1970
    assert rep["violations_per_rule"]["V7"] == 1     # r5 'N,0"'
    assert rep["violations_per_rule"]["V1"] == 1     # r6 goal 0
    # dropped rows are not in the output
    assert "Epoch Bug" not in set(out["name"])
    assert "Corrupt Country" not in set(out["name"])


def test_sub_day_is_flagged_not_dropped(base_config, tmp_path, raw_df):
    base_config["dataset"]["quarantine_path"] = str(tmp_path / "q.csv")
    iv = DataIntegrityValidator(config=base_config, mode="train")
    out = iv.transform(_prep(raw_df))
    assert iv.report()["rows_flagged_kept"]["V2"] == 1
    assert "Same Day" in set(out["name"])            # kept despite sub-day duration


def test_quarantine_file_written(base_config, tmp_path, raw_df):
    qpath = tmp_path / "q.csv"
    base_config["dataset"]["quarantine_path"] = str(qpath)
    DataIntegrityValidator(config=base_config, mode="train").transform(_prep(raw_df))
    assert qpath.is_file()
    quarantined = pd.read_csv(qpath)
    assert "quarantine_reason" in quarantined.columns
    assert len(quarantined) >= 3                     # epoch + corrupt + zero-goal


def test_inference_mode_flags_but_drops_nothing(base_config, raw_df):
    iv = DataIntegrityValidator(config=base_config, mode="inference")
    out = iv.transform(_prep(raw_df))
    assert len(out) == len(raw_df)                   # nothing dropped
    assert "integrity_valid" in out.columns
    assert out["integrity_valid"].sum() < len(out)   # some rows flagged invalid
