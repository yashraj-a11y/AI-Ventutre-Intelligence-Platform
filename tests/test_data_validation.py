"""Day 17 · Phase C — data-validation framework (processed dataset + feature matrix + target).

The existing `test_validation.py` validates the RAW schema. This validates the *processed* artifacts a
model actually consumes — schema, target integrity, ranges, missing values, category cardinality, and
feature-matrix sanity. The validation LOGIC is tested fast on a tiny fixture (CI-safe); the REAL 50 MB
dataset + 865-wide feature matrix are validated in a `heavy` test that also emits the JSON report to
`artifacts/testing/data_validation.json`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

_REPO = Path(__file__).resolve().parents[1]
_REQUIRED = {"name_clean", "target", "main_category", "category", "usd_goal_real", "country", "currency"}


def build_data_validation_report(df: pd.DataFrame, X: Optional[sp.spmatrix] = None,
                                 y: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Run every processed-data check and return a metrics-as-data report (no exceptions).

    Severity matters: ``critical`` failures make the data invalid (a model must not train on them);
    ``warning`` failures are quality issues that downstream code already handles (e.g. NaN cleaned
    titles are ``fillna("")``-ed by TextPreprocessor), so they are surfaced but don't fail validation.
    ``passed`` == no *critical* failures.
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, severity: str = "critical", detail: str = "") -> None:
        checks.append({"name": name, "status": "PASS" if ok else "FAIL",
                       "severity": severity, "detail": detail})

    add("required_columns", _REQUIRED.issubset(df.columns), detail=f"missing={sorted(_REQUIRED - set(df.columns))}")
    if "target" in df:
        add("target_binary", set(pd.unique(df["target"].dropna())) <= {0, 1}, detail="target must be {0,1}")
        add("target_no_missing", bool(df["target"].notna().all()), detail="no NaN labels")
    if "usd_goal_real" in df:
        add("goal_positive", bool((df["usd_goal_real"] > 0).all()), detail="all goals > 0")
    if "name_clean" in df:                                    # downstream fillna("") -> warning, not critical
        n_nan = int(df["name_clean"].isna().sum())
        add("name_clean_present", n_nan == 0, severity="warning",
            detail=f"{n_nan} NaN cleaned titles (handled downstream by fillna(''))")
    if "main_category" in df:
        add("main_category_cardinality", int(df["main_category"].nunique()) <= 20, severity="warning",
            detail=f"{df['main_category'].nunique()} categories")
    add("no_duplicate_rows", int(df.duplicated().sum()) == 0, severity="warning",
        detail=f"{int(df.duplicated().sum())} dup rows")

    if X is not None:
        add("feature_matrix_2d", X.ndim == 2 and X.shape[1] > 0, detail=f"shape={X.shape}")
        data = X.data if sp.issparse(X) else np.asarray(X).ravel()
        add("feature_no_nan_inf", bool(np.isfinite(data).all()), detail="no NaN/inf in features")
        if y is not None:
            add("feature_target_aligned", X.shape[0] == len(y), detail=f"X rows {X.shape[0]} vs y {len(y)}")
            add("y_binary", set(np.unique(y)) <= {0, 1}, detail="labels {0,1}")

    crit_fail = [c for c in checks if c["status"] == "FAIL" and c["severity"] == "critical"]
    warn_fail = [c for c in checks if c["status"] == "FAIL" and c["severity"] == "warning"]
    return {"passed": len(crit_fail) == 0, "n_rows": int(len(df)), "n_checks": len(checks),
            "n_critical_failed": len(crit_fail), "n_warnings": len(warn_fail), "checks": checks,
            "created_utc": datetime.now(timezone.utc).isoformat()}


# ------------------------------------------------------------------ FAST (fixture) — CI-safe
def _valid_processed() -> pd.DataFrame:
    return pd.DataFrame({
        "name_clean": ["cool game", "great book"], "target": [1, 0],
        "main_category": ["Games", "Publishing"], "category": ["Tabletop", "Poetry"],
        "usd_goal_real": [1000.0, 500.0], "country": ["US", "GB"], "currency": ["USD", "GBP"]})


def test_valid_processed_passes():
    rep = build_data_validation_report(_valid_processed())
    assert rep["passed"], rep["checks"]


def test_non_binary_target_fails():
    df = _valid_processed(); df.loc[0, "target"] = 2          # illegal label
    rep = build_data_validation_report(df)
    assert not rep["passed"]
    assert any(c["name"] == "target_binary" and c["status"] == "FAIL" for c in rep["checks"])


def test_non_positive_goal_fails():
    df = _valid_processed(); df.loc[1, "usd_goal_real"] = 0.0
    rep = build_data_validation_report(df)
    assert any(c["name"] == "goal_positive" and c["status"] == "FAIL" for c in rep["checks"])


def test_feature_matrix_nan_fails():
    X = sp.csr_matrix(np.array([[1.0, np.nan], [2.0, 3.0]]))
    rep = build_data_validation_report(_valid_processed().head(2), X, np.array([0, 1]))
    assert any(c["name"] == "feature_no_nan_inf" and c["status"] == "FAIL" for c in rep["checks"])


def test_nan_name_clean_is_warning_not_failure():
    df = _valid_processed(); df.loc[0, "name_clean"] = np.nan   # downstream-handled quality wart
    rep = build_data_validation_report(df)
    assert rep["passed"]                                        # warnings don't invalidate the data
    assert rep["n_warnings"] == 1
    assert any(c["name"] == "name_clean_present" and c["severity"] == "warning" for c in rep["checks"])


# ------------------------------------------------------------------ HEAVY (real data) — local only
@pytest.mark.heavy
def test_real_processed_dataset_valid_and_report():
    df = pd.read_csv(_REPO / "data/processed/kickstarter_processed.csv")
    X = sp.load_npz(_REPO / "artifacts/fused_features/X_train_selected.npz")
    y = np.load(_REPO / "artifacts/fused_features/y_train.npy")

    rep = build_data_validation_report(df, X, y)
    out = _REPO / "artifacts/testing/data_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str))

    assert X.shape[1] == 865, f"feature width {X.shape[1]} != 865"
    assert rep["passed"], [c for c in rep["checks"] if c["status"] == "FAIL" and c["severity"] == "critical"]
    # the known, downstream-handled wart: 4 NaN cleaned titles surface as a warning, not a failure
    nc = next(c for c in rep["checks"] if c["name"] == "name_clean_present")
    assert nc["severity"] == "warning"
