"""Day 17 · Phase D — model validation (calibrated LightGBM incumbent + LogReg baseline).

All `heavy` (needs the real trained model + fused feature matrices). We assert **thresholds and
properties**, not frozen decimals (Phase A): the model clears quality bars, beats the mandatory
baseline, isn't badly over-fit, predicts deterministically, and re-training from config reproduces the
incumbent (Day-16 guarantee). Neural-Network validation is **deferred** — no NN was ever trained.
Report → `artifacts/testing/model_validation.json`.
"""
from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pytest
import scipy.sparse as sp

from src.training.trainer import _metrics_at_threshold

_REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.heavy                       # every test here needs real artifacts


def _load():
    model = joblib.load(_REPO / "artifacts/models/calibrated_model.joblib")
    meta = json.loads((_REPO / "artifacts/models/model_metadata.json").read_text())
    Xtr = sp.load_npz(_REPO / "artifacts/fused_features/X_train_selected.npz")
    ytr = np.load(_REPO / "artifacts/fused_features/y_train.npy")
    Xte = sp.load_npz(_REPO / "artifacts/fused_features/X_test_selected.npz")
    yte = np.load(_REPO / "artifacts/fused_features/y_test.npy")
    logreg = json.loads((_REPO / "artifacts/models/logreg_results.json").read_text())["logreg_l2"]["metrics"]
    return model, meta, Xtr, ytr, Xte, yte, logreg


def test_calibrated_model_meets_quality_thresholds():
    model, meta, Xtr, ytr, Xte, yte, logreg = _load()
    thr = float(meta["decision_threshold"])
    test_m = _metrics_at_threshold(model, Xte, yte, thr)
    train_m = _metrics_at_threshold(model, Xtr, ytr, thr)
    gap = round(train_m["roc_auc"] - test_m["roc_auc"], 4)

    checks = {
        "pr_auc>=0.68": test_m["pr_auc"] >= 0.68,
        "roc_auc>=0.77": test_m["roc_auc"] >= 0.77,
        "brier<=0.20": test_m["brier"] <= 0.20,
        "f1>=0.63": test_m["f1"] >= 0.63,
        "recall>=0.75": test_m["recall"] >= 0.75,
        "beats_logreg_baseline": test_m["pr_auc"] > logreg["pr_auc"] + 0.02,
        "no_overfit_gap<0.10": gap < 0.10,
    }
    report = {
        "model": meta["model"]["type"], "threshold": thr,
        "test_metrics": test_m, "train_metrics": train_m, "roc_auc_gap": gap,
        "baseline_logreg": logreg, "checks": checks,
        "passed": all(checks.values()),
        "deferred": {"neural_network": "no NN trained (torch is SBERT-only) — validation deferred"},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = _REPO / "artifacts/testing/model_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))

    assert report["passed"], {k: v for k, v in checks.items() if not v}


def test_beats_mandatory_baseline():
    model, meta, _, _, Xte, yte, logreg = _load()
    test_m = _metrics_at_threshold(model, Xte, yte, float(meta["decision_threshold"]))
    assert test_m["pr_auc"] > logreg["pr_auc"], f"{test_m['pr_auc']} !> baseline {logreg['pr_auc']}"


def test_not_overfit():
    model, meta, Xtr, ytr, Xte, yte, _ = _load()
    thr = float(meta["decision_threshold"])
    gap = _metrics_at_threshold(model, Xtr, ytr, thr)["roc_auc"] - _metrics_at_threshold(model, Xte, yte, thr)["roc_auc"]
    assert gap < 0.10, f"train-test ROC-AUC gap {gap:.4f} suggests overfitting"


def test_inference_is_deterministic():
    model, _, _, _, Xte, _, _ = _load()
    p1 = model.predict_proba(Xte[:1000])[:, 1]
    p2 = model.predict_proba(Xte[:1000])[:, 1]
    assert np.array_equal(p1, p2)                    # same input -> identical probabilities


def test_training_reproduces_incumbent():
    """Re-training from the recorded config reproduces the incumbent (Day-16 guarantee), sandboxed."""
    from src.training.train_pipeline import TrainingPipeline
    from src.utils.config import load_config
    cfg = copy.deepcopy(load_config())
    sandbox = "artifacts/_test_modelval"                          # repo-relative (run() uses relative_to)
    cfg["day16"]["training"]["runs_dir"] = f"{sandbox}/runs"
    try:
        rep = TrainingPipeline(cfg).run(register=False, track=False)   # no registry/mlflow side effects
        assert rep["reproduces_incumbent"] is True, f"test PR-AUC {rep['test_metrics']['pr_auc']}"
    finally:
        shutil.rmtree(_REPO / sandbox, ignore_errors=True)
