"""Day 17 · Phase F — explainability validation (SHAP explains the real model, additively).

All `heavy` (loads the model + SHAP explainer). The core property is **additivity**: in log-odds
margin space, ``base_value + Σ shap ≈ model_margin`` — the SHAP explainer asserts this internally and
exposes ``additivity_error``. We also verify the explanation reports the model's *actual* calibrated
probability, that contribution signs are consistent, and that explanations are deterministic. Report →
`artifacts/testing/explainability_validation.json`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

_REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.heavy


@pytest.fixture(scope="module")
def setup():
    from src.explainability.local_explainer import LocalExplainer
    le = LocalExplainer()
    Xte = sp.load_npz(_REPO / "artifacts/fused_features/X_test_selected.npz")
    return le, Xte


def test_shap_additivity_holds(setup):
    """base_value + Σ shap reconstructs the model's raw margin (the additivity guarantee)."""
    le, Xte = setup
    res = le.shap.explain(Xte[:8].toarray())              # (8, 865) signed SHAP in log-odds space
    recon = res.base_value + res.values.sum(axis=1)       # per-row reconstructed margin
    assert res.additivity_error < 1e-3, res.additivity_error
    assert np.all(np.isfinite(recon))


def test_local_margin_equals_base_plus_full_shap(setup):
    """The LocalExplanation.margin equals base_value + the FULL shap sum (not just top-k)."""
    le, Xte = setup
    res = le.shap.explain(Xte[7].toarray())
    exp = le.explain_transformed(Xte[7])
    recon = round(float(res.base_value + res.values[0].sum()), 4)
    assert abs(exp.margin - recon) < 1e-3, f"margin {exp.margin} vs base+Σshap {recon}"


def test_explanation_reports_models_actual_probability(setup):
    """The explanation's probability IS the calibrated model's prediction for that row."""
    le, Xte = setup
    exp = le.explain_transformed(Xte[7])
    p_model = round(float(le.engine.model.predict_proba(Xte[7])[0, 1]), 4)
    assert exp.funding_probability == p_model
    assert exp.predicted_label == ("successful" if p_model >= exp.decision_threshold else "failed")


def test_contribution_signs_are_consistent(setup):
    le, Xte = setup
    exp = le.explain_transformed(Xte[7])
    assert all(c.contribution > 0 for c in exp.top_positive)   # positives push funding up
    assert all(c.contribution < 0 for c in exp.top_negative)   # negatives push it down


def test_explanation_is_deterministic(setup):
    le, Xte = setup
    a = le.explain_transformed(Xte[7])
    b = le.explain_transformed(Xte[7])
    assert a.funding_probability == b.funding_probability
    assert [c.feature for c in a.top_positive] == [c.feature for c in b.top_positive]


def test_writes_report(setup):
    le, Xte = setup
    res = le.shap.explain(Xte[:8].toarray())
    exp = le.explain_transformed(Xte[7])
    checks = {
        "additivity_error<1e-3": res.additivity_error < 1e-3,
        "margin_reconstructs": abs(exp.margin - round(float(res.base_value + le.shap.explain(Xte[7].toarray()).values[0].sum()), 4)) < 1e-3,
        "positive_signs": all(c.contribution > 0 for c in exp.top_positive),
        "negative_signs": all(c.contribution < 0 for c in exp.top_negative),
    }
    report = {"additivity_error": res.additivity_error, "sample_probability": exp.funding_probability,
              "sample_margin": exp.margin, "base_value": exp.base_value, "checks": checks,
              "passed": all(checks.values()),
              "honesty_note": "SHAP explains the uncalibrated log-odds margin; calibration is a monotonic squash",
              "created_utc": datetime.now(timezone.utc).isoformat()}
    out = _REPO / "artifacts/testing/explainability_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    assert report["passed"], {k: v for k, v in checks.items() if not v}
