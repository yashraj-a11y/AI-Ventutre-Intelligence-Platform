"""Day 14 · Phase E — Explainability service (Day-5 LocalExplainer / SHAP)."""
from __future__ import annotations
from typing import Any
import pandas as pd
from src.services.base import campaign_of


def explain(registry: Any, req: Any) -> dict[str, Any]:
    """Top +/- real-feature drivers of the prediction."""
    camp = campaign_of(req)
    e = registry.explainer.explain_raw(pd.DataFrame([camp]))[0].to_dict()
    return {"funding_probability": e["funding_probability"],
            "predicted_label": e["predicted_label"],
            "top_positive": e["top_positive"], "top_negative": e["top_negative"],
            "honesty_note": e.get("honesty_note")}
