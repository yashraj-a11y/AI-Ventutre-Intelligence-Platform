"""Day 14 · Phase E — Prediction service (Day-4 PredictionEngine)."""
from __future__ import annotations
from typing import Any
import pandas as pd
from src.services.base import campaign_of, summary


def predict(registry: Any, req: Any) -> dict[str, Any]:
    """Funding prediction for one campaign."""
    camp = campaign_of(req)
    out = registry.engine.predict(pd.DataFrame([camp]))[0].to_dict()
    return {"campaign": summary(camp),
            "funding_probability": out["funding_probability"],
            "predicted_label": out["predicted_label"],
            "decision_threshold": out["decision_threshold"]}
