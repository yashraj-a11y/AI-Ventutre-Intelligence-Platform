"""Day 14 · Phase C — Request schema (the API's honest input contract).

``CampaignRequest`` accepts ONLY real Kickstarter campaign fields — the ones the Day-3 feature
pipeline actually consumes. ``model_config = extra="forbid"`` means any fabricated field the platform
never modeled (founder profile / MRR / pitch-deck metadata / moat / expected return) is **rejected
with a 422** at the boundary. This is where the platform's honesty is *enforced*, not just documented.

Derived / label fields are handled correctly:
- ``name_clean`` is **derived** from ``name`` via the Day-2 :class:`TextCleaner` (train/serve
  consistency), not required from the caller.
- ``target`` / ``state`` are outcome labels — **not** request fields (that's what we predict).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.preprocessing.cleaning import TextCleaner

_CLEANER = TextCleaner()   # reused to derive name_clean exactly as training did


class CampaignRequest(BaseModel):
    """A Kickstarter campaign to analyze — real fields only (fabricated fields are 422-rejected)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "name": "Artificer - Board Game", "main_category": "Games", "category": "Tabletop Games",
        "currency": "USD", "country": "US", "usd_goal_real": 30000.0,
        "launched": "2016-05-01 12:00:00", "deadline": "2016-05-31 12:00:00"}})

    name: str = Field(..., min_length=1, description="Campaign title.")
    main_category: str = Field(..., description="Top-level category, e.g. 'Games'.")
    category: Optional[str] = Field(None, description="Sub-category; defaults to main_category.")
    currency: str = Field("USD", description="ISO currency code.")
    country: str = Field("US", description="ISO country code.")
    usd_goal_real: float = Field(..., gt=0, description="Funding goal in USD (> 0).")
    goal: Optional[float] = Field(None, gt=0, description="Goal in the campaign currency; defaults to usd_goal_real.")
    launched: str = Field(..., description="Launch datetime (parseable), e.g. '2016-05-01 12:00:00'.")
    deadline: str = Field(..., description="Deadline datetime (parseable), after launched.")
    name_clean: Optional[str] = Field(None, description="Derived from name if omitted (do not usually send).")

    @model_validator(mode="after")
    def _defaults_and_checks(self) -> "CampaignRequest":
        if self.category is None:
            self.category = self.main_category
        if self.goal is None:
            self.goal = self.usd_goal_real
        launched = pd.to_datetime(self.launched, errors="coerce")
        deadline = pd.to_datetime(self.deadline, errors="coerce")
        if pd.isna(launched) or pd.isna(deadline):
            raise ValueError("launched and deadline must be parseable datetimes")
        if deadline <= launched:
            raise ValueError("deadline must be after launched")
        return self

    def to_campaign(self) -> dict:
        """The raw campaign dict the pipelines consume (with name_clean derived if absent)."""
        raw = self.model_dump()
        if not raw.get("name_clean"):
            raw["name_clean"] = str(_CLEANER.transform(pd.DataFrame([{"name": self.name}]))
                                    ["name_clean"].iloc[0])
        return raw
