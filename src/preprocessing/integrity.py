"""Phase F — Data Integrity & Business Consistency.

Phase A asked "is the data *readable*?". This phase asks a deeper question:
**"does each row make business *sense*?"** A row can be perfectly typed and still
be impossible (a campaign that ends before it starts, a $0 goal, a 1970 launch).

Rules (V1–V8), derived from the Day 1 Data Quality Report:

======  =========================================  =========
Rule    Check                                       Action
======  =========================================  =========
V1      goal > 0                                    DROP
V2      campaign duration > 0 (in whole days)       FLAG  (sub-1-day / short)
V3      campaign duration <= 90                     FLAG
V4      launch date < deadline                      DROP
V5      launched != 1970 epoch sentinel             DROP
V6      state in the valid set                      DROP
V7      country != 'N,0"' (corrupt cohort)          DROP (quarantine)
V8      required business fields present            FLAG
======  =========================================  =========

Note: V4 (timestamp: launched < deadline) is the true "impossible campaign" check.
V2 uses ``.dt.days`` which truncates hours, so it fires on sub-24-hour campaigns —
those are *unusually short*, not impossible, so V2 is a FLAG (kept), not a DROP.

Duration is computed **internally** (``deadline - launched``); it does not depend
on Day-3 feature engineering.

Row-changing -> **Stage 1**, not an sklearn transformer.

Training vs inference (frozen policy):
- ``mode="train"``      -> dropped rows are QUARANTINED to CSV then removed.
- ``mode="inference"``  -> nothing is dropped; a boolean ``integrity_valid``
                          column flags rows so the caller can REJECT them
                          (you must never silently drop the row you were asked
                          to predict).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

# Rules that remove a row vs. rules that only flag it.
# V2 (sub-1-day, a .dt.days truncation artifact) is a FLAG, not a DROP — the true
# "impossible campaign" check is V4 (launched < deadline by timestamp).
_DROP_RULES = ["V1", "V4", "V5", "V6", "V7"]
_FLAG_RULES = ["V2", "V3", "V8"]


class DataIntegrityValidator:
    """Apply business-rule validation; quarantine/flag invalid rows."""

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        mode: str = "train",
        quarantine_path: Optional[str] = None,
        logger=None,
    ) -> None:
        if mode not in {"train", "inference"}:
            raise ValueError("mode must be 'train' or 'inference'")
        self.mode = mode
        self.logger = logger or get_logger("avip.preprocessing.integrity")
        cfg = config if config is not None else load_config()

        ic = cfg.get("integrity", {})
        self.min_goal = float(ic.get("min_goal", 0.0))
        self.max_duration = int(ic.get("max_duration_days", 90))
        self.epoch_year = int(ic.get("epoch_sentinel_year", 1970))
        self.sentinel = ic.get("corrupt_country_sentinel", 'N,0"')
        self.valid_states = list(ic.get("valid_states", []))

        self.quarantine_path = quarantine_path or cfg.get("dataset", {}).get(
            "quarantine_path", "data/interim/quarantined_invalid_rows.csv"
        )
        # Business fields that must be present (V8).
        self.required_fields = [
            "goal", "state", "launched", "deadline",
            "category", "main_category", "currency", "country",
        ]
        self.report_: dict[str, Any] = {}

    # ------------------------------------------------------------ rule masks
    def _violation_masks(self, X: pd.DataFrame) -> dict[str, pd.Series]:
        """Return a boolean mask per rule (True = row VIOLATES the rule)."""
        idx = X.index
        false = pd.Series(False, index=idx)
        duration = (X["deadline"] - X["launched"]).dt.days

        masks = {
            "V1": ~(X["goal"] > self.min_goal) if "goal" in X else false,
            "V2": ~(duration > 0),
            "V3": ~(duration <= self.max_duration),
            "V4": ~(X["launched"] < X["deadline"]),
            "V5": (X["launched"].dt.year == self.epoch_year).fillna(False),
            "V6": ~X["state"].isin(self.valid_states) if "state" in X else false,
            "V7": X["country"].astype("string") == self.sentinel if "country" in X else false,
            "V8": X[[c for c in self.required_fields if c in X.columns]].isna().any(axis=1),
        }
        # NaN comparisons yield False; ~False = True correctly marks bad/missing.
        return {k: v.fillna(False).astype(bool) for k, v in masks.items()}

    # ------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        masks = self._violation_masks(X)

        drop_mask = pd.Series(False, index=X.index)
        for rule in _DROP_RULES:
            drop_mask |= masks[rule]

        # Per-row reason string (which drop rules it failed).
        reasons = pd.Series("", index=X.index)
        for rule in _DROP_RULES:
            hit = masks[rule]
            reasons = reasons.where(~hit, reasons.str.cat(pd.Series(f"{rule};", index=X.index)))

        self.report_ = {
            "total_rows": len(X),
            "violations_per_rule": {r: int(masks[r].sum()) for r in _DROP_RULES + _FLAG_RULES},
            "rows_dropped": int(drop_mask.sum()),
            "rows_flagged_kept": {r: int(masks[r].sum()) for r in _FLAG_RULES},
            "rows_remaining": int((~drop_mask).sum()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.logger.info("Integrity report: %s", self.report_)

        if self.mode == "inference":
            out = X.copy()
            out["integrity_valid"] = ~drop_mask
            self.logger.info(
                "Integrity (inference): flagged %d invalid row(s); none dropped.",
                int(drop_mask.sum()),
            )
            return out

        # train mode: quarantine then remove
        if drop_mask.any():
            self._quarantine(X.loc[drop_mask].assign(quarantine_reason=reasons[drop_mask]))
        kept = X.loc[~drop_mask].reset_index(drop=True)
        self.logger.info(
            "Integrity (train): dropped %d row(s), quarantined to %s; %d remain.",
            int(drop_mask.sum()),
            self.quarantine_path,
            len(kept),
        )
        return kept

    def fit(self, X: pd.DataFrame, y=None) -> "DataIntegrityValidator":
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------- quarantine
    def _quarantine(self, rows: pd.DataFrame) -> None:
        path = Path(self.quarantine_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(path, index=False)
        self.logger.warning("Quarantined %d row(s) -> %s", len(rows), path)

    def report(self) -> dict[str, Any]:
        return self.report_
