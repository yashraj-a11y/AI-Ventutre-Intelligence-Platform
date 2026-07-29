"""Phase B — Venture Knowledge Base (Day-6, Architecture Layer 2).

The platform's **historical memory**. :class:`VentureKnowledgeBase` turns the *train*
campaigns (known outcomes) into two aligned things:

1. a **structured metadata table** over the real columns only — title, category,
   main_category, country, currency, goal, usd_goal_real, launch year, and the known
   ``outcome`` (funded/failed) — used later for filtering and evidence aggregation;
2. a **serialized venture-record string** per campaign — a short natural-language
   sentence built from those fields — which Phase C embeds with SBERT.

Two decisions from the frozen spec, enforced here:

- **Train split only.** The KB is *historical* knowledge with known outcomes; test / new
  campaigns query it. Building from the persisted ``train_index`` keeps "evidence supports
  the prediction" non-circular (a query's own outcome is never in the KB it searches).
- **Real columns only — no fabrication.** There are no founder / valuation / exit / team /
  market fields in this dataset, so the schema simply *omits* them (not stored as empty).
  Any "founder benchmark" retrieval is deferred downstream, never faked.

Why a *serialized record* rather than the bare title: a 5.6-word title carries weak
semantic signal, so each record folds the structured facts into the sentence — e.g.
*"A Winter Storm - A Novel. A Fiction project in the Publishing category, based in the
United States, seeking a modest goal of about $6,000 (launched 2015)."* — giving the
embedding both title semantics and structured context.

Example
-------
>>> kb = VentureKnowledgeBase().build()
>>> len(kb.records()), kb.metadata().shape
(265171, (265171, 13))
>>> kb.records()[0]
'A Winter Storm - A Novel. A Fiction project in the Publishing category, ...'
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

# 22 ISO country codes present in the data -> readable names (fallback: the code).
_COUNTRY_NAMES = {
    "US": "the United States", "GB": "the United Kingdom", "CA": "Canada",
    "AU": "Australia", "DE": "Germany", "FR": "France", "IT": "Italy", "ES": "Spain",
    "NL": "the Netherlands", "SE": "Sweden", "NZ": "New Zealand", "JP": "Japan",
    "MX": "Mexico", "DK": "Denmark", "IE": "Ireland", "CH": "Switzerland",
    "NO": "Norway", "HK": "Hong Kong", "SG": "Singapore", "AT": "Austria",
    "BE": "Belgium", "LU": "Luxembourg",
}
# Goal bands (on usd_goal_real) -> a human magnitude phrase carried in the text.
_GOAL_BANDS = [
    (1_000, "a very small goal"),
    (5_000, "a small goal"),
    (20_000, "a modest goal"),
    (100_000, "a large goal"),
    (float("inf"), "a very large goal"),
]


def _a(word: str) -> str:
    """'a'/'an' by the following word's initial (our category names have no traps)."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def goal_band(usd_goal: float) -> str:
    """Human magnitude label for a goal in USD."""
    for hi, label in _GOAL_BANDS:
        if usd_goal < hi:
            return label
    return _GOAL_BANDS[-1][1]


class VentureKnowledgeBase:
    """Build the train-split historical KB: metadata table + serialized records.

    Parameters
    ----------
    config:
        Parsed configuration. If ``None``, loaded from ``configs/preprocessing.yaml``.
    logger:
        Optional logger; a namespaced one is created if not supplied.
    """

    _FIELDS = ["campaign_id", "title", "category", "main_category", "country",
               "currency", "goal", "usd_goal_real", "launch_year", "target",
               "outcome", "goal_band", "text"]

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.kb.venture_database")

        c = self.config.get("day6", {}).get("knowledge_base", {})
        self.processed_path = Path(c.get("processed_path",
                                         "data/processed/kickstarter_processed.csv"))
        self.split_dir = Path(c.get("split_dir", "artifacts/splits"))
        self.use_split = c.get("use_split", "train")
        self.subsample = c.get("subsample", None)
        self.random_state = int(c.get("random_state", 42))
        self.out_dir = Path(c.get("out_dir", "artifacts/embeddings"))
        self.metadata_csv = Path(c.get("metadata_csv", self.out_dir / "metadata.csv"))
        self.manifest_path = Path(c.get("manifest_path", self.out_dir / "metadata.json"))

        self.metadata_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ public
    def build(self) -> "VentureKnowledgeBase":
        """Load the processed data, select the train split, build records + metadata."""
        df = pd.read_csv(self.processed_path)
        idx = np.load(self.split_dir / f"{self.use_split}_index.npy")
        sub = df.iloc[idx].copy()
        sub["campaign_id"] = idx                       # trace back to the processed row
        if self.subsample:                             # optional fast-dev sample
            sub = sub.sample(min(int(self.subsample), len(sub)),
                             random_state=self.random_state)

        sub["title"] = sub["name"].fillna("").astype(str).str.strip()
        sub["launch_year"] = pd.to_datetime(sub["launched"], errors="coerce").dt.year
        sub["outcome"] = np.where(sub["target"] == 1, "successful", "failed")
        sub["goal_band"] = sub["usd_goal_real"].apply(goal_band)
        sub["text"] = sub.apply(self.to_text, axis=1)

        self.metadata_ = sub[self._FIELDS].reset_index(drop=True)
        self.logger.info(
            "KB built from %s split: %d records, %.4f success rate.",
            self.use_split, len(self.metadata_), float(sub["target"].mean()),
        )
        return self

    def to_text(self, row: pd.Series) -> str:
        """Serialize one campaign into a natural-language venture record."""
        cat = str(row["category"])
        main = str(row["main_category"])
        country = _COUNTRY_NAMES.get(str(row["country"]), str(row["country"]))
        band = row["goal_band"] if "goal_band" in row and pd.notna(row.get("goal_band")) \
            else goal_band(float(row["usd_goal_real"]))
        amount = float(row["usd_goal_real"])
        year = row.get("launch_year")
        title = str(row.get("title") or row.get("name") or "").strip()

        sentence = (f"{_a(cat).capitalize()} {cat} project in the {main} category, "
                    f"based in {country}, seeking {band} of about ${amount:,.0f}")
        if pd.notna(year):
            sentence += f" (launched {int(year)})"
        sentence += "."
        # Prepend the human title (semantic signal) when present.
        text = f"{title}. {sentence}" if title else sentence
        return _fix_article(text)

    def records(self) -> list[str]:
        """The serialized venture-record strings (Phase-C embedding input)."""
        self._ensure()
        return self.metadata_["text"].tolist()

    def metadata(self) -> pd.DataFrame:
        """The structured metadata table (real columns + outcome + text)."""
        self._ensure()
        return self.metadata_

    def save(self) -> dict[str, Path]:
        """Persist the full table (CSV) + a compact JSON manifest."""
        self._ensure()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_.to_csv(self.metadata_csv, index=False)
        self.manifest_path.write_text(json.dumps(self._manifest(), indent=2))
        self.logger.info("Saved KB metadata -> %s and manifest -> %s",
                         self.metadata_csv, self.manifest_path)
        return {"metadata_csv": self.metadata_csv, "manifest": self.manifest_path}

    # --------------------------------------------------------------- internals
    def _ensure(self) -> None:
        if self.metadata_ is None:
            self.build()

    def _manifest(self) -> dict[str, Any]:
        m = self.metadata_
        return {
            "source": str(self.processed_path),
            "split": self.use_split,
            "n_records": int(len(m)),
            "success_rate": round(float((m["target"] == 1).mean()), 4),
            "fields": self._FIELDS,
            "n_main_category": int(m["main_category"].nunique()),
            "n_category": int(m["category"].nunique()),
            "n_country": int(m["country"].nunique()),
            "goal_band_counts": m["goal_band"].value_counts().to_dict(),
            "deferred_fields": ["founder", "team_size", "valuation", "exit_status",
                                "business_model", "financial_metrics", "market_reports"],
            "sample_records": m["text"].head(3).tolist(),
        }


def _fix_article(text: str) -> str:
    """'a X' -> 'an X' before a vowel (e.g. 'A Experimental' -> 'An Experimental')."""
    return re.sub(r"\b([Aa]) ([aeiouAEIOU])", lambda mm: ("An" if mm.group(1) == "A" else "an") + " " + mm.group(2), text)
