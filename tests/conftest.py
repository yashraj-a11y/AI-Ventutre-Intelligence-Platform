"""Shared pytest fixtures.

Unit tests run on a tiny, hand-built dataframe with deliberate edge cases — NOT the
55 MB real file. Small fixtures make tests fast, deterministic, and readable: every
row exists to exercise one specific behaviour.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config


def _row(**kw):
    """Build one raw-schema row; the space-named 'usd pledged' via kwargs dict."""
    base = dict(
        category="Poetry", main_category="Publishing", currency="USD",
        pledged=100.0, backers=5, country="US",
        usd_pledged_real=100.0, usd_goal_real=1000.0,
    )
    base.update(kw)
    base["usd pledged"] = base.pop("usd_pledged", 100.0)
    return base


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """10-row raw dataset; each row targets a specific rule/behaviour."""
    rows = [
        # r1 normal successful
        _row(ID=1, name="Alpha Project", goal=1000.0,
             launched="2015-01-01 10:00:00", deadline="2015-02-01 10:00:00",
             state="successful"),
        # r2 normal failed
        _row(ID=2, name="Beta Idea", goal=5000.0,
             launched="2015-03-01 10:00:00", deadline="2015-04-01 10:00:00",
             state="failed"),
        # r3 undecided -> removed by TargetBuilder
        _row(ID=3, name="Gamma", goal=2000.0,
             launched="2015-05-01 10:00:00", deadline="2015-06-01 10:00:00",
             state="canceled"),
        # r4 epoch-1970 launch -> V5 drop
        _row(ID=4, name="Epoch Bug", goal=3000.0,
             launched="1970-01-01 01:00:00", deadline="2015-06-01 10:00:00",
             state="successful"),
        # r5 corrupt country -> V7 drop
        _row(ID=5, name="Corrupt Country", goal=4000.0, country='N,0"',
             launched="2015-02-01 10:00:00", deadline="2015-03-01 10:00:00",
             state="failed"),
        # r6 non-positive goal -> V1 drop
        _row(ID=6, name="Zero Goal", goal=0.0,
             launched="2015-02-01 10:00:00", deadline="2015-03-01 10:00:00",
             state="successful"),
        # r7 sub-day duration -> V2 FLAG (kept)
        _row(ID=7, name="Same Day", goal=1500.0,
             launched="2015-05-01 09:00:00", deadline="2015-05-01 17:00:00",
             state="successful"),
        # r8 outcome marker in title -> TextCleaner strips it
        _row(ID=8, name="Cool Thing (Canceled)", goal=2500.0,
             launched="2015-07-01 10:00:00", deadline="2015-08-01 10:00:00",
             state="failed"),
        # r9 missing title -> MissingValueHandler fills ""
        _row(ID=9, name=np.nan, goal=1200.0,
             launched="2015-08-01 10:00:00", deadline="2015-09-01 10:00:00",
             state="successful"),
        # r10 normal failed
        _row(ID=10, name="Omega", goal=800.0,
             launched="2015-09-01 10:00:00", deadline="2015-10-01 10:00:00",
             state="failed"),
    ]
    cols = [
        "ID", "name", "category", "main_category", "currency", "deadline", "goal",
        "launched", "pledged", "state", "backers", "country", "usd pledged",
        "usd_pledged_real", "usd_goal_real",
    ]
    return pd.DataFrame(rows)[cols]


@pytest.fixture
def base_config():
    """A fresh, deep-copied config per test (so overrides don't leak between tests)."""
    return copy.deepcopy(load_config())
