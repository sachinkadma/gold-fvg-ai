"""
Global pytest fixtures for test suite.
Loads historical datasets once per test session and shares them across test modules.
Caches cleaned parquet for instant sub-second retrieval.
"""

import os
import sys
import pytest
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.data_loader import load_and_clean_data

_TRAIN_DATA_CACHE = None
_GOLD_DATA_CACHE = None


@pytest.fixture(scope="session")
def train_data():
    """
    Session-scoped fixture providing the cleaned TRAIN dataset (xauusd_15m_long.csv).
    Loaded once and shared across all test files.
    Strictly sliced to the TRAIN period (up to 2021-12-31) so tests never touch TEST rows.
    """
    global _TRAIN_DATA_CACHE
    if _TRAIN_DATA_CACHE is None:
        cache_path = os.path.join(REPO_ROOT, "data", ".xauusd_15m_long_cache.parquet")
        src_path = os.path.join(REPO_ROOT, "data", "xauusd_15m_long.csv")
        if os.path.exists(cache_path) and os.path.exists(src_path) and os.path.getmtime(cache_path) >= os.path.getmtime(src_path):
            try:
                cached = pd.read_parquet(cache_path)
                if cached.index.max() <= pd.Timestamp("2021-12-31 23:59:59", tz=cached.index.tz):
                    _TRAIN_DATA_CACHE = cached
            except Exception:
                _TRAIN_DATA_CACHE = None
        if _TRAIN_DATA_CACHE is None:
            full_df = load_and_clean_data(
                src_path,
                data_tz="UTC",
                target_tz="America/New_York",
                verbose=False
            )
            _TRAIN_DATA_CACHE = full_df.loc[:"2021-12-31 23:59:59"].copy()
            try:
                _TRAIN_DATA_CACHE.to_parquet(cache_path)
            except Exception:
                pass
    return _TRAIN_DATA_CACHE


@pytest.fixture(scope="session")
def sample_dst_data(train_data):
    """Shares session train_data for DST tests."""
    return train_data


@pytest.fixture(scope="session")
def gold_data(train_data):
    """
    Session-scoped fixture providing historical data for causality and lookahead tests.
    Shares the train_data fixture so tests strictly remain in the TRAIN split
    and avoid loading external/TEST files.
    """
    return train_data


@pytest.fixture(scope="session")
def dataset(gold_data):
    """Shares gold_data for test_no_lookahead."""
    return gold_data
