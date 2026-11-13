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
    """
    global _TRAIN_DATA_CACHE
    if _TRAIN_DATA_CACHE is None:
        cache_path = os.path.join(REPO_ROOT, "data", ".xauusd_15m_long_cache.parquet")
        src_path = os.path.join(REPO_ROOT, "data", "xauusd_15m_long.csv")
        if os.path.exists(cache_path) and os.path.exists(src_path) and os.path.getmtime(cache_path) >= os.path.getmtime(src_path):
            try:
                _TRAIN_DATA_CACHE = pd.read_parquet(cache_path)
            except Exception:
                _TRAIN_DATA_CACHE = None
        if _TRAIN_DATA_CACHE is None:
            _TRAIN_DATA_CACHE = load_and_clean_data(
                src_path,
                data_tz="UTC",
                target_tz="America/New_York",
                verbose=False
            )
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
def gold_data():
    """
    Session-scoped fixture providing the default 15m dataset.
    Loaded once and shared across causality and lookahead tests.
    """
    global _GOLD_DATA_CACHE
    if _GOLD_DATA_CACHE is None:
        _GOLD_DATA_CACHE = load_and_clean_data(verbose=False)
    return _GOLD_DATA_CACHE


@pytest.fixture(scope="session")
def dataset(gold_data):
    """Shares gold_data for test_no_lookahead."""
    return gold_data
