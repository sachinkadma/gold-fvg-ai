"""
Pytest causality verification suite.
Validates zero look-ahead bias across strategy signal generation.
"""

import pytest
import numpy as np
import pandas as pd
import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from core.data_loader import load_and_clean_data
    from core import config
    from core.signals import evaluate_causal_signal_at_bar
except ImportError:
    from data_loader import load_and_clean_data
    import config
    from signals import evaluate_causal_signal_at_bar


@pytest.fixture(scope="module")
def gold_data():
    df = load_and_clean_data(config.DATA_PATH, data_tz=config.DATA_TZ, verbose=False)
    return df


def test_no_lookahead(gold_data):
    """
    Causality verification:
    For 20 random timestamps, compute signal at timestamp t using full data,
    then recompute signal at timestamp t using data strictly truncated at t.
    The signal at that timestamp must be identical.
    """
    df = gold_data
    n_bars = len(df)
    assert n_bars > 200, "Insufficient bars in dataset"

    # Pick 20 random timestamps across the dataset
    np.random.seed(42)
    random_indices = list(np.random.choice(range(100, n_bars - 1), size=20, replace=False))

    for idx in random_indices:
        # 1. Signal evaluated at idx using the FULL dataframe
        signal_full = evaluate_causal_signal_at_bar(df, idx)

        # 2. Data strictly truncated at idx (zero future bars exist in the DataFrame)
        df_truncated = df.iloc[: idx + 1].copy()
        signal_truncated = evaluate_causal_signal_at_bar(df_truncated, len(df_truncated) - 1)

        # 3. Assert exact equality
        if signal_full is None:
            assert signal_truncated is None, f"Lookahead mismatch at index {idx}: full=None, truncated={signal_truncated}"
        else:
            assert signal_truncated is not None, f"Lookahead mismatch at index {idx}: full={signal_full}, truncated=None"
            assert signal_full["direction"] == signal_truncated["direction"]
            assert signal_full["entry"] == signal_truncated["entry"]
            assert signal_full["sl"] == signal_truncated["sl"]
            assert signal_full["tp"] == signal_truncated["tp"]
            assert signal_full["killzone"] == signal_truncated["killzone"]
