"""
Automated Causality and Anti-Lookahead Test Suite.
Verifies that evaluating signals at any bar t yields identical results
whether evaluated on the full historical dataset or on data strictly truncated at t.
"""

import random
import numpy as np
import pytest
import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from core.data_loader import load_and_clean_data
    from core.signals import evaluate_causal_signal_at_bar
except ImportError:
    from data_loader import load_and_clean_data
    from signals import evaluate_causal_signal_at_bar


def test_no_lookahead(dataset):
    """
    For 20 random timestamps, compute signal using full data,
    then recompute using data truncated at that timestamp.
    The signal at that timestamp must be identical.
    Fails if any future look-ahead leakage exists.
    """
    df = dataset
    n = len(df)
    assert n >= 500, f"Insufficient data length for test: {n}"

    # Set deterministic seed for reproducibility
    random.seed(42)

    # Pick 20 random timestamps with sufficient lookback history
    test_indices = random.sample(range(100, n - 10), 20)

    for idx in test_indices:
        ts = df.index[idx]

        # 1. Evaluate signal on full data at bar idx
        sig_full = evaluate_causal_signal_at_bar(df, idx)

        # 2. Evaluate signal on truncated dataset up to idx (strictly no future bars)
        df_truncated = df.iloc[: idx + 1].copy()
        sig_truncated = evaluate_causal_signal_at_bar(df_truncated, idx)

        # Assert identical results
        if sig_full is None:
            assert sig_truncated is None, (
                f"Lookahead failure at {ts} (idx {idx}): "
                f"full data returned None but truncated returned {sig_truncated}"
            )
        else:
            assert sig_truncated is not None, (
                f"Lookahead failure at {ts} (idx {idx}): "
                f"full data returned {sig_full} but truncated returned None"
            )

            # Check key fields match exactly
            assert sig_full["direction"] == sig_truncated["direction"]
            assert sig_full["entry"] == sig_truncated["entry"]
            assert sig_full["sl"] == sig_truncated["sl"]
            assert sig_full["tp"] == sig_truncated["tp"]
            assert sig_full["killzone"] == sig_truncated["killzone"]
            assert sig_full["risk_dist"] == sig_truncated["risk_dist"]
            assert sig_full["sweep_level"] == sig_truncated["sweep_level"]


def test_intentional_lookahead_fails(dataset):
    """
    Sanity test: Confirms that an intentional future leak (e.g. checking t+1)
    would fail the strict anti-lookahead check.
    """
    df = dataset
    idx = 200

    # Simulate an engine that peeks at future high
    def lookahead_engine(d, t):
        if t + 1 < len(d):
            future_high = d["high"].iloc[t + 1]
            return {"peek": float(future_high)}
        return None

    full_res = lookahead_engine(df, idx)
    df_trunc = df.iloc[: idx + 1]
    trunc_res = lookahead_engine(df_trunc, idx)

    # The truncated evaluation should NOT have access to t+1
    assert full_res != trunc_res, "Validation check: peeking at t+1 must cause a mismatch."
