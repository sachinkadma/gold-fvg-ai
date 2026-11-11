"""
Test candidate generation across all 24 hours of the day (killzone_gate=False).
"""

import time
import pandas as pd
import numpy as np
import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from core import config
    from core.data_loader import load_and_clean_data
    from core.signals import evaluate_causal_signal_at_bar
except ImportError:
    import config
    from data_loader import load_and_clean_data
    from signals import evaluate_causal_signal_at_bar

def test_candidates():
    df = load_and_clean_data("data/xauusd_15m_long.csv", data_tz="UTC", target_tz="America/New_York", verbose=False)
    df_train = df.loc["2015-01-01":"2021-12-31"]

    t0 = time.time()
    candidates = []
    n = len(df_train)
    
    # Track when an order is pending or active
    # Note: To avoid duplicate overlapping entries from identical swings,
    # evaluate new signals when no pending order is active, or evaluate every candle
    # Let's check how many candidates fire if we evaluate every candle
    sample_bars = 500
    for t in range(35, min(sample_bars, n)):
        sig = evaluate_causal_signal_at_bar(df_train, t, killzone_gate=False)
        if sig is not None:
            candidates.append({
                "bar": t,
                "time": df_train.index[t],
                "dir": sig["direction"],
                "entry": sig["entry"],
                "sl": sig["sl"],
                "kz": sig["killzone"]
            })
    t1 = time.time()
    assert isinstance(candidates, list)

if __name__ == "__main__":
    test_candidates()
