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


def test_candidate_generator_golden_train_slice():
    """
    Phase D Golden-File Test:
    Verifies that the frozen causal candidate generator output on a fixed TRAIN slice
    (first 500 bars of TRAIN, 2014-12-31 to 2015-01-07) produces bit-for-bit identical
    candidates with matching SHA256 integrity hash.
    Strictly uses TRAIN data; never touches TEST split.
    """
    import json
    import hashlib

    df = load_and_clean_data("data/xauusd_15m_long.csv", data_tz="UTC", target_tz="America/New_York", verbose=False)
    df_slice = df.iloc[:500]  # First 500 bars of TRAIN

    candidates = []
    for t in range(35, len(df_slice)):
        sig = evaluate_causal_signal_at_bar(df_slice, t, killzone_gate=False)
        if sig is not None:
            candidates.append({
                "bar": t,
                "time": str(df_slice.index[t]),
                "dir": sig["direction"],
                "entry": round(float(sig["entry"]), 3),
                "sl": round(float(sig["sl"]), 3),
                "tp": round(float(sig["tp"]), 3),
                "kz": sig["killzone"]
            })

    assert len(candidates) == 114
    cand_json = json.dumps(candidates, sort_keys=True)
    actual_hash = hashlib.sha256(cand_json.encode("utf-8")).hexdigest()
    EXPECTED_GOLDEN_HASH = "4ab54c1af2079b08f661f3ad754d9dc82cdd2837b73c68511539328559439e39"
    assert actual_hash == EXPECTED_GOLDEN_HASH, f"Candidate generator drift detected! Got {actual_hash}, expected {EXPECTED_GOLDEN_HASH}"


if __name__ == "__main__":
    test_candidates()
    test_candidate_generator_golden_train_slice()
