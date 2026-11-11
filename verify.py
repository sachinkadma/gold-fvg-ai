"""
Automated Verification Script.
Runs test suite, re-executes baseline simulation on TRAIN,
and verifies byte-for-byte SHA256 reproducibility of trades.csv.
Quiet output everywhere: outputs PASS or FAIL.
"""

import sys
import os
import hashlib
import subprocess

EXPECTED_TRADES_SHA256 = "babdf832d79f1ccd7b6f28cd4bb1a6af5d5bbd64e88fda905b287c6dc4c8026a"

def get_file_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_verification():
    # 1. Run unit test suite quietly
    test_proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short"],
        capture_output=True,
        text=True
    )
    if test_proc.returncode != 0:
        print("FAIL (pytest failed)")
        return False

    # 2. Re-run baseline on TRAIN split quietly
    backtest_cmd = [
        sys.executable,
        os.path.join("core", "backtester.py"),
        "--data", os.path.join("data", "xauusd_15m_long.csv"),
        "--data-tz", "UTC",
        "--split", "train"
    ]
    bt_proc = subprocess.run(backtest_cmd, capture_output=True, text=True)
    if bt_proc.returncode != 0:
        print("FAIL (backtest simulation error)")
        return False

    # 3. Compare trades.csv SHA256 hash
    trades_path = "trades.csv" if os.path.exists("trades.csv") else os.path.join("research", "meta_labeling", "trades.csv")
    actual_hash = get_file_sha256(trades_path)
    if actual_hash != EXPECTED_TRADES_SHA256:
        print(f"FAIL (hash mismatch: got {actual_hash[:8]}..., expected {EXPECTED_TRADES_SHA256[:8]}...)")
        return False

    print("PASS")
    return True

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
