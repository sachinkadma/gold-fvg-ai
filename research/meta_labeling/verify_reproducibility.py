"""
Script to prove that backtester runs are 100% deterministic and reproducible.
Runs backtest twice on the same dataset and split, compares SHA256 hashes of trades.csv.
"""

import hashlib
import os
import subprocess
import sys

def get_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_reproducibility(split="train", data_path="data/xauusd_15m.csv", data_tz="Europe/Athens"):
    cmd = [
        sys.executable,
        os.path.join("core", "backtester.py"),
        "--split", split,
        "--data", data_path,
        "--data-tz", data_tz
    ]
    
    print(f"=== REPRODUCIBILITY PROOF TEST ===")
    print(f"Dataset: {data_path} | Timezone: {data_tz} | Split: {split}")
    
    # Run 1
    print("\n[Run 1] Executing simulation...")
    res1 = subprocess.run(cmd, capture_output=True, text=True)
    if res1.returncode != 0:
        print("[ERROR] Run 1 failed:", res1.stderr)
        return False
    hash1 = get_file_hash("trades.csv")
    with open("trades.csv", "r") as f:
        lines1 = f.readlines()
    print(f"[Run 1] SHA256(trades.csv) = {hash1} ({len(lines1) - 1} trades)")

    # Run 2
    print("\n[Run 2] Executing identical simulation...")
    res2 = subprocess.run(cmd, capture_output=True, text=True)
    if res2.returncode != 0:
        print("[ERROR] Run 2 failed:", res2.stderr)
        return False
    hash2 = get_file_hash("trades.csv")
    with open("trades.csv", "r") as f:
        lines2 = f.readlines()
    print(f"[Run 2] SHA256(trades.csv) = {hash2} ({len(lines2) - 1} trades)")

    # Compare
    print("\n" + "=" * 65)
    print("HASH VERIFICATION COMPARISON")
    print("=" * 65)
    print(f"Run 1 Hash: {hash1}")
    print(f"Run 2 Hash: {hash2}")
    if hash1 == hash2:
        print("MATCH VERIFIED: True (100% byte-for-byte identical output)")
        return True
    else:
        print("MATCH VERIFIED: False (Output differs!)")
        return False

if __name__ == "__main__":
    test_reproducibility()
