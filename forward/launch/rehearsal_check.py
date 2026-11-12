"""
Rehearsal Isolation & Audit Check Tool (Read-Only).
Verifies:
1. Hash chain cryptographic integrity via verify_chain().
2. External anchor ledger integrity via verify_against_anchor().
3. Event type counts.
4. SIGNAL counts by filter_status.
5. GAP records breakdown (WEEKEND_MARKET_CLOSE vs DAILY_MARKET_BREAK vs DOWNTIME_GAP) with durations.
6. Candle timestamp uniqueness (no duplicate candle timestamps).
7. Missing 15m candles outside scheduled market breaks.

Returns Exit Code 0 ONLY if all checks are completely clean.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from forward.logger import verify_chain, verify_against_anchor


def audit_rehearsal_log(log_path, anchor_ledger):
    print("=" * 70)
    print("FORWARD REHEARSAL VERIFICATION AUDIT")
    print("=" * 70)
    print(f"Log Path       : {log_path}")
    print(f"Anchor Ledger  : {anchor_ledger}")
    print("-" * 70)

    clean = True

    # 1. Existence check
    if not os.path.exists(log_path):
        print(f"[ERROR] Rehearsal log file not found: {log_path}")
        return False

    # 2. verify_chain()
    chain_valid, chain_msg, rec_count = verify_chain(log_path)
    print(f"[1] CHAIN INTEGRITY       : {'PASS' if chain_valid else 'FAIL'} ({chain_msg}) | Records: {rec_count}")
    if not chain_valid:
        clean = False

    # 3. verify_against_anchor()
    if anchor_ledger and os.path.exists(anchor_ledger):
        anc_res = verify_against_anchor(log_path, anchor_ledger)
        if isinstance(anc_res, (tuple, list)) and len(anc_res) == 3:
            anc_valid, anc_msg, anc_count = anc_res
        elif isinstance(anc_res, (tuple, list)) and len(anc_res) == 2:
            anc_valid, anc_msg = anc_res
        else:
            anc_valid, anc_msg = False, str(anc_res)

        print(f"[2] ANCHOR INTEGRITY      : {'PASS' if anc_valid else 'FAIL'} ({anc_msg})")
        if not anc_valid:
            clean = False
    else:
        print(f"[2] ANCHOR INTEGRITY      : WARNING - Anchor ledger not found at {anchor_ledger}")
        anc_valid = False
        clean = False

    # Read events
    events = []
    with open(log_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_str = line.strip()
            if line_str:
                events.append(json.loads(line_str))

    # 4. Counts of each event type
    event_counts = {}
    for e in events:
        evt = e.get("event", "UNKNOWN")
        event_counts[evt] = event_counts.get(evt, 0) + 1
    print(f"[3] EVENT COUNTS          : {event_counts}")

    # 5. SIGNAL counts by filter_status
    signals = [e for e in events if e.get("event") == "SIGNAL"]
    signal_status_counts = {}
    signal_candle_times = []
    for s in signals:
        d = s.get("data", {})
        st = d.get("filter_status", "UNKNOWN")
        signal_status_counts[st] = signal_status_counts.get(st, 0) + 1
        ts_ny = d.get("timestamp_ny")
        if ts_ny:
            signal_candle_times.append(ts_ny)
    print(f"[4] SIGNALS BY STATUS     : {signal_status_counts} (Total signals: {len(signals)})")

    # 6. GAP records breakdown
    gaps = [e for e in events if e.get("event") == "GAP"]
    gap_summary = {"WEEKEND_MARKET_CLOSE": 0, "DAILY_MARKET_BREAK": 0, "DOWNTIME_GAP": 0}
    downtime_gaps = []
    for g in gaps:
        d = g.get("data", {})
        gt = d.get("gap_type", "UNKNOWN")
        gap_summary[gt] = gap_summary.get(gt, 0) + 1
        if gt == "DOWNTIME_GAP":
            downtime_gaps.append(d)

    print(f"[5] GAP BREAKDOWN         : Weekend: {gap_summary.get('WEEKEND_MARKET_CLOSE', 0)} | "
          f"Daily Break: {gap_summary.get('DAILY_MARKET_BREAK', 0)} | "
          f"Downtime (Unexpected): {gap_summary.get('DOWNTIME_GAP', 0)}")

    if gaps:
        print("    Gap details:")
        for idx, g in enumerate(gaps, 1):
            d = g.get("data", {})
            gt = d.get("gap_type", "UNKNOWN")
            dur = d.get("elapsed_minutes", 0)
            st = d.get("gap_start_utc", "N/A")
            et = d.get("gap_end_utc", "N/A")
            mb = d.get("missing_bars", 0)
            print(f"      - Gap #{idx} [{gt}]: {st} -> {et} ({dur}m / {dur/60.0:.1f}h, {mb} bars)")

    # 7. Check for duplicate candle timestamps
    duplicate_times = []
    seen_times = set()
    for t in signal_candle_times:
        if t in seen_times:
            duplicate_times.append(t)
        seen_times.add(t)

    # Also check buffer snapshot if present
    base_no_ext, _ = os.path.splitext(log_path)
    buf_path = f"{base_no_ext}_buffer.json"
    if os.path.exists(buf_path):
        try:
            with open(buf_path, "r", encoding="utf-8") as bf:
                buf_data = json.load(bf)
                buf_times = buf_data.get("times_ny", [])
                if len(buf_times) != len(set(buf_times)):
                    duplicate_times.append("DUPLICATE_FOUND_IN_BUFFER_SNAPSHOT")
        except Exception:
            pass

    if duplicate_times:
        print(f"[6] DUPLICATE CANDLES     : FAIL - Found duplicates: {duplicate_times}")
        clean = False
    else:
        print("[6] DUPLICATE CANDLES     : PASS - None (all candle timestamps unique)")

    # 8. Check missing 15m candles outside market breaks
    if downtime_gaps:
        print(f"[7] MISSING CANDLES       : FAIL - Found {len(downtime_gaps)} downtime gap(s) outside market breaks!")
        for dg in downtime_gaps:
            print(f"      * Downtime: {dg.get('gap_start_utc')} -> {dg.get('gap_end_utc')} "
                  f"({dg.get('elapsed_minutes')} min, {dg.get('missing_bars')} bars missing)")
        clean = False
    else:
        print("[7] MISSING CANDLES       : PASS - None (all gaps belong to scheduled weekend or daily breaks)")

    print("-" * 70)
    if clean:
        print("OVERALL AUDIT VERDICT: PASS (All checks clean, exit 0)")
        print("=" * 70)
        return True
    else:
        print("OVERALL AUDIT VERDICT: FAIL (Discrepancies detected, exit 1)")
        print("=" * 70)
        return False


def main():
    parser = argparse.ArgumentParser(description="Audit and verify a forward rehearsal or production run.")
    parser.add_argument("--log", default="forward/rehearsal.jsonl", help="Path to JSONL event log")
    parser.add_argument("--anchor-ledger", default="forward/rehearsal_anchors.json", help="Path to external anchor ledger")
    args = parser.parse_args()

    success = audit_rehearsal_log(args.log, args.anchor_ledger)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
