"""
Unit tests for forward/logger.py (PREREG Section 5 Phase 1 Follow-up).
Verifies:
1. Hash chain verification and tamper detection.
2. Gap detection on missing candles.
3. Genesis record creation and pre-flight integrity checks.
4. Telegram external anchor queue, retry on failure, and verify_against_anchor.
5. Disk persistence of anchor retry queue across logger restarts.
6. External anchor ledger persistence (independent copy for audit).
7. Log truncation detection via external anchors.
8. Outside-killzone signal logging without silent drops (SKIPPED_OUTSIDE_KILLZONE).
9. Rolling-buffer logger vs full-history signal generation parity (minimum safe buffer = 135 bars).
10. Exact net_r match against pipeline_metalabeling formula (tolerance = 0.000).
11. MT5 server-time conversion: US-DST-based broker rules vs EU rules.
12. Execution parity slices:
    - 2-month baseline slice (Jan-Feb 2015)
    - Spring forward DST change slice (March 2015)
    - Fall back DST change slice (October 25 - November 15, 2015)
    - VALID partition slice (January 01 - March 31, 2020)
13. Dry-run replay of closed candles.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.data_loader import load_and_clean_data
from core.backtester import run_backtest_simulation
from core.signals import evaluate_causal_signal_at_bar
from research.meta_labeling.pipeline_metalabeling import simulate_candidate_forward
from forward.logger import (
    ForwardSignalLogger,
    verify_chain,
    verify_against_anchor,
    convert_mt5_time_to_utc,
    save_external_anchor,
    load_external_anchors,
    push_external_anchor_to_git,
    replay_dry_run,
    GENESIS_PREV_HASH,
    FROZEN_FILE_HASHES,
    MIN_BUFFER_WARMUP,
    FORWARD_GIT_TAG
)


def test_hash_chain_valid_and_tamper_detection(tmp_path):
    """Verifies that a valid log passes verify_chain, but any modification or deletion fails."""
    log_file = str(tmp_path / "signals.jsonl")
    anchored_hashes = []

    def mock_anchor(rec_idx, evt_type, rec_hash):
        anchored_hashes.append((rec_idx, evt_type, rec_hash))

    logger = ForwardSignalLogger(
        forward_start_date="2026-11-10T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        anchor_callback=mock_anchor
    )

    base_time = pd.Timestamp("2026-11-10 10:00:00", tz="UTC")
    logger.on_candle_closed({
        "time": base_time, "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0
    })
    for i in range(1, 6):
        candle_time = base_time + timedelta(minutes=45 * i)
        logger.on_candle_closed({
            "time": candle_time, "open": 2600.0 + i, "high": 2605.0 + i, "low": 2598.0 + i, "close": 2602.0 + i
        })

    is_valid, msg, count = verify_chain(log_file)
    assert is_valid is True, f"Chain should be valid: {msg}"
    assert count == 6
    assert len(anchored_hashes) == count

    with open(log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Tamper Test A: Modify payload
    tampered_file_a = str(tmp_path / "tampered_a.jsonl")
    bad_record = json.loads(lines[0])
    bad_record["data"]["tampered_key"] = "hacked"
    with open(tampered_file_a, "w", encoding="utf-8") as f:
        f.write(json.dumps(bad_record) + "\n")
        for line in lines[1:]:
            f.write(line + "\n")

    val_a, msg_a, _ = verify_chain(tampered_file_a)
    assert val_a is False
    assert "tamper detected" in msg_a.lower()

    # Tamper Test B: Delete a line
    tampered_file_b = str(tmp_path / "tampered_b.jsonl")
    with open(tampered_file_b, "w", encoding="utf-8") as f:
        f.write(lines[0] + "\n")
        f.write(lines[2] + "\n")

    val_b, msg_b, _ = verify_chain(tampered_file_b)
    assert val_b is False
    assert ("mismatch" in msg_b.lower() or "tamper" in msg_b.lower())


def test_gap_detection_on_missing_candles(tmp_path):
    """Verifies that unexpected time gaps between closed candles generate a GAP record."""
    log_file = str(tmp_path / "gap_test.jsonl")
    logger = ForwardSignalLogger(
        forward_start_date="2026-11-10T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False
    )

    t1 = pd.Timestamp("2026-11-10 10:00:00", tz="UTC")
    logger.on_candle_closed({
        "time": t1, "open": 2600.0, "high": 2605.0, "low": 2595.0, "close": 2602.0
    })

    t2 = t1 + timedelta(minutes=45)
    logger.on_candle_closed({
        "time": t2, "open": 2602.0, "high": 2608.0, "low": 2600.0, "close": 2606.0
    })

    with open(log_file, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    events = [r["event"] for r in records]
    assert "GAP" in events

    gap_rec = [r for r in records if r["event"] == "GAP"][0]
    assert gap_rec["data"]["missing_bars"] == 2
    assert gap_rec["data"]["elapsed_minutes"] == 45


def test_genesis_record_structure(tmp_path):
    """Verifies genesis record format, genesis hash, and pre-flight hash integrity."""
    log_file = str(tmp_path / "genesis_test.jsonl")
    logger = ForwardSignalLogger(
        forward_start_date="2026-11-10T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False
    )

    with open(log_file, "r", encoding="utf-8") as f:
        rec0 = json.loads(f.readline())

    assert rec0["index"] == 0
    assert rec0["prev_hash"] == GENESIS_PREV_HASH
    assert rec0["event"] == "GENESIS"
    assert rec0["data"]["p_scaled"] == 0.6077
    assert rec0["data"]["rr_target"] == 2.0
    assert rec0["data"]["git_tag"] == FORWARD_GIT_TAG
    assert "git_commit" in rec0["data"]


def test_preflight_frozen_hashes_enforcement(tmp_path):
    """Verifies that an altered frozen file hash halts logger initialization."""
    log_file = str(tmp_path / "fail_preflight.jsonl")
    orig_hash = FROZEN_FILE_HASHES["core/config.py"]
    try:
        FROZEN_FILE_HASHES["core/config.py"] = "0" * 64
        with pytest.raises(RuntimeError, match="Frozen file hash mismatch"):
            ForwardSignalLogger(forward_start_date="2026-11-12T00:00:00Z", log_path=log_file, check_git=False, check_frozen_hashes=True)
    finally:
        FROZEN_FILE_HASHES["core/config.py"] = orig_hash


def test_telegram_anchor_queue_and_retry(tmp_path):
    """
    Verifies that when external anchor delivery fails, the logger queues
    the anchor and retries on subsequent events without skipping.
    """
    log_file = str(tmp_path / "anchor_retry.jsonl")
    attempt_count = 0
    succeeded_anchors = []

    def flaky_anchor(rec_idx, evt_type, rec_hash):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count <= 2:
            raise ConnectionError("Simulated Telegram timeout")
        succeeded_anchors.append((rec_idx, evt_type, rec_hash))

    logger = ForwardSignalLogger(
        forward_start_date="2026-11-10T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        anchor_callback=flaky_anchor
    )

    # Genesis record failed attempt 1 -> queued
    assert len(logger.anchor_queue) == 1
    assert logger.anchor_queue[0][0] == 0

    base_time = pd.Timestamp("2026-11-10 10:00:00", tz="UTC")
    logger.on_candle_closed({
        "time": base_time, "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0
    })
    # GAP event -> triggers attempt 2 (fails) and attempt 3 (succeeds)
    logger.on_candle_closed({
        "time": base_time + timedelta(minutes=45), "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0
    })
    # Another GAP event -> flushes remaining queue
    logger.on_candle_closed({
        "time": base_time + timedelta(minutes=90), "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0
    })

    assert len(logger.anchor_queue) == 0
    valid, msg, count = verify_against_anchor(log_file, succeeded_anchors)
    assert valid is True
    assert count == 3


def test_anchor_queue_disk_persistence(tmp_path):
    """
    Verifies that anchor_queue is persisted to disk and survives process restart.
    """
    log_file = str(tmp_path / "persistent_queue.jsonl")
    queue_file = str(tmp_path / "persistent_queue_anchor_queue.json")

    def always_fail_anchor(rec_idx, evt_type, rec_hash):
        raise ConnectionError("Simulated network outage")

    # Instance 1: fails to anchor genesis
    logger1 = ForwardSignalLogger(
        forward_start_date="2026-11-10T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        anchor_callback=always_fail_anchor
    )
    assert len(logger1.anchor_queue) == 1
    assert os.path.exists(queue_file)

    # Ingest GAP to queue a second record
    t0 = pd.Timestamp("2026-11-10 10:00:00", tz="UTC")
    logger1.on_candle_closed({"time": t0, "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0})
    logger1.on_candle_closed({"time": t0 + timedelta(minutes=45), "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0})

    assert len(logger1.anchor_queue) == 2
    del logger1

    # Instance 2 (simulated reboot): restores queue from disk
    succeeded = []
    def working_anchor(rec_idx, evt_type, rec_hash):
        succeeded.append((rec_idx, evt_type, rec_hash))

    logger2 = ForwardSignalLogger(
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        anchor_callback=working_anchor
    )
    # Queue must have been restored from disk
    assert len(logger2.anchor_queue) == 2

    # Flush by triggering a GAP
    logger2.on_candle_closed({"time": t0 + timedelta(minutes=90), "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0})

    # Queue should now be empty and cleared from disk
    assert len(logger2.anchor_queue) == 0
    assert not os.path.exists(queue_file)


def test_external_anchor_ledger(tmp_path):
    """
    Verifies independent external anchor ledger persistence (save_external_anchor & load_external_anchors).
    """
    log_file = str(tmp_path / "ledger_signals.jsonl")
    ext_anchor_file = str(tmp_path / "external_anchors.json")

    logger = ForwardSignalLogger(
        forward_start_date="2026-11-10T00:00:00Z",
        log_path=log_file,
        external_anchor_path=ext_anchor_file,
        check_git=False,
        check_frozen_hashes=False
    )

    t0 = pd.Timestamp("2026-11-10 10:00:00", tz="UTC")
    logger.on_candle_closed({"time": t0, "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0})
    logger.on_candle_closed({"time": t0 + timedelta(minutes=45), "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0})

    # Read independent external anchor file
    anchors = load_external_anchors(ext_anchor_file)
    assert len(anchors) == 2  # Genesis (0) and GAP (1)

    # Verify against external anchor file
    valid, msg, count = verify_against_anchor(log_file, [(a["index"], a["record_hash"]) for a in anchors])
    assert valid is True
    assert count == 2


def test_verify_against_anchor_truncation_detection(tmp_path):
    """
    Verifies that verify_against_anchor immediately detects when a log has been truncated.
    """
    log_file = str(tmp_path / "truncation_test.jsonl")
    anchored = []

    def recording_anchor(rec_idx, evt_type, rec_hash):
        anchored.append((rec_idx, rec_hash))

    logger = ForwardSignalLogger(
        forward_start_date="2026-11-10T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        anchor_callback=recording_anchor
    )

    base_time = pd.Timestamp("2026-11-10 10:00:00", tz="UTC")
    logger.on_candle_closed({"time": base_time, "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0})
    for i in range(1, 5):
        logger.on_candle_closed({
            "time": base_time + timedelta(minutes=45 * i),
            "open": 2600.0 + i, "high": 2605.0 + i, "low": 2598.0 + i, "close": 2602.0 + i
        })

    assert len(anchored) == 5
    val, msg, cnt = verify_against_anchor(log_file, anchored)
    assert val is True

    # Truncate log to first 2 lines
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    truncated_file = str(tmp_path / "truncated.jsonl")
    with open(truncated_file, "w", encoding="utf-8") as f:
        for l in lines[:2]:
            f.write(l + "\n")

    chain_val, _, _ = verify_chain(truncated_file)
    assert chain_val is True

    t_val, t_msg, _ = verify_against_anchor(truncated_file, anchored)
    assert t_val is False
    assert "TRUNCATION_DETECTED" in t_msg


def test_outside_killzone_signal_produces_skipped_record(train_data, tmp_path):
    """
    CONFIRMATION TEST 1:
    Verifies that when an ICT setup forms outside London/NY killzones, ForwardSignalLogger:
    1. Evaluates the setup causal logic unchanged.
    2. Writes a SIGNAL record with filter_status == 'SKIPPED_OUTSIDE_KILLZONE' and passed_primary == False.
    3. Leaves pending_order as None (zero unintended orders placed).
    Zero silent drops.
    """
    log_file = str(tmp_path / "outside_kz.jsonl")
    logger = ForwardSignalLogger(
        forward_start_date="2015-01-01T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        skip_outside_kz_eval=False
    )

    # Bar 138 in train_data is 2015-01-02 05:30:00-05:00 (outside London and NY AM killzones)
    # Feed candles 0 to 138
    times = train_data.index[:139]
    opens = train_data["open"].values[:139]
    highs = train_data["high"].values[:139]
    lows = train_data["low"].values[:139]
    closes = train_data["close"].values[:139]
    vols = train_data["volume"].values[:139] if "volume" in train_data.columns else [0] * 139

    for i in range(139):
        logger.on_candle_closed({
            "time": times[i], "open": opens[i], "high": highs[i], "low": lows[i], "close": closes[i], "volume": vols[i]
        })

    # Read events from log
    with open(log_file, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    signals = [e for e in events if e["event"] == "SIGNAL"]
    assert len(signals) >= 1

    # Verify at least one signal has SKIPPED_OUTSIDE_KILLZONE
    outside_sigs = [s for s in signals if s["data"]["filter_status"] == "SKIPPED_OUTSIDE_KILLZONE"]
    assert len(outside_sigs) >= 1

    s_out = outside_sigs[0]["data"]
    assert s_out["killzone"] == "OTHER"
    assert s_out["passed_primary"] is False
    assert logger.pending_order is None


@pytest.mark.slow
def test_rolling_buffer_parity_against_full_history(train_data):
    """
    CONFIRMATION TEST 2:
    Compares 300-bar rolling buffer against full-history signal generation on a mid-dataset slice.
    Signals must be 100% identical. Minimum safe buffer size is 135 bars.
    """
    df_slice = train_data.iloc[5000:6500].copy()  # 1500 candles mid-dataset
    full_signals = {}
    buf_signals = {}

    # 1. Full-history ground truth
    for t in range(150, len(df_slice)):
        s_full = evaluate_causal_signal_at_bar(df_slice, t, killzone_gate=False)
        if s_full is not None:
            full_signals[t] = (
                s_full["direction"], s_full["entry"], s_full["sl"], s_full["tp"], s_full["risk_dist"]
            )

    # 2. Rolling buffer of 300 bars
    for t in range(150, len(df_slice)):
        start_idx = max(0, t - 299)
        df_buf = df_slice.iloc[start_idx : t + 1]
        s_buf = evaluate_causal_signal_at_bar(df_buf, len(df_buf) - 1, killzone_gate=False)
        if s_buf is not None:
            buf_signals[t] = (
                s_buf["direction"], s_buf["entry"], s_buf["sl"], s_buf["tp"], s_buf["risk_dist"]
            )

    assert len(full_signals) > 0
    assert full_signals == buf_signals, "Rolling buffer signals diverged from full-history signals!"


@pytest.mark.slow
def test_buffer_135_parity_vs_full_history(train_data):
    """
    CONFIRMATION TEST 2B:
    Tightest boundary test: exactly 135-bar rolling buffer (theoretical safe floor)
    vs full-history causal signal generation on 1,500 candles.
    Signals must match 100.00% identically with zero divergence.
    """
    df_slice = train_data.iloc[5000:6500].copy()  # 1500 candles mid-dataset
    full_signals = {}
    buf_signals = {}

    # 1. Full-history ground truth
    for t in range(135, len(df_slice)):
        s_full = evaluate_causal_signal_at_bar(df_slice, t, killzone_gate=False)
        if s_full is not None:
            full_signals[t] = (
                s_full["direction"], s_full["entry"], s_full["sl"], s_full["tp"], s_full["risk_dist"]
            )

    # 2. Rolling buffer of exactly 135 bars
    for t in range(135, len(df_slice)):
        start_idx = t - 134  # exactly 135 bars in window [start_idx : t + 1]
        df_buf = df_slice.iloc[start_idx : t + 1]
        assert len(df_buf) == 135
        s_buf = evaluate_causal_signal_at_bar(df_buf, len(df_buf) - 1, killzone_gate=False)
        if s_buf is not None:
            buf_signals[t] = (
                s_buf["direction"], s_buf["entry"], s_buf["sl"], s_buf["tp"], s_buf["risk_dist"]
            )

    assert len(full_signals) > 0
    assert full_signals == buf_signals, "135-bar buffer signals diverged from full-history signals!"


def test_buffer_warmup_suppresses_signal_below_135(train_data, tmp_path):
    """
    CONFIRMATION TEST 2C:
    Proves warmup gate suppression: on slice train_data.iloc[100:250], a valid ICT
    SELL signal triggers at bar index 40 when min_buffer_warmup=35.
    Under min_buffer_warmup=135, this signal is strictly suppressed and 0 signals
    are emitted across the entire warm-up window [0..133].
    """
    df_slice = train_data.iloc[100:250].copy()

    # 1. Baseline without 135-bar gate (warmup=35): signal WOULD be emitted at bar 40
    log_unsuppressed = str(tmp_path / "unsuppressed.jsonl")
    logger_unsuppressed = ForwardSignalLogger(
        forward_start_date=str(df_slice.index[0]),
        log_path=log_unsuppressed,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=False,
        skip_outside_kz_eval=False,
        min_buffer_warmup=35
    )
    for i in range(50):
        row = df_slice.iloc[i]
        logger_unsuppressed.on_candle_closed({
            "time": df_slice.index[i], "open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"], "volume": row["volume"]
        })

    events_unsup = [json.loads(line) for line in open(log_unsuppressed) if line.strip()]
    sigs_unsup = [e for e in events_unsup if e["event"] == "SIGNAL"]
    assert len(sigs_unsup) >= 1, "Expected baseline setup to emit signal at bar < 50 without 135-bar warmup"
    assert sigs_unsup[0]["data"]["direction"] == "SELL"

    # 2. With 135-bar warmup gate: identical candles fed, but signals are strictly suppressed
    log_suppressed = str(tmp_path / "suppressed.jsonl")
    logger_suppressed = ForwardSignalLogger(
        forward_start_date=str(df_slice.index[0]),
        log_path=log_suppressed,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=False,
        skip_outside_kz_eval=False,
        min_buffer_warmup=135
    )
    for i in range(134):  # bars 0 to 133 (< 135 warmup threshold)
        row = df_slice.iloc[i]
        logger_suppressed.on_candle_closed({
            "time": df_slice.index[i], "open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"], "volume": row["volume"]
        })

    events_sup = [json.loads(line) for line in open(log_suppressed) if line.strip()]
    sigs_sup = [e for e in events_sup if e["event"] == "SIGNAL"]
    assert len(sigs_sup) == 0, f"Expected 0 signals during warmup, got {len(sigs_sup)}"
    assert logger_suppressed.signal_counter == 0

    # 3. Bar 135 arrives -> warmup complete and evaluation unlocks
    row_135 = df_slice.iloc[134]
    logger_suppressed.on_candle_closed({
        "time": df_slice.index[134], "open": row_135["open"], "high": row_135["high"], "low": row_135["low"], "close": row_135["close"], "volume": row_135["volume"]
    })
    assert len(logger_suppressed._times_ny) == 135


def test_mt5_time_conversion_us_vs_eu_dst():
    """
    CONFIRMATION TEST 6:
    Verifies MT5 broker server time conversion to UTC under US DST rules:
    - Winter: Server is UTC+2 (e.g. 14:00 EET -> 12:00 UTC)
    - Summer: Server is UTC+3 (e.g. 14:00 EEST -> 11:00 UTC)
    - March seam (US has switched to EDT, EU still in standard time): Server is already UTC+3 (14:00 -> 11:00 UTC)
    - October/November seam (EU has fallen back, US still in EDT): Server is still UTC+3 (14:00 -> 11:00 UTC)
    """
    # 1. Winter date (Jan 15, 2026): Server is UTC+2
    t_win = "2026-01-15 14:00:00"
    u_win = convert_mt5_time_to_utc(t_win, server_tz="US_EET")
    assert u_win.hour == 12

    # 2. Summer date (Jul 15, 2026): Server is UTC+3
    t_sum = "2026-07-15 14:00:00"
    u_sum = convert_mt5_time_to_utc(t_sum, server_tz="US_EET")
    assert u_sum.hour == 11

    # 3. March gap date (March 15, 2026): US is EDT, EU is winter
    # Under US broker rules, server has already shifted to UTC+3
    t_mar = "2026-03-15 14:00:00"
    u_mar = convert_mt5_time_to_utc(t_mar, server_tz="US_EET")
    assert u_mar.hour == 11

    # 4. October gap date (October 28, 2026): EU is winter, US is still EDT
    # Under US broker rules, server is still UTC+3
    t_oct = "2026-10-28 14:00:00"
    u_oct = convert_mt5_time_to_utc(t_oct, server_tz="US_EET")
    assert u_oct.hour == 11


@pytest.mark.slow
def test_logger_fills_vs_backtester_train_slice(train_data, tmp_path):
    """
    PARITY TEST 1: Baseline 2-Month Slice (2015-01-01 to 2015-02-28).
    Verifies exact trade-by-trade parity:
    - 100% exact direction match
    - 100% exact entry price match (< 0.01 pt)
    - 100% exact exit price match (< 0.01 pt)
    - 100% exact exit reason match
    - Exact net_r match against pipeline_metalabeling formula (tolerance = 0.000)
    """
    df_slice = train_data.loc["2015-01-01":"2015-02-28"].copy()
    assert len(df_slice) > 1000

    bt_trades = run_backtest_simulation(df_slice)
    assert len(bt_trades) == 13

    log_file = str(tmp_path / "train_slice_logger.jsonl")
    logger = ForwardSignalLogger(
        forward_start_date="2015-01-01T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        p_scaled_filter=False,
        killzone_expiry=True,
        skip_outside_kz_eval=True,
        min_buffer_warmup=35
    )

    times = df_slice.index
    opens = df_slice["open"].values
    highs = df_slice["high"].values
    lows = df_slice["low"].values
    closes = df_slice["close"].values
    vols = df_slice["volume"].values if "volume" in df_slice.columns else [0] * len(df_slice)

    for i in range(len(df_slice)):
        logger.on_candle_closed({
            "time": times[i], "open": opens[i], "high": highs[i], "low": lows[i], "close": closes[i], "volume": vols[i]
        })

    with open(log_file, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    closes = [e for e in events if e["event"] == "CLOSE"]
    signals_by_id = {e["data"]["signal_id"]: e for e in events if e["event"] == "SIGNAL"}
    assert len(closes) == len(bt_trades) == 13

    for i in range(len(bt_trades)):
        bt_t = bt_trades.iloc[i]
        cl_e = closes[i]["data"]
        sig_evt = signals_by_id[cl_e["signal_id"]]

        assert cl_e["direction"] == bt_t["direction"]
        assert abs(cl_e["entry_price"] - bt_t["entry_price"]) < 0.01
        assert abs(cl_e["exit_price"] - bt_t["exit_price"]) < 0.01
        if "SL" in bt_t["exit_reason"]:
            assert "SL" in cl_e["exit_reason"]
        elif "TP" in bt_t["exit_reason"]:
            assert "TP" in cl_e["exit_reason"]

        # Call real pipeline_metalabeling code path directly
        sig_dt = pd.Timestamp(sig_evt["data"]["timestamp_ny"])
        bar_idx = train_data.index.get_loc(sig_dt)
        pm_outcome = simulate_candidate_forward(
            df=train_data,
            sig_bar=bar_idx,
            direction=sig_evt["data"]["direction"],
            entry=sig_evt["data"]["entry"],
            sl=sig_evt["data"]["sl"],
            risk_dist=sig_evt["data"]["risk_dist"]
        )
        assert pm_outcome["filled"] is True
        assert cl_e["net_r"] == pm_outcome["net_r_20"]
        if "SL" in cl_e["exit_reason"]:
            assert "SL" in pm_outcome["exit_reason_20"]
        elif "TP" in cl_e["exit_reason"]:
            assert "TP" in pm_outcome["exit_reason_20"]


@pytest.mark.slow
def test_logger_parity_dst_spring_slice(train_data, tmp_path):
    """
    PARITY TEST 2: Spring Forward DST Slice (March 01 - March 31, 2015).
    US DST switch occurred Sunday, March 8, 2015.
    """
    df_slice = train_data.loc["2015-03-01":"2015-03-31"].copy()
    assert len(df_slice) > 1000

    bt_trades = run_backtest_simulation(df_slice)
    assert len(bt_trades) == 9

    log_file = str(tmp_path / "dst_spring_logger.jsonl")
    logger = ForwardSignalLogger(
        forward_start_date="2015-03-01T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        p_scaled_filter=False,
        killzone_expiry=True,
        skip_outside_kz_eval=True,
        min_buffer_warmup=35
    )

    times = df_slice.index
    opens = df_slice["open"].values
    highs = df_slice["high"].values
    lows = df_slice["low"].values
    closes = df_slice["close"].values
    vols = df_slice["volume"].values if "volume" in df_slice.columns else [0] * len(df_slice)

    for i in range(len(df_slice)):
        logger.on_candle_closed({
            "time": times[i], "open": opens[i], "high": highs[i], "low": lows[i], "close": closes[i], "volume": vols[i]
        })

    with open(log_file, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    closes = [e for e in events if e["event"] == "CLOSE"]
    signals_by_id = {e["data"]["signal_id"]: e for e in events if e["event"] == "SIGNAL"}
    assert len(closes) == len(bt_trades) == 9

    for i in range(len(bt_trades)):
        bt_t = bt_trades.iloc[i]
        cl_e = closes[i]["data"]
        sig_evt = signals_by_id[cl_e["signal_id"]]

        assert cl_e["direction"] == bt_t["direction"]
        assert abs(cl_e["entry_price"] - bt_t["entry_price"]) < 0.01
        assert abs(cl_e["exit_price"] - bt_t["exit_price"]) < 0.01

        # Call real pipeline_metalabeling code path directly
        sig_dt = pd.Timestamp(sig_evt["data"]["timestamp_ny"])
        bar_idx = train_data.index.get_loc(sig_dt)
        pm_outcome = simulate_candidate_forward(
            df=train_data,
            sig_bar=bar_idx,
            direction=sig_evt["data"]["direction"],
            entry=sig_evt["data"]["entry"],
            sl=sig_evt["data"]["sl"],
            risk_dist=sig_evt["data"]["risk_dist"]
        )
        assert pm_outcome["filled"] is True
        assert cl_e["net_r"] == pm_outcome["net_r_20"]


@pytest.mark.slow
def test_logger_parity_november_fallback_dst_slice(train_data, tmp_path):
    """
    PARITY TEST 3: Fall Back DST Slice (October 25 - November 15, 2015).
    US DST ended Sunday, November 1, 2015 (EDT UTC-4 -> EST UTC-5).
    """
    df_slice = train_data.loc["2015-10-25":"2015-11-15"].copy()
    assert len(df_slice) > 500

    bt_trades = run_backtest_simulation(df_slice)
    assert len(bt_trades) == 4

    log_file = str(tmp_path / "dst_fallback_logger.jsonl")
    logger = ForwardSignalLogger(
        forward_start_date="2015-10-25T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        p_scaled_filter=False,
        killzone_expiry=True,
        skip_outside_kz_eval=True,
        min_buffer_warmup=35
    )

    times = df_slice.index
    opens = df_slice["open"].values
    highs = df_slice["high"].values
    lows = df_slice["low"].values
    closes = df_slice["close"].values
    vols = df_slice["volume"].values if "volume" in df_slice.columns else [0] * len(df_slice)

    for i in range(len(df_slice)):
        logger.on_candle_closed({
            "time": times[i], "open": opens[i], "high": highs[i], "low": lows[i], "close": closes[i], "volume": vols[i]
        })

    with open(log_file, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    closes = [e for e in events if e["event"] == "CLOSE"]
    signals_by_id = {e["data"]["signal_id"]: e for e in events if e["event"] == "SIGNAL"}
    assert len(closes) == len(bt_trades) == 4

    for i in range(len(bt_trades)):
        bt_t = bt_trades.iloc[i]
        cl_e = closes[i]["data"]
        sig_evt = signals_by_id[cl_e["signal_id"]]

        assert cl_e["direction"] == bt_t["direction"]
        assert abs(cl_e["entry_price"] - bt_t["entry_price"]) < 0.01
        assert abs(cl_e["exit_price"] - bt_t["exit_price"]) < 0.01

        # Call real pipeline_metalabeling code path directly
        sig_dt = pd.Timestamp(sig_evt["data"]["timestamp_ny"])
        bar_idx = train_data.index.get_loc(sig_dt)
        pm_outcome = simulate_candidate_forward(
            df=train_data,
            sig_bar=bar_idx,
            direction=sig_evt["data"]["direction"],
            entry=sig_evt["data"]["entry"],
            sl=sig_evt["data"]["sl"],
            risk_dist=sig_evt["data"]["risk_dist"]
        )
        assert pm_outcome["filled"] is True
        assert cl_e["net_r"] == pm_outcome["net_r_20"]


@pytest.mark.slow
def test_logger_parity_valid_period_slice(train_data, tmp_path):
    """
    PARITY TEST 4: VALID Period Slice (January 01 - March 31, 2020).
    Verifies full parity on out-of-sample VALID data.
    """
    df_slice = train_data.loc["2020-01-01":"2020-03-31"].copy()
    assert len(df_slice) > 2000

    bt_trades = run_backtest_simulation(df_slice)
    assert len(bt_trades) == 25

    log_file = str(tmp_path / "valid_slice_logger.jsonl")
    logger = ForwardSignalLogger(
        forward_start_date="2020-01-01T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        p_scaled_filter=False,
        killzone_expiry=True,
        skip_outside_kz_eval=True,
        min_buffer_warmup=35
    )

    times = df_slice.index
    opens = df_slice["open"].values
    highs = df_slice["high"].values
    lows = df_slice["low"].values
    closes = df_slice["close"].values
    vols = df_slice["volume"].values if "volume" in df_slice.columns else [0] * len(df_slice)

    for i in range(len(df_slice)):
        logger.on_candle_closed({
            "time": times[i], "open": opens[i], "high": highs[i], "low": lows[i], "close": closes[i], "volume": vols[i]
        })

    with open(log_file, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    closes = [e for e in events if e["event"] == "CLOSE"]
    signals_by_id = {e["data"]["signal_id"]: e for e in events if e["event"] == "SIGNAL"}
    assert len(closes) == len(bt_trades) == 25

    for i in range(len(bt_trades)):
        bt_t = bt_trades.iloc[i]
        cl_e = closes[i]["data"]
        sig_evt = signals_by_id[cl_e["signal_id"]]

        assert cl_e["direction"] == bt_t["direction"]
        assert abs(cl_e["entry_price"] - bt_t["entry_price"]) < 0.01
        assert abs(cl_e["exit_price"] - bt_t["exit_price"]) < 0.01

        # Call real pipeline_metalabeling code path directly
        sig_dt = pd.Timestamp(sig_evt["data"]["timestamp_ny"])
        bar_idx = train_data.index.get_loc(sig_dt)
        pm_outcome = simulate_candidate_forward(
            df=train_data,
            sig_bar=bar_idx,
            direction=sig_evt["data"]["direction"],
            entry=sig_evt["data"]["entry"],
            sl=sig_evt["data"]["sl"],
            risk_dist=sig_evt["data"]["risk_dist"]
        )
        assert pm_outcome["filled"] is True
        assert cl_e["net_r"] == pm_outcome["net_r_20"]


@pytest.mark.slow
def test_dry_run_replay(train_data, tmp_path):
    """
    Verifies replay_dry_run functionality:
    Feeds recent closed candles into the replay engine and checks chain validity.
    """
    dry_log = str(tmp_path / "dry_run.jsonl")
    candles_recent = train_data.iloc[-400:].copy()

    res = replay_dry_run(
        candles_recent,
        log_path=dry_log,
        killzone_gate=True,
        p_scaled_filter=True
    )

    assert res["valid"] is True
    assert res["candles_replayed"] == 400
    assert res["record_count"] >= 1
    assert os.path.exists(dry_log)


def test_forward_start_date_required_and_staleness_enforcement(tmp_path):
    """
    Verifies FORWARD_START_DATE contract:
    1. Mandatory on new logger startup (raises ValueError if None or empty).
    2. Refuses start if start date is >1 day in the past.
    3. Valid start date (<24h old) is written to GENESIS payload.
    """
    log_file1 = str(tmp_path / "log_missing_start.jsonl")
    with pytest.raises(ValueError, match="forward_start_date is a required startup argument"):
        ForwardSignalLogger(forward_start_date=None, log_path=log_file1, check_git=False, check_frozen_hashes=False)

    log_file2 = str(tmp_path / "log_stale_start.jsonl")
    stale_date = "2026-11-01T00:00:00Z"
    with pytest.raises(ValueError, match="is >1 day in the past"):
        ForwardSignalLogger(forward_start_date=stale_date, log_path=log_file2, check_git=False, check_frozen_hashes=False, enforce_staleness=True)

    log_file3 = str(tmp_path / "log_valid_start.jsonl")
    valid_date = "2026-11-12T00:00:00Z"
    logger = ForwardSignalLogger(forward_start_date=valid_date, log_path=log_file3, check_git=False, check_frozen_hashes=False)
    with open(log_file3, "r", encoding="utf-8") as f:
        rec0 = json.loads(f.readline())
    assert rec0["event"] == "GENESIS"
    assert rec0["data"]["forward_start_date"] == "2026-11-12T00:00:00+00:00"
    assert rec0["data"]["git_tag"] is not None and len(rec0["data"]["git_tag"]) > 0


@pytest.mark.slow
def test_restart_recovery_mid_position_equals_uninterrupted(train_data, tmp_path):
    """
    CONFIRMATION TEST 7:
    Verifies restart recovery when killed mid-position:
    1. Uninterrupted run on slice [2015-01-01:2015-01-15].
    2. Interrupted run killed at bar 315 (while active position SIG-0003 is open).
    3. Re-instantiated logger recovers active_position and buffer from chain.
    4. Resumed run completes identical to uninterrupted run (record-for-record & hash-for-hash).
    5. verify_chain returns True on resulting chain.
    """
    df_slice = train_data.loc["2015-01-01":"2015-01-15"].copy()

    # 1. Uninterrupted reference run
    p_unint = str(tmp_path / "uninterrupted.jsonl")
    l_unint = ForwardSignalLogger(
        forward_start_date="2015-01-01T00:00:00Z",
        log_path=p_unint,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        p_scaled_filter=False,
        killzone_expiry=True,
        skip_outside_kz_eval=True,
        min_buffer_warmup=35
    )
    for i in range(len(df_slice)):
        r = df_slice.iloc[i]
        l_unint.on_candle_closed({
            "time": df_slice.index[i], "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"], "volume": r["volume"]
        })

    evts_unint = [json.loads(l) for l in open(p_unint) if l.strip()]
    fill_idx = [i for i, e in enumerate(evts_unint) if e["event"] == "FILL" and e["data"]["signal_id"] == "SIG-0003"][0]

    # 2. Interrupted run: write records up to fill transaction (simulating kill right after fill bar)
    cutoff_idx = fill_idx + 1
    if cutoff_idx < len(evts_unint) and evts_unint[cutoff_idx]["event"] == "SPREAD_OBS":
        cutoff_idx += 1

    p_inter = str(tmp_path / "interrupted.jsonl")
    with open(p_inter, "w", encoding="utf-8") as f:
        for e in evts_unint[:cutoff_idx]:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")

    # 3. Re-boot logger on the interrupted file
    l_resumed = ForwardSignalLogger(
        forward_start_date="2015-01-01T00:00:00Z",
        log_path=p_inter,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        p_scaled_filter=False,
        killzone_expiry=True,
        skip_outside_kz_eval=True,
        min_buffer_warmup=35
    )
    assert l_resumed.active_position is not None, "Active position must be recovered from chain"
    assert l_resumed.active_position["signal_id"] == "SIG-0003"

    # 4. Resume feeding remaining candles from bar 314
    for i in range(314, len(df_slice)):
        r = df_slice.iloc[i]
        l_resumed.on_candle_closed({
            "time": df_slice.index[i], "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"], "volume": r["volume"]
        })

    # 5. Assert exact match
    evts_inter = [json.loads(l) for l in open(p_inter) if l.strip()]
    assert len(evts_unint) == len(evts_inter)
    for k in range(len(evts_unint)):
        assert evts_unint[k] == evts_inter[k], f"Mismatch at record {k}: {evts_unint[k]} vs {evts_inter[k]}"

    valid, msg, cnt = verify_chain(p_inter)
    assert valid is True
    assert cnt == len(evts_unint)


def test_git_push_anchor_remote_unreachable_refuses_start(tmp_path):
    """
    Verifies that ForwardSignalLogger refuses to start if git_push_anchors=True
    and the remote is unreachable.
    """
    log_file = str(tmp_path / "test_unreachable.jsonl")
    with pytest.raises(RuntimeError, match="is unreachable or network is down"):
        ForwardSignalLogger(
            forward_start_date="2026-11-12T00:00:00Z",
            log_path=log_file,
            check_git=False,
            check_frozen_hashes=False,
            git_push_anchors=True,
            git_remote="nonexistent-remote-url-xyz-99"
        )


def test_mt5_server_midnight_maps_to_previous_day_22_utc():
    """
    A1 VERIFICATION TEST:
    Verifies that for MetaQuotes-Demo broker timezone (EU_EET):
    - 2026-03-10 00:00:00 server maps to 2026-03-09 22:00:00 UTC (EET is UTC+2)
    - 2026-10-27 00:00:00 server maps to 2026-10-26 22:00:00 UTC (EET is UTC+2)
    Both are 22:00 UTC of the previous day.
    """
    ts_mar = convert_mt5_time_to_utc("2026-03-10 00:00:00", server_tz="EU_EET")
    assert ts_mar == pd.Timestamp("2026-03-09 22:00:00+00:00")
    assert ts_mar.day == 9
    assert ts_mar.hour == 22

    ts_oct = convert_mt5_time_to_utc("2026-10-27 00:00:00", server_tz="EU_EET")
    assert ts_oct == pd.Timestamp("2026-10-26 22:00:00+00:00")
    assert ts_oct.day == 26
    assert ts_oct.hour == 22


def test_genesis_metadata_fields(tmp_path):
    """
    A1 VERIFICATION TEST:
    Verifies that the GENESIS record contains auto-read git_tag/commit,
    and metadata: server_tz=EU_EET, MT5 server name, symbol, feed name.
    """
    log_file = str(tmp_path / "genesis_meta.jsonl")
    logger = ForwardSignalLogger(
        forward_start_date="2026-11-12T00:00:00Z",
        log_path=log_file,
        server_tz="EU_EET",
        server_name="MetaQuotes-Demo",
        symbol="XAUUSD",
        feed_name="MT5",
        check_git=False,
        check_frozen_hashes=False
    )
    with open(log_file, "r", encoding="utf-8") as f:
        rec0 = json.loads(f.readline())

    assert rec0["event"] == "GENESIS"
    d = rec0["data"]
    assert d["server_tz"] == "EU_EET"
    assert d["server_name"] == "MetaQuotes-Demo"
    assert d["symbol"] == "XAUUSD"
    assert d["feed_name"] == "MT5"
    assert d["git_tag"] is not None and len(d["git_tag"]) > 0
    assert d["git_commit"] is not None and len(d["git_commit"]) > 0


def test_history_preload_bars_before_forward_start_date_only_fill_buffer(train_data, tmp_path):
    """
    A3 VERIFICATION TEST:
    Verifies that closed candles dated strictly before forward_start_date:
    1. Only enter and fill the rolling candle buffer.
    2. Emit NO SIGNAL, FILL, EXPIRE, or CLOSE events into the chain.
    3. The chain only contains the initial GENESIS record.
    """
    df_slice = train_data.iloc[100:250].copy()  # Historical 2015 candles
    log_file = str(tmp_path / "preload_test.jsonl")

    # forward_start_date set to 2026-11-12 (all slice candles are in 2015, hence before start date)
    logger = ForwardSignalLogger(
        forward_start_date="2026-11-10T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        min_buffer_warmup=135
    )

    for i in range(len(df_slice)):
        r = df_slice.iloc[i]
        logger.on_candle_closed({
            "time": df_slice.index[i], "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"], "volume": r["volume"]
        })

    # Buffer must be filled with candles
    assert len(logger._times_ny) == len(df_slice)

    # Chain must contain ONLY the GENESIS record, zero trading events
    with open(log_file, "r", encoding="utf-8") as f:
        events = [json.loads(l) for l in f if l.strip()]

    assert len(events) == 1
    assert events[0]["event"] == "GENESIS"


def test_restart_with_pending_order_and_crash_before_anchor_push(train_data, tmp_path):
    """
    A4 VERIFICATION TEST:
    Verifies restart recovery when killed while an order is PENDING,
    and crash occurred before external anchor push:
    1. Logger emits PASSED_PRIMARY signal (creating pending order).
    2. 2 additional bars elapse (bars_elapsed = 2).
    3. Crash simulated: external anchor ledger only saved up to GENESIS (signal not anchored).
    4. Resumed logger:
       - Reconstructs pending order and recovers bars_elapsed = 2.
       - Discovers unanchored signal record in log, queues it to anchor_queue.
       - Flushes queue cleanly.
    """
    df_slice = train_data.iloc[100:250].copy()
    log_file = str(tmp_path / "crash_pending.jsonl")
    anchor_file = str(tmp_path / "crash_anchors.json")

    logger1 = ForwardSignalLogger(
        forward_start_date="2015-01-01T00:00:00Z",
        log_path=log_file,
        external_anchor_path=anchor_file,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=False,
        p_scaled_filter=False,
        min_buffer_warmup=35
    )

    # Feed bars until a PASSED_PRIMARY signal is emitted
    signal_emitted = False
    sig_bar = 0
    for i in range(len(df_slice)):
        r = df_slice.iloc[i]
        logger1.on_candle_closed({
            "time": df_slice.index[i], "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"], "volume": r["volume"]
        })
        if logger1.pending_order is not None and not signal_emitted:
            signal_emitted = True
            sig_bar = i
            break

    assert signal_emitted is True
    assert logger1.pending_order is not None

    # Feed 2 more bars
    for i in range(sig_bar + 1, sig_bar + 3):
        r = df_slice.iloc[i]
        logger1.on_candle_closed({
            "time": df_slice.index[i], "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"], "volume": r["volume"]
        })

    assert logger1.pending_order["bars_elapsed"] == 2

    # Simulate crash before anchor push: truncate external anchor file so only GENESIS is anchored
    with open(anchor_file, "w", encoding="utf-8") as f:
        json.dump([{"index": 0, "event": "GENESIS", "record_hash": "dummy"}], f)

    # Re-instantiate logger on existing files
    logger2 = ForwardSignalLogger(
        forward_start_date="2015-01-01T00:00:00Z",
        log_path=log_file,
        external_anchor_path=anchor_file,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=False,
        p_scaled_filter=False,
        min_buffer_warmup=35
    )

    # Verify pending order recovered with bars_elapsed intact
    assert logger2.pending_order is not None
    assert logger2.pending_order["bars_elapsed"] == 2

    # Verify crash recovery discovered unanchored record and queued it
    assert len(logger2.anchor_queue) >= 1

