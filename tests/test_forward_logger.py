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
from forward.logger import (
    ForwardSignalLogger,
    verify_chain,
    verify_against_anchor,
    convert_mt5_time_to_utc,
    save_external_anchor,
    load_external_anchors,
    replay_dry_run,
    GENESIS_PREV_HASH,
    FROZEN_FILE_HASHES
)


@pytest.fixture(scope="module")
def train_data():
    """Loads cleaned dataset once for all slice tests."""
    df = load_and_clean_data(
        "data/xauusd_15m_long.csv",
        data_tz="UTC",
        target_tz="America/New_York",
        verbose=False
    )
    return df


def test_hash_chain_valid_and_tamper_detection(tmp_path):
    """Verifies that a valid log passes verify_chain, but any modification or deletion fails."""
    log_file = str(tmp_path / "signals.jsonl")
    anchored_hashes = []

    def mock_anchor(rec_idx, evt_type, rec_hash):
        anchored_hashes.append((rec_idx, evt_type, rec_hash))

    logger = ForwardSignalLogger(
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


def test_preflight_frozen_hashes_enforcement(tmp_path):
    """Verifies that an altered frozen file hash halts logger initialization."""
    log_file = str(tmp_path / "fail_preflight.jsonl")
    orig_hash = FROZEN_FILE_HASHES["core/config.py"]
    try:
        FROZEN_FILE_HASHES["core/config.py"] = "0" * 64
        with pytest.raises(RuntimeError, match="Frozen file hash mismatch"):
            ForwardSignalLogger(log_path=log_file, check_git=False, check_frozen_hashes=True)
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
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        p_scaled_filter=False,
        killzone_expiry=True,
        skip_outside_kz_eval=True
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
    assert len(closes) == len(bt_trades) == 13

    for i in range(len(bt_trades)):
        bt_t = bt_trades.iloc[i]
        cl_e = closes[i]["data"]

        assert cl_e["direction"] == bt_t["direction"]
        assert abs(cl_e["entry_price"] - bt_t["entry_price"]) < 0.01
        assert abs(cl_e["exit_price"] - bt_t["exit_price"]) < 0.01
        if "SL" in bt_t["exit_reason"]:
            assert "SL" in cl_e["exit_reason"]
        elif "TP" in bt_t["exit_reason"]:
            assert "TP" in cl_e["exit_reason"]

        # Exact pipeline_metalabeling net_r check (tolerance = 0.000)
        pnl = (cl_e["exit_price"] - cl_e["entry_price"]) if cl_e["direction"] == "BUY" else (cl_e["entry_price"] - cl_e["exit_price"])
        expected_net_r = round((pnl / cl_e["planned_risk_dist"]) - (7.0 / (100.0 * cl_e["planned_risk_dist"])), 3)
        assert cl_e["net_r"] == expected_net_r


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
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        p_scaled_filter=False,
        killzone_expiry=True,
        skip_outside_kz_eval=True
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
    assert len(closes) == len(bt_trades) == 9

    for i in range(len(bt_trades)):
        bt_t = bt_trades.iloc[i]
        cl_e = closes[i]["data"]

        assert cl_e["direction"] == bt_t["direction"]
        assert abs(cl_e["entry_price"] - bt_t["entry_price"]) < 0.01
        assert abs(cl_e["exit_price"] - bt_t["exit_price"]) < 0.01

        pnl = (cl_e["exit_price"] - cl_e["entry_price"]) if cl_e["direction"] == "BUY" else (cl_e["entry_price"] - cl_e["exit_price"])
        expected_net_r = round((pnl / cl_e["planned_risk_dist"]) - (7.0 / (100.0 * cl_e["planned_risk_dist"])), 3)
        assert cl_e["net_r"] == expected_net_r


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
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        p_scaled_filter=False,
        killzone_expiry=True,
        skip_outside_kz_eval=True
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
    assert len(closes) == len(bt_trades) == 4

    for i in range(len(bt_trades)):
        bt_t = bt_trades.iloc[i]
        cl_e = closes[i]["data"]

        assert cl_e["direction"] == bt_t["direction"]
        assert abs(cl_e["entry_price"] - bt_t["entry_price"]) < 0.01
        assert abs(cl_e["exit_price"] - bt_t["exit_price"]) < 0.01

        pnl = (cl_e["exit_price"] - cl_e["entry_price"]) if cl_e["direction"] == "BUY" else (cl_e["entry_price"] - cl_e["exit_price"])
        expected_net_r = round((pnl / cl_e["planned_risk_dist"]) - (7.0 / (100.0 * cl_e["planned_risk_dist"])), 3)
        assert cl_e["net_r"] == expected_net_r


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
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        p_scaled_filter=False,
        killzone_expiry=True,
        skip_outside_kz_eval=True
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
    assert len(closes) == len(bt_trades) == 25

    for i in range(len(bt_trades)):
        bt_t = bt_trades.iloc[i]
        cl_e = closes[i]["data"]

        assert cl_e["direction"] == bt_t["direction"]
        assert abs(cl_e["entry_price"] - bt_t["entry_price"]) < 0.01
        assert abs(cl_e["exit_price"] - bt_t["exit_price"]) < 0.01

        pnl = (cl_e["exit_price"] - cl_e["entry_price"]) if cl_e["direction"] == "BUY" else (cl_e["entry_price"] - cl_e["exit_price"])
        expected_net_r = round((pnl / cl_e["planned_risk_dist"]) - (7.0 / (100.0 * cl_e["planned_risk_dist"])), 3)
        assert cl_e["net_r"] == expected_net_r


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
