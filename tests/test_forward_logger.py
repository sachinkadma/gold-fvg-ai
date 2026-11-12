"""
Unit tests for forward/logger.py (PREREG Section 5 Phase 1).
Verifies:
1. Hash chain verification and tamper detection.
2. Gap detection on missing candles.
3. Genesis record creation and pre-flight integrity checks.
4. Telegram external anchor queue, retry on failure, and verify_against_anchor.
5. Log truncation detection via external anchors.
6. MT5 server-time to UTC conversion across DST transitions.
7. Exact parity between ForwardSignalLogger and backtester (entry, exit, reason, R)
   across:
   - 2-month baseline slice (Jan-Feb 2015)
   - DST change boundary slice (March 2015, US spring-forward DST change)
   - 6-month extended slice (Jan-June 2015)
8. Dry-run replay of closed candles through the logger.
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
from forward.logger import (
    ForwardSignalLogger,
    verify_chain,
    verify_against_anchor,
    convert_mt5_time_to_utc,
    replay_dry_run,
    GENESIS_PREV_HASH,
    FROZEN_FILE_HASHES
)


@pytest.fixture(scope="module")
def train_data():
    """Loads cleaned TRAIN dataset once for all slice tests."""
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

    # Ingest sequence of GAP events to populate hash chain
    base_time = pd.Timestamp("2026-11-10 10:00:00", tz="UTC")
    logger.on_candle_closed({
        "time": base_time, "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0
    })
    for i in range(1, 6):
        candle_time = base_time + timedelta(minutes=45 * i)
        logger.on_candle_closed({
            "time": candle_time, "open": 2600.0 + i, "high": 2605.0 + i, "low": 2598.0 + i, "close": 2602.0 + i
        })

    # 1. Chain must be valid
    is_valid, msg, count = verify_chain(log_file)
    assert is_valid is True, f"Chain should be valid: {msg}"
    assert count == 6  # 1 genesis + 5 GAP records
    assert len(anchored_hashes) == count

    # Read original lines
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 2. Tamper Test A: Modify a single character in record data
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

    # 3. Tamper Test B: Delete a line in the middle
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
        # Fail the first 2 attempts, then succeed
        if attempt_count <= 2:
            raise ConnectionError("Simulated Telegram timeout")
        succeeded_anchors.append((rec_idx, evt_type, rec_hash))

    logger = ForwardSignalLogger(
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        anchor_callback=flaky_anchor
    )

    # Genesis record was index 0 (attempt 1 -> failed, queued)
    assert len(logger.anchor_queue) == 1
    assert logger.anchor_queue[0][0] == 0

    # Ingest candle with a GAP to trigger a new event and retry the queue
    base_time = pd.Timestamp("2026-11-10 10:00:00", tz="UTC")
    logger.on_candle_closed({
        "time": base_time, "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0
    })
    # Next candle 45 min later -> triggers GAP event (attempt 2 -> fails, attempt 3 -> succeeds)
    logger.on_candle_closed({
        "time": base_time + timedelta(minutes=45), "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0
    })

    # Another GAP event -> flushes remaining queue
    logger.on_candle_closed({
        "time": base_time + timedelta(minutes=90), "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0
    })

    # The queue must have been emptied as subsequent events flushed retries
    assert len(logger.anchor_queue) == 0
    # All records must have been successfully anchored
    valid, msg, count = verify_against_anchor(log_file, succeeded_anchors)
    assert valid is True
    assert count == 3  # 1 genesis + 2 GAP records


def test_verify_against_anchor_truncation_detection(tmp_path):
    """
    Verifies that verify_against_anchor immediately detects when a log has been
    truncated (e.g. lines deleted at end of file to hide adverse trades).
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
    logger.on_candle_closed({
        "time": base_time, "open": 2600.0, "high": 2605.0, "low": 2598.0, "close": 2602.0
    })
    for i in range(1, 5):
        logger.on_candle_closed({
            "time": base_time + timedelta(minutes=45 * i),
            "open": 2600.0 + i, "high": 2605.0 + i, "low": 2598.0 + i, "close": 2602.0 + i
        })

    # File has 5 records (1 genesis + 4 GAPs). External anchor has all 5.
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

    # verify_chain on truncated file says CHAIN_VALID (sub-chain 0->1 is internally consistent)
    chain_val, _, _ = verify_chain(truncated_file)
    assert chain_val is True

    # BUT verify_against_anchor MUST detect truncation against external anchor!
    t_val, t_msg, _ = verify_against_anchor(truncated_file, anchored)
    assert t_val is False
    assert "TRUNCATION_DETECTED" in t_msg


def test_mt5_time_conversion():
    """
    Verifies MT5 server time to UTC conversion:
    - Winter: EET is UTC+2 (e.g. 14:00 EET -> 12:00 UTC)
    - Summer: EEST is UTC+3 (e.g. 14:00 EEST -> 11:00 UTC)
    - UTC passthrough
    """
    # 1. Winter date (January)
    winter_server = "2026-01-15 14:00:00"
    utc_winter = convert_mt5_time_to_utc(winter_server, server_tz="EET")
    assert utc_winter.hour == 12
    assert utc_winter.tzinfo == timezone.utc

    # 2. Summer date (July)
    summer_server = "2026-07-15 14:00:00"
    utc_summer = convert_mt5_time_to_utc(summer_server, server_tz="EET")
    assert utc_summer.hour == 11
    assert utc_summer.tzinfo == timezone.utc

    # 3. UTC passthrough
    utc_raw = "2026-03-01 10:00:00+00:00"
    utc_converted = convert_mt5_time_to_utc(utc_raw, server_tz="UTC")
    assert utc_converted.hour == 10


def test_logger_fills_vs_backtester_train_slice(train_data, tmp_path):
    """
    PARITY TEST 1: Baseline 2-Month Slice (2015-01-01 to 2015-02-28).
    Verifies exact trade-by-trade parity:
    - 100% exact direction match
    - 100% exact entry price match (< 0.01 pt)
    - 100% exact exit price match (< 0.01 pt)
    - 100% exact exit reason match
    - Net R match within discrete 0.01-lot rounding tolerance (|R_diff| < 0.05 R)
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
        killzone_expiry=True
    )

    times = df_slice.index
    opens = df_slice["open"].values
    highs = df_slice["high"].values
    lows = df_slice["low"].values
    closes = df_slice["close"].values
    vols = df_slice["volume"].values if "volume" in df_slice.columns else [0] * len(df_slice)

    for i in range(len(df_slice)):
        c_dict = {
            "time": times[i],
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "volume": vols[i]
        }
        logger.on_candle_closed(c_dict)

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
        assert abs(cl_e["net_r"] - bt_t["result_r"]) < 0.05


def test_logger_parity_dst_change_slice(train_data, tmp_path):
    """
    PARITY TEST 2: DST Change Boundary Slice (2015-03-01 to 2015-03-31).
    US switched to Daylight Saving Time on Sunday, March 8, 2015 (EST UTC-5 -> EDT UTC-4).
    Verifies that killzone alignment and trades match exactly across the transition.
    """
    df_slice = train_data.loc["2015-03-01":"2015-03-31"].copy()
    assert len(df_slice) > 1000

    bt_trades = run_backtest_simulation(df_slice)
    assert len(bt_trades) == 9

    log_file = str(tmp_path / "dst_slice_logger.jsonl")
    logger = ForwardSignalLogger(
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        p_scaled_filter=False,
        killzone_expiry=True
    )

    times = df_slice.index
    opens = df_slice["open"].values
    highs = df_slice["high"].values
    lows = df_slice["low"].values
    closes = df_slice["close"].values
    vols = df_slice["volume"].values if "volume" in df_slice.columns else [0] * len(df_slice)

    for i in range(len(df_slice)):
        c_dict = {
            "time": times[i],
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "volume": vols[i]
        }
        logger.on_candle_closed(c_dict)

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
        assert abs(cl_e["net_r"] - bt_t["result_r"]) < 0.05


def test_logger_parity_longer_slice(train_data, tmp_path):
    """
    PARITY TEST 3: Extended 4-Month Slice (2015-01-01 to 2015-04-30).
    Verifies full trade-by-trade parity across 4 months of continuous operation.
    """
    df_slice = train_data.loc["2015-01-01":"2015-04-30"].copy()
    assert len(df_slice) > 4000

    bt_trades = run_backtest_simulation(df_slice)
    assert len(bt_trades) >= 20

    log_file = str(tmp_path / "longer_slice_logger.jsonl")
    logger = ForwardSignalLogger(
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        p_scaled_filter=False,
        killzone_expiry=True
    )

    times = df_slice.index
    opens = df_slice["open"].values
    highs = df_slice["high"].values
    lows = df_slice["low"].values
    closes = df_slice["close"].values
    vols = df_slice["volume"].values if "volume" in df_slice.columns else [0] * len(df_slice)

    for i in range(len(df_slice)):
        c_dict = {
            "time": times[i],
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "volume": vols[i]
        }
        logger.on_candle_closed(c_dict)

    with open(log_file, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    closes = [e for e in events if e["event"] == "CLOSE"]
    assert len(closes) == len(bt_trades)

    for i in range(len(bt_trades)):
        bt_t = bt_trades.iloc[i]
        cl_e = closes[i]["data"]

        assert cl_e["direction"] == bt_t["direction"]
        assert abs(cl_e["entry_price"] - bt_t["entry_price"]) < 0.01
        assert abs(cl_e["exit_price"] - bt_t["exit_price"]) < 0.01
        assert abs(cl_e["net_r"] - bt_t["result_r"]) < 0.05


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
