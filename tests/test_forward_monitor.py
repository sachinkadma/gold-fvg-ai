"""
Unit tests for forward/monitor.py (PREREG Section 5 Monitoring & Gates).

Verifies:
1. Exact equality between monitor.block_bootstrap_ci and run_prereg_test.block_bootstrap_ci.
2. Read-only forward monitor evaluation on a synthetic valid chain (smoke test).
3. Chain integrity check: outputs CHAIN_INVALID and halts on corrupted chain.
4. Telegram alerting graceful failure (never crashes or blocks execution).
5. Observational-only status of directional breakdown block.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from forward.monitor import (
    ForwardMonitor,
    block_bootstrap_ci as monitor_bootstrap,
    send_telegram_alert,
    alert_new_signal,
    alert_trade_closed,
    TARGET_N_FORWARD,
    EARLY_FAIL_N
)
from forward.logger import ForwardSignalLogger, compute_record_hash, verify_chain
from research.meta_labeling.run_prereg_test import block_bootstrap_ci as prereg_bootstrap


def test_bootstrap_ci_exact_equality():
    """
    Verifies that monitor.block_bootstrap_ci is identical in methodology,
    resampling, and output to run_prereg_test.block_bootstrap_ci.
    """
    fixed_r = np.array([
        1.85, -1.02, 1.91, -0.98, -1.05, 1.88,
        -1.01, 1.95, -0.99, 1.80, -1.03, -1.00,
        1.92, -1.01, 1.89
    ])
    months = [
        pd.Period("2026-11", "M"), pd.Period("2026-11", "M"), pd.Period("2026-12", "M"),
        pd.Period("2026-12", "M"), pd.Period("2027-01", "M"), pd.Period("2027-01", "M"),
        pd.Period("2027-02", "M"), pd.Period("2027-02", "M"), pd.Period("2027-03", "M"),
        pd.Period("2027-03", "M"), pd.Period("2027-04", "M"), pd.Period("2027-04", "M"),
        pd.Period("2027-05", "M"), pd.Period("2027-05", "M"), pd.Period("2027-06", "M")
    ]

    # Test with default seed=42
    ci_monitor = monitor_bootstrap(fixed_r, months, n_resamples=2000, seed=42)
    ci_prereg = prereg_bootstrap(fixed_r, months, n_resamples=2000, seed=42)

    assert ci_monitor == ci_prereg
    assert isinstance(ci_monitor, tuple)
    assert len(ci_monitor) == 2
    assert ci_monitor[0] <= ci_monitor[1]

    # Test with different seed and smaller array (< 5 items fallback)
    short_r = np.array([1.5, -1.0, 0.5])
    short_m = months[:3]
    assert monitor_bootstrap(short_r, short_m) == prereg_bootstrap(short_r, short_m)


def test_monitor_smoke_synthetic_chain(tmp_path):
    """
    Smoke test: runs ForwardMonitor on a synthetic chain containing GENESIS,
    SIGNAL, SPREAD_OBS, FILL, CLOSE, and GAP events.
    Verifies metrics, read-only behavior, and console summary formatting.
    """
    log_file = str(tmp_path / "forward_chain.jsonl")
    status_file = str(tmp_path / "forward_status.json")

    # Generate synthetic chain using ForwardSignalLogger
    logger = ForwardSignalLogger(
        forward_start_date="2026-11-12T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        min_buffer_warmup=5,
        track_spread=True
    )

    # 1. Warm-up bars
    base = pd.Timestamp("2026-11-12 02:00:00-05:00")
    for i in range(10):
        t = base + pd.Timedelta(minutes=15 * i)
        logger.on_candle_closed({
            "time": t, "open": 2600.0, "high": 2605.0, "low": 2595.0, "close": 2601.0,
            "volume": 100.0, "spread": 35.0
        })

    # 2. Append completed trade 1 (BUY, WIN)
    ts1 = "2026-11-12T08:00:00+00:00"
    logger._append_event("SIGNAL", ts1, {
        "signal_id": "SIG-0001", "signal_num": 1, "timestamp_ny": "2026-11-12T03:00:00-05:00",
        "direction": "BUY", "entry": 2600.0, "sl": 2580.0, "tp": 2640.0,
        "risk_dist": 20.0, "passed_primary": True, "filter_status": "PASSED_PRIMARY"
    })
    logger._append_event("SPREAD_OBS", ts1, {
        "trigger_event": "SIGNAL", "ref_id": "SIG-0001", "session": "London",
        "realized_spread_pts": 0.35, "source": "BAR"
    })
    logger._append_event("FILL", ts1, {
        "signal_id": "SIG-0001", "direction": "BUY", "fill_price": 2600.0,
        "sl": 2580.0, "tp": 2640.0, "planned_risk_dist": 20.0
    })
    logger._append_event("SPREAD_OBS", ts1, {
        "trigger_event": "FILL", "ref_id": "SIG-0001", "session": "London",
        "realized_spread_pts": 0.35, "source": "BAR"
    })
    logger._append_event("CLOSE", "2026-11-12T10:00:00+00:00", {
        "signal_id": "SIG-0001", "direction": "BUY", "entry_price": 2600.0,
        "exit_price": 2640.0, "exit_reason": "TP", "net_r": 1.95,
        "fill_time_utc": ts1, "exit_time_utc": "2026-11-12T10:00:00+00:00"
    })

    # 3. Gap record (should be ignored by trade extractor)
    logger._append_event("GAP", "2026-11-13T22:00:00+00:00", {
        "gap_type": "WEEKEND_MARKET_CLOSE", "elapsed_seconds": 172800
    })

    # 4. Append completed trade 2 (SELL, LOSS)
    ts2 = "2026-11-16T08:00:00+00:00"
    logger._append_event("SIGNAL", ts2, {
        "signal_id": "SIG-0002", "signal_num": 2, "timestamp_ny": "2026-11-16T03:00:00-05:00",
        "direction": "SELL", "entry": 2610.0, "sl": 2630.0, "tp": 2570.0,
        "risk_dist": 20.0, "passed_primary": True, "filter_status": "PASSED_PRIMARY"
    })
    logger._append_event("FILL", ts2, {
        "signal_id": "SIG-0002", "direction": "SELL", "fill_price": 2610.0,
        "sl": 2630.0, "tp": 2570.0, "planned_risk_dist": 20.0
    })
    logger._append_event("CLOSE", "2026-11-16T12:00:00+00:00", {
        "signal_id": "SIG-0002", "direction": "SELL", "entry_price": 2610.0,
        "exit_price": 2630.0, "exit_reason": "SL", "net_r": -1.02,
        "fill_time_utc": ts2, "exit_time_utc": "2026-11-16T12:00:00+00:00"
    })

    # 5. Non-primary signal (should NOT count toward N)
    logger._append_event("SIGNAL", "2026-11-17T08:00:00+00:00", {
        "signal_id": "SIG-0003", "passed_primary": False, "filter_status": "SKIPPED_OUTSIDE_KILLZONE"
    })

    # Read log file byte size before monitor evaluation (must be unchanged afterwards)
    size_before = os.path.getsize(log_file)

    monitor = ForwardMonitor(log_path=log_file, status_path=status_file)
    status = monitor.evaluate(current_time="2026-11-18T00:00:00Z")

    # Verify read-only: file size unchanged
    assert os.path.getsize(log_file) == size_before

    # Verify chain integrity check passed
    assert status["chain_valid"] is True
    assert status["verdict"] == "CONTINUE"
    assert status["time_limit_reached"] is False

    # Check metrics
    m = status["metrics"]
    assert m["n_trades"] == 2
    assert m["target_n"] == TARGET_N_FORWARD
    assert m["early_fail_n"] == EARLY_FAIL_N
    assert m["win_rate_pct"] == 50.0
    assert abs(m["expectancy_r"] - 0.465) < 1e-4
    assert abs(m["total_r"] - 0.93) < 1e-4
    assert m["profit_factor"] == round(1.95 / 1.02, 2)

    # Check direction breakdown is observation-only
    obs = status["observation_direction_breakdown"]
    assert obs["label"] == "[OBSERVATION ONLY - NOT FOR DECISIONS]"
    assert obs["long"]["n"] == 1
    assert obs["long"]["expectancy_r"] == 1.95
    assert obs["short"]["n"] == 1
    assert obs["short"]["expectancy_r"] == -1.02

    # Status file written
    assert os.path.exists(status_file)
    with open(status_file, "r", encoding="utf-8") as f:
        saved_status = json.load(f)
    assert saved_status["verdict"] == "CONTINUE"

    # Console summary formatting
    summary = monitor.format_console_summary(status)
    assert "PREREG SECTION 5 FORWARD MONITOR - STATUS: CONTINUE" in summary
    assert "[OBSERVATION ONLY - NOT FOR DECISIONS]" in summary


def test_monitor_chain_invalid_handling(tmp_path):
    """
    Verifies that monitor outputs CHAIN_INVALID and halts if the hash chain is tampered.
    """
    log_file = str(tmp_path / "corrupted_chain.jsonl")

    logger = ForwardSignalLogger(
        forward_start_date="2026-11-12T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False
    )
    logger._append_event("SIGNAL", "2026-11-12T08:00:00Z", {"signal_id": "SIG-0001", "passed_primary": True})

    # Tamper with the log file
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    corrupted_rec = json.loads(lines[-1])
    corrupted_rec["data"]["signal_id"] = "TAMPERED"
    lines[-1] = json.dumps(corrupted_rec) + "\n"
    with open(log_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    monitor = ForwardMonitor(log_path=log_file)
    status = monitor.evaluate()

    assert status["chain_valid"] is False
    assert status["verdict"] == "CHAIN_INVALID"
    assert "integrity check failed" in status["status_detail"]
    assert status["n_trades"] == 0


def test_telegram_alert_safe_failure():
    """
    Verifies that send_telegram_alert fails safely and returns False without crashing
    when credentials are dummy or network is unavailable.
    """
    ok = send_telegram_alert("Test alert", token="dummy_invalid_token", chat_id="123456")
    assert ok is False

    # Also test signal and trade closed helpers
    ok_sig = alert_new_signal({"signal_id": "SIG-0001", "direction": "BUY"}, token="dummy", chat_id="123")
    assert ok_sig is False

    ok_trade = alert_trade_closed({"signal_id": "SIG-0001", "net_r": 1.0}, token="dummy", chat_id="123")
    assert ok_trade is False
