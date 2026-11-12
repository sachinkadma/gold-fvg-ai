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


def _create_chain_with_trades(log_path, trade_r_list, start_date="2026-11-12T00:00:00Z"):
    """Helper to build a strictly valid event hash chain with specified trade net_r values."""
    logger = ForwardSignalLogger(
        forward_start_date=start_date,
        log_path=log_path,
        check_git=False,
        check_frozen_hashes=False,
        min_buffer_warmup=1,
        track_spread=True
    )
    start_dt = pd.Timestamp(start_date)
    for i, r in enumerate(trade_r_list):
        t_trade = start_dt + pd.DateOffset(days=i * 20, hours=10)
        t_str = t_trade.isoformat()
        t_exit = (t_trade + pd.Timedelta(hours=2)).isoformat()
        sig_id = f"SIG-{i+1:04d}"

        logger._append_event("SIGNAL", t_str, {
            "signal_id": sig_id, "signal_num": i + 1, "timestamp_ny": t_str,
            "direction": "BUY" if i % 2 == 0 else "SELL", "entry": 2600.0, "sl": 2580.0, "tp": 2640.0,
            "risk_dist": 20.0, "passed_primary": True, "filter_status": "PASSED_PRIMARY"
        })
        logger._append_event("SPREAD_OBS", t_str, {
            "trigger_event": "SIGNAL", "ref_id": sig_id, "realized_spread_pts": 0.35, "source": "BAR"
        })
        logger._append_event("FILL", t_str, {
            "signal_id": sig_id, "direction": "BUY" if i % 2 == 0 else "SELL",
            "fill_price": 2600.0, "sl": 2580.0, "tp": 2640.0, "planned_risk_dist": 20.0
        })
        logger._append_event("SPREAD_OBS", t_str, {
            "trigger_event": "FILL", "ref_id": sig_id, "realized_spread_pts": 0.35, "source": "BAR"
        })
        logger._append_event("CLOSE", t_exit, {
            "signal_id": sig_id, "direction": "BUY" if i % 2 == 0 else "SELL",
            "entry_price": 2600.0, "exit_price": 2640.0 if r > 0 else 2580.0,
            "exit_reason": "TP" if r > 0 else "SL", "net_r": round(float(r), 3),
            "fill_time_utc": t_str, "exit_time_utc": t_exit
        })
    valid, msg, _ = verify_chain(log_path)
    assert valid is True, f"Chain invalid: {msg}"
    return log_path


def test_gate_boundary_n11_continue(tmp_path):
    log_file = str(tmp_path / "n11.jsonl")
    r_list = [1.9, -1.0] * 5 + [1.9]  # 11 trades
    _create_chain_with_trades(log_file, r_list)
    mon = ForwardMonitor(log_path=log_file)
    status = mon.evaluate(current_time="2027-06-01T00:00:00Z")
    assert status["metrics"]["n_trades"] == 11
    assert status["verdict"] == "CONTINUE"


def test_gate_boundary_n12_fail_fwd(tmp_path):
    log_file = str(tmp_path / "n12_fail.jsonl")
    r_list = [1.0] * 6 + [-1.05] * 6  # 12 trades, mean = -0.025 <= 0
    _create_chain_with_trades(log_file, r_list)
    mon = ForwardMonitor(log_path=log_file)
    status = mon.evaluate(current_time="2027-06-01T00:00:00Z")
    assert status["metrics"]["n_trades"] == 12
    assert status["verdict"] == "FAIL_FWD"
    assert "Early-fail latched at N=12" in status["verdict_detail"]


def test_gate_boundary_n12_continue(tmp_path):
    log_file = str(tmp_path / "n12_cont.jsonl")
    r_list = [1.9] * 6 + [-1.0] * 6  # 12 trades, mean = +0.45 > 0
    _create_chain_with_trades(log_file, r_list)
    mon = ForwardMonitor(log_path=log_file)
    status = mon.evaluate(current_time="2027-06-01T00:00:00Z")
    assert status["metrics"]["n_trades"] == 12
    assert status["verdict"] == "CONTINUE"
    assert "Early-fail avoided at N=12" in status["verdict_detail"]


def test_gate_boundary_n13_negative_after_positive_n12_continues(tmp_path):
    log_file = str(tmp_path / "n13_cont.jsonl")
    # First 12 trades: 6 wins of 1.9, 6 losses of -1.0 (mean = +0.45 > 0)
    # 13th trade: massive loss of -10.0 => cumulative mean becomes negative!
    r_list = [1.9] * 6 + [-1.0] * 6 + [-10.0]
    _create_chain_with_trades(log_file, r_list)
    mon = ForwardMonitor(log_path=log_file)
    status = mon.evaluate(current_time="2027-07-01T00:00:00Z")
    assert status["metrics"]["n_trades"] == 13
    assert status["metrics"]["expectancy_r"] < 0.0  # Cumulative is negative
    assert status["verdict"] == "CONTINUE"  # But verdict remains CONTINUE (never re-evaluated for fail at N=13..19)


def test_gate_boundary_n19_continue(tmp_path):
    log_file = str(tmp_path / "n19.jsonl")
    r_list = [1.9] * 10 + [-1.0] * 9  # 19 trades, positive at N=12
    _create_chain_with_trades(log_file, r_list)
    mon = ForwardMonitor(log_path=log_file)
    status = mon.evaluate(current_time="2027-11-01T00:00:00Z")
    assert status["metrics"]["n_trades"] == 19
    assert status["verdict"] == "CONTINUE"


def test_gate_boundary_n20_pass_fwd(tmp_path):
    log_file = str(tmp_path / "n20_pass.jsonl")
    # 20 trades with high win rate: 15 wins of 1.95, 5 losses of -1.0 (mean = +1.2125R)
    r_list = [1.95] * 15 + [-1.0] * 5
    _create_chain_with_trades(log_file, r_list)
    mon = ForwardMonitor(log_path=log_file)
    status = mon.evaluate(current_time="2028-01-01T00:00:00Z")
    assert status["metrics"]["n_trades"] == 20
    assert status["metrics"]["expectancy_r"] > 0.0
    assert status["metrics"]["ci_95"][0] > 0.0
    assert status["verdict"] == "PASS_FWD"


def test_gate_boundary_n20_ci_lower_nonpositive_inconclusive_final(tmp_path):
    log_file = str(tmp_path / "n20_inconc.jsonl")
    # 20 trades with tiny positive expectancy: 10 wins of 1.02, 10 losses of -1.0 (mean = +0.01R)
    r_list = [1.02] * 10 + [-1.0] * 10
    _create_chain_with_trades(log_file, r_list)
    mon = ForwardMonitor(log_path=log_file)
    status = mon.evaluate(current_time="2028-01-01T00:00:00Z")
    assert status["metrics"]["n_trades"] == 20
    assert status["metrics"]["expectancy_r"] > 0.0
    assert status["metrics"]["ci_95"][0] <= 0.0
    assert status["verdict"] == "INCONCLUSIVE-FINAL"


def test_gate_boundary_n20_exp_nonpositive_fail_fwd(tmp_path):
    log_file = str(tmp_path / "n20_fail.jsonl")
    # First 12: 6 wins of 1.9, 6 losses of -1.0 (mean = +0.45 > 0, passes N=12)
    # Next 8: all losses of -1.0 (sum = 6*1.9 - 14*1.0 = 11.4 - 14 = -2.6 <= 0)
    r_list = [1.9] * 6 + [-1.0] * 6 + [-1.0] * 8
    _create_chain_with_trades(log_file, r_list)
    mon = ForwardMonitor(log_path=log_file)
    status = mon.evaluate(current_time="2028-01-01T00:00:00Z")
    assert status["metrics"]["n_trades"] == 20
    assert status["metrics"]["expectancy_r"] <= 0.0
    assert status["verdict"] == "FAIL_FWD"


def test_gate_boundary_30_months_n15_exp_positive_inconclusive_final(tmp_path):
    log_file = str(tmp_path / "m30_n15.jsonl")
    # 15 trades with positive expectancy, evaluated after 31 months
    r_list = [1.9] * 10 + [-1.0] * 5
    _create_chain_with_trades(log_file, r_list)
    mon = ForwardMonitor(log_path=log_file)
    # 31 months after 2026-11-12 is 2029-06-12
    status = mon.evaluate(current_time="2029-06-15T00:00:00Z")
    assert status["time_limit_reached"] is True
    assert status["metrics"]["n_trades"] == 15
    assert status["verdict"] == "INCONCLUSIVE-FINAL"


def test_gate_boundary_30_months_n8_inconclusive_final(tmp_path):
    log_file = str(tmp_path / "m30_n8.jsonl")
    # 8 trades (< 12) evaluated after 31 months
    r_list = [1.9] * 4 + [-1.0] * 4
    _create_chain_with_trades(log_file, r_list)
    mon = ForwardMonitor(log_path=log_file)
    status = mon.evaluate(current_time="2029-06-15T00:00:00Z")
    assert status["time_limit_reached"] is True
    assert status["metrics"]["n_trades"] == 8
    assert status["verdict"] == "INCONCLUSIVE-FINAL"


def test_gate_boundary_21st_trade_ignored(tmp_path):
    log_file = str(tmp_path / "n21.jsonl")
    # 20 pass trades, plus a 21st extreme trade
    r_list = [1.95] * 15 + [-1.0] * 5 + [-99.0]
    _create_chain_with_trades(log_file, r_list)
    mon = ForwardMonitor(log_path=log_file)
    status = mon.evaluate(current_time="2028-01-01T00:00:00Z")
    # Capped at exactly 20 trades
    assert status["metrics"]["n_trades"] == 20
    assert status["verdict"] == "PASS_FWD"
    # Ensure -99.0 was not included in total_r or expectancy
    assert status["metrics"]["total_r"] > 0.0

