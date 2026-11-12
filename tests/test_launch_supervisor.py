"""
Tests for Phase L Launch Infrastructure:
1. Calendar 30-month computation (no 913.125-day approximation).
2. XAUUSD market schedule and stale feed alert suppression during weekend close and daily break.
3. Supervisor startup refusal handling and retry capping.
"""

import os
import sys
import json
import pytest
from datetime import datetime, timezone
import zoneinfo
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from forward.launch.heartbeat import is_xauusd_market_open, ForwardHeartbeatDaemon
from forward.monitor import ForwardMonitor
from forward.logger import ForwardSignalLogger, verify_chain

NY_TZ = zoneinfo.ZoneInfo("America/New_York")


def test_calendar_30m_computation_and_monitor_boundary(tmp_path):
    """
    Verifies that the 30-month end date is computed as exact calendar months,
    stored in the GENESIS record, and used by ForwardMonitor to trigger time limit.
    """
    start_date_str = "2026-11-12T00:00:00Z"
    log_file = str(tmp_path / "calendar_30m.jsonl")

    logger = ForwardSignalLogger(
        forward_start_date=start_date_str,
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False
    )

    with open(log_file, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())

    genesis_data = first_line["data"]
    expected_30m_end = pd.Timestamp("2029-05-12T00:00:00Z")
    assert "forward_end_date_30m" in genesis_data
    assert pd.Timestamp(genesis_data["forward_end_date_30m"]) == expected_30m_end

    mon = ForwardMonitor(log_path=log_file)

    # 1 day before 30 calendar months: time limit NOT reached
    st_before = mon.evaluate(current_time="2029-05-11T00:00:00Z")
    assert st_before["time_limit_reached"] is False
    assert st_before["elapsed_months"] < 30.0

    # 1 second before 30 calendar months: time limit NOT reached
    st_sec_before = mon.evaluate(current_time="2029-05-11T23:59:59Z")
    assert st_sec_before["time_limit_reached"] is False

    # At 30 calendar months: time limit reached
    st_at = mon.evaluate(current_time="2029-05-12T00:00:00Z")
    assert st_at["time_limit_reached"] is True
    assert st_at["elapsed_months"] == 30.0
    assert st_at["verdict"] == "INCONCLUSIVE-FINAL"


def test_calendar_30m_leap_and_month_end_handling():
    """Verifies calendar month arithmetic handles month-end days cleanly."""
    ts1 = pd.Timestamp("2026-08-31T00:00:00Z")
    end1 = ts1 + pd.DateOffset(months=30)
    # Aug 31 + 30 months = Feb 28 2029
    assert end1 == pd.Timestamp("2029-02-28T00:00:00Z")

    ts2 = pd.Timestamp("2026-01-31T00:00:00Z")
    end2 = ts2 + pd.DateOffset(months=30)
    # Jan 31 + 30 months = July 31 2028
    assert end2 == pd.Timestamp("2028-07-31T00:00:00Z")


def test_xauusd_market_schedule_checks():
    """Verifies XAUUSD market open/close detection in America/New_York clock time."""
    # 1. Friday before close (16:45 NY) -> OPEN
    t_fri_open = datetime(2026, 11, 13, 16, 45, tzinfo=NY_TZ)
    assert is_xauusd_market_open(t_fri_open) is True

    # 2. Friday after close (17:05 NY) -> CLOSED
    t_fri_closed = datetime(2026, 11, 13, 17, 5, tzinfo=NY_TZ)
    assert is_xauusd_market_open(t_fri_closed) is False

    # 3. Saturday midday (12:00 NY) -> CLOSED
    t_sat = datetime(2026, 11, 14, 12, 0, tzinfo=NY_TZ)
    assert is_xauusd_market_open(t_sat) is False

    # 4. Sunday afternoon (15:00 NY) -> CLOSED
    t_sun_closed = datetime(2026, 11, 15, 15, 0, tzinfo=NY_TZ)
    assert is_xauusd_market_open(t_sun_closed) is False

    # 5. Sunday open (18:05 NY) -> OPEN
    t_sun_open = datetime(2026, 11, 15, 18, 5, tzinfo=NY_TZ)
    assert is_xauusd_market_open(t_sun_open) is True

    # 6. Tuesday trading session (04:00 NY London KZ) -> OPEN
    t_tue_open = datetime(2026, 11, 17, 4, 0, tzinfo=NY_TZ)
    assert is_xauusd_market_open(t_tue_open) is True

    # 7. Tuesday daily maintenance break (17:30 NY) -> CLOSED
    t_tue_break = datetime(2026, 11, 17, 17, 30, tzinfo=NY_TZ)
    assert is_xauusd_market_open(t_tue_break) is False


def test_heartbeat_stale_feed_rule_suppressed_on_weekend(tmp_path):
    """
    Verifies that STALE_FEED alerts are strictly suppressed during the weekend market close
    and daily breaks, but trigger when market is open and candles stop arriving.
    """
    log_file = str(tmp_path / "heartbeat_test.jsonl")

    # Create chain with last candle on Friday 16:45 NY (21:45 UTC)
    t_fri_last_candle = "2026-11-13T21:45:00+00:00"
    logger = ForwardSignalLogger(
        forward_start_date="2026-11-12T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False
    )
    logger._append_event("SIGNAL", t_fri_last_candle, {
        "signal_id": "SIG-0001", "passed_primary": True
    })

    daemon = ForwardHeartbeatDaemon(
        log_path=log_file,
        stale_threshold_seconds=45 * 60  # 45 minutes
    )

    # Check 1: Saturday 12:00 NY (market closed). Last candle was 19 hours ago!
    # MUST NOT trigger stale feed alert.
    t_sat_check = datetime(2026, 11, 14, 17, 0, tzinfo=timezone.utc)
    res_sat = daemon.run_check_cycle(now_utc=t_sat_check)
    assert res_sat["market_open"] is False
    assert res_sat["stale_alert_triggered"] is False

    # Check 2: Tuesday 17:30 NY (daily break, 22:30 UTC).
    # MUST NOT trigger stale feed alert.
    t_tue_break = datetime(2026, 11, 17, 22, 30, tzinfo=timezone.utc)
    res_break = daemon.run_check_cycle(now_utc=t_tue_break)
    assert res_break["market_open"] is False
    assert res_break["stale_alert_triggered"] is False

    # Check 3: Tuesday 04:00 NY (market actively open, 09:00 UTC).
    # Last candle was Friday, so gap is ~3.5 days during open market hours!
    # MUST trigger stale feed alert.
    t_tue_active = datetime(2026, 11, 17, 9, 0, tzinfo=timezone.utc)
    res_active = daemon.run_check_cycle(now_utc=t_tue_active)
    assert res_active["market_open"] is True
    assert res_active["stale_alert_triggered"] is True
