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


def test_rehearsal_run_never_pushes_to_forward_anchors(tmp_path, monkeypatch):
    """
    PRE-REHEARSAL CHECK TEST:
    Verifies that a rehearsal run:
    1. Enforces dedicated rehearsal log path (rejects signals.jsonl).
    2. Enforces dedicated anchor branch 'forward-anchors-rehearsal' (rejects forward-anchors).
    3. Git push attempts in rehearsal mode strictly target forward-anchors-rehearsal.
    """
    rehearsal_log = str(tmp_path / "rehearsal.jsonl")
    rehearsal_anchors = str(tmp_path / "rehearsal_anchors.json")

    # 1. Prohibit writing to production signals.jsonl
    prod_log = os.path.join(REPO_ROOT, "forward", "signals.jsonl")
    with pytest.raises(ValueError, match="strictly prohibits writing to production 'forward/signals.jsonl'"):
        ForwardSignalLogger(
            forward_start_date="2026-11-12T00:00:00Z",
            log_path=prod_log,
            rehearsal=True,
            check_git=False,
            check_frozen_hashes=False
        )

    # 2. Prohibit pushing to production forward-anchors branch
    with pytest.raises(ValueError, match="strictly prohibits pushing to production 'forward-anchors' branch"):
        ForwardSignalLogger(
            forward_start_date="2026-11-12T00:00:00Z",
            log_path=rehearsal_log,
            git_branch="forward-anchors",
            rehearsal=True,
            check_git=False,
            check_frozen_hashes=False
        )

    # 3. Clean instantiation defaults to 'forward-anchors-rehearsal'
    logger = ForwardSignalLogger(
        forward_start_date="2026-11-12T00:00:00Z",
        log_path=rehearsal_log,
        external_anchor_path=rehearsal_anchors,
        rehearsal=True,
        check_git=False,
        check_frozen_hashes=False
    )
    assert logger.git_branch == "forward-anchors-rehearsal"

    # 4. Intercept subprocess.run to verify the push target branch
    pushed_branches = []

    def mock_run(cmd, *args, **kwargs):
        class DummyProc:
            returncode = 0
            stdout = "dummy_tree_or_commit_hash\n"
            stderr = ""
        # Capture git push commands
        if isinstance(cmd, list) and "push" in cmd:
            refspec = cmd[-1]
            pushed_branches.append(refspec)
        return DummyProc()

    monkeypatch.setattr("subprocess.run", mock_run)
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "dummy_blob_hash\n")

    # Trigger anchor push via logger
    from forward.logger import push_external_anchor_to_git
    with open(rehearsal_anchors, "w") as f:
        json.dump([{"index": 0, "hash": "abc"}], f)

    push_external_anchor_to_git(rehearsal_anchors, remote="origin", branch=logger.git_branch)

    assert len(pushed_branches) == 1
    assert "forward-anchors-rehearsal" in pushed_branches[0]
    assert "forward-anchors:" not in pushed_branches[0]
    assert pushed_branches[0] == "refs/heads/forward-anchors-rehearsal:refs/heads/forward-anchors-rehearsal"


def test_rehearsal_check_tool(tmp_path):
    """
    Verifies forward/launch/rehearsal_check.py:
    1. Returns exit code 0 on clean rehearsal log with no downtime gaps or duplicates.
    2. Returns exit code 1 on log with unexpected downtime gaps outside market breaks.
    """
    from forward.launch.rehearsal_check import audit_rehearsal_log

    clean_log = str(tmp_path / "clean_rehearsal.jsonl")
    clean_anchors = str(tmp_path / "clean_anchors.json")

    logger = ForwardSignalLogger(
        forward_start_date="2026-11-12T00:00:00Z",
        log_path=clean_log,
        external_anchor_path=clean_anchors,
        rehearsal=True,
        check_git=False,
        check_frozen_hashes=False
    )
    # Add a clean weekend gap
    t_start = "2026-11-13T22:00:00+00:00"
    t_end = "2026-11-15T23:00:00+00:00"
    logger._append_event("GAP", t_end, {
        "gap_type": "WEEKEND_MARKET_CLOSE",
        "gap_start_utc": t_start,
        "gap_end_utc": t_end,
        "elapsed_minutes": 2940,
        "missing_bars": 196
    })

    # Test clean audit
    assert audit_rehearsal_log(clean_log, clean_anchors) is True

    # Now add an unscheduled downtime gap
    dirty_log = str(tmp_path / "dirty_rehearsal.jsonl")
    dirty_anchors = str(tmp_path / "dirty_anchors.json")
    logger_dirty = ForwardSignalLogger(
        forward_start_date="2026-11-12T00:00:00Z",
        log_path=dirty_log,
        external_anchor_path=dirty_anchors,
        rehearsal=True,
        check_git=False,
        check_frozen_hashes=False
    )
    logger_dirty._append_event("GAP", "2026-11-12T14:00:00+00:00", {
        "gap_type": "DOWNTIME_GAP",
        "gap_start_utc": "2026-11-12T12:00:00+00:00",
        "gap_end_utc": "2026-11-12T14:00:00+00:00",
        "elapsed_minutes": 120,
        "missing_bars": 8
    })

    # Test dirty audit
    assert audit_rehearsal_log(dirty_log, dirty_anchors) is False


def test_rehearsal_mode_end_to_end_train_week(tmp_path):
    """
    End-to-end test in rehearsal mode:
    Replays a historical TRAIN week (2015-01-02 to 2015-01-09) containing filled and closed trades.
    Asserts:
    1. Rehearsal mode initializes cleanly with dedicated branch 'forward-anchors-rehearsal'.
    2. Replays candles through ForwardSignalLogger's on_candle_closed handler.
    3. SIGNAL, FILL, SPREAD_OBS (with source='BAR') and CLOSE events are written to the log.
    4. Resulting hash chain is cryptographically valid (verify_chain returns True).
    """
    from core.data_loader import load_and_clean_data
    df_raw = load_and_clean_data(
        "data/xauusd_15m_long.csv",
        data_tz="UTC",
        target_tz="America/New_York",
        verbose=False
    )
    df_week = df_raw.loc["2015-01-02":"2015-01-09"].copy()
    assert len(df_week) > 200

    rehearsal_log = str(tmp_path / "e2e_rehearsal.jsonl")
    rehearsal_anchors = str(tmp_path / "e2e_rehearsal_anchors.json")

    logger = ForwardSignalLogger(
        forward_start_date="2015-01-02T00:00:00Z",
        log_path=rehearsal_log,
        external_anchor_path=rehearsal_anchors,
        rehearsal=True,
        check_git=False,
        check_frozen_hashes=False,
        killzone_gate=True,
        p_scaled_filter=False,
        killzone_expiry=True,
        skip_outside_kz_eval=True,
        track_spread=True,
        min_buffer_warmup=35
    )

    times = df_week.index
    opens = df_week["open"].values
    highs = df_week["high"].values
    lows = df_week["low"].values
    closes = df_week["close"].values
    vols = df_week["volume"].values if "volume" in df_week.columns else [0] * len(df_week)

    for i in range(len(df_week)):
        logger.on_candle_closed({
            "time": times[i],
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "volume": vols[i]
        }, realized_spread=0.35)

    valid, msg, count = verify_chain(rehearsal_log)
    assert valid is True
    assert msg == "CHAIN_VALID"

    with open(rehearsal_log, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    event_types = [e["event"] for e in events]
    assert "GENESIS" in event_types
    assert "SIGNAL" in event_types
    assert "FILL" in event_types
    assert "SPREAD_OBS" in event_types
    assert "CLOSE" in event_types

    spread_obs_events = [e for e in events if e["event"] == "SPREAD_OBS"]
    assert len(spread_obs_events) > 0
    for obs in spread_obs_events:
        assert obs["data"]["source"] == "BAR"

    close_events = [e for e in events if e["event"] == "CLOSE"]
    assert len(close_events) >= 1


