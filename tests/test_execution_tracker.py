"""
Tests for Phase B Execution Quality Tracker (forward/execution_tracker.py).

Verifies:
1. SPREAD_OBS event creation and hash chain validity.
2. Report generation strictly ignores GAP records and non-spread events.
3. Statistical metrics (mean, median, p95, worst) and diff vs assumed 0.35 pts.
4. Correct breakdown by session (London vs NY AM).
5. Source detection: live tick vs bar spread fallback.
"""

import os
import sys
import json
import pytest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from forward.execution_tracker import ExecutionTracker, generate_spread_report, ASSUMED_SPREAD_PTS
from forward.logger import ForwardSignalLogger, verify_chain


def test_spread_obs_event_builder_live_tick_and_bar():
    tracker = ExecutionTracker(assumed_spread_pts=0.35, point_size=0.01)

    # 1. Live tick source
    obs_tick = tracker.build_spread_observation(
        trigger_event="SIGNAL",
        ref_id="SIG-0001",
        timestamp_ny="2026-11-12T03:15:00-05:00",
        session="London",
        tick_bid=2650.10,
        tick_ask=2650.48
    )
    assert obs_tick["trigger_event"] == "SIGNAL"
    assert obs_tick["ref_id"] == "SIG-0001"
    assert obs_tick["session"] == "London"
    assert obs_tick["bid"] == 2650.10
    assert obs_tick["ask"] == 2650.48
    assert abs(obs_tick["realized_spread_pts"] - 0.38) < 1e-5
    assert abs(obs_tick["spread_diff_pts"] - 0.03) < 1e-5
    assert obs_tick["source"] == "live_tick"
    assert obs_tick["units"] == "price_points_usd"

    # 2. Bar spread source (broker points e.g. 40 points -> 0.40 pts)
    obs_bar = tracker.build_spread_observation(
        trigger_event="FILL",
        ref_id="SIG-0001",
        timestamp_ny="2026-11-12T08:30:00-05:00",
        session="NY AM",
        bar_spread_raw=40.0
    )
    assert obs_bar["trigger_event"] == "FILL"
    assert obs_bar["session"] == "NY AM"
    assert obs_bar["bid"] is None
    assert obs_bar["ask"] is None
    assert abs(obs_bar["realized_spread_pts"] - 0.40) < 1e-5
    assert abs(obs_bar["spread_diff_pts"] - 0.05) < 1e-5
    assert obs_bar["source"] == "bar_spread"

    # 3. Default fallback
    obs_def = tracker.build_spread_observation(
        trigger_event="SIGNAL",
        ref_id="SIG-0002",
        timestamp_ny="2026-11-12T04:00:00-05:00",
        session="London"
    )
    assert obs_def["realized_spread_pts"] == 0.35
    assert obs_def["spread_diff_pts"] == 0.0
    assert obs_def["source"] == "assumed_default"


def test_chain_stays_valid_with_spread_obs_events(tmp_path):
    """
    Verifies that ForwardSignalLogger logs SPREAD_OBS events at SIGNAL and FILL,
    and the resulting hash chain remains 100% valid.
    """
    log_file = str(tmp_path / "spread_chain.jsonl")
    logger = ForwardSignalLogger(
        forward_start_date="2026-11-10T00:00:00Z",
        log_path=log_file,
        check_git=False,
        check_frozen_hashes=False,
        track_spread=True,
        min_buffer_warmup=5
    )

    # Feed 10 synthetic bars with spread info
    import pandas as pd
    base = pd.Timestamp("2026-11-12 02:00:00-05:00")  # London KZ
    for i in range(10):
        t = base + pd.Timedelta(minutes=15 * i)
        logger.on_candle_closed({
            "time": t,
            "open": 2600.0 + i,
            "high": 2605.0 + i,
            "low": 2598.0 + i,
            "close": 2602.0 + i,
            "volume": 100.0,
            "spread": 35.0,
            "tick_bid": 2602.0 + i,
            "tick_ask": 2602.35 + i
        })

    valid, msg, count = verify_chain(log_file)
    assert valid is True
    assert "CHAIN_VALID" in msg

    with open(log_file, "r", encoding="utf-8") as f:
        events = [json.loads(l) for l in f if l.strip()]

    # SPREAD_OBS events must be present if any signal occurred
    sig_count = len([e for e in events if e["event"] == "SIGNAL"])
    fill_count = len([e for e in events if e["event"] == "FILL"])
    spread_obs_count = len([e for e in events if e["event"] == "SPREAD_OBS"])
    assert spread_obs_count == (sig_count + fill_count)


def test_spread_report_ignores_gap_and_non_spread_records(tmp_path):
    """
    Strict test: constructs a chain containing GENESIS, GAP (weekend/downtime),
    SIGNAL, SPREAD_OBS, FILL, SPREAD_OBS, and CLOSE records.
    Verifies that generate_spread_report strictly ignores all GAP and non-spread records.
    """
    records = [
        {"index": 0, "event": "GENESIS", "data": {"start": "2026-11-12"}},
        {"index": 1, "event": "GAP", "data": {"gap_type": "WEEKEND_MARKET_CLOSE", "elapsed_seconds": 172800}},
        {"index": 2, "event": "GAP", "data": {"gap_type": "DAILY_MARKET_BREAK", "elapsed_seconds": 3600}},
        {"index": 3, "event": "GAP", "data": {"gap_type": "DOWNTIME_GAP", "elapsed_seconds": 1800}},
        {"index": 4, "event": "SIGNAL", "data": {"signal_id": "SIG-0001", "direction": "BUY"}},
        {
            "index": 5,
            "event": "SPREAD_OBS",
            "data": {
                "trigger_event": "SIGNAL",
                "ref_id": "SIG-0001",
                "session": "London",
                "realized_spread_pts": 0.32,
                "assumed_spread_pts": 0.35,
                "spread_diff_pts": -0.03
            }
        },
        {"index": 6, "event": "FILL", "data": {"signal_id": "SIG-0001", "fill_price": 2650.0}},
        {
            "index": 7,
            "event": "SPREAD_OBS",
            "data": {
                "trigger_event": "FILL",
                "ref_id": "SIG-0001",
                "session": "London",
                "realized_spread_pts": 0.36,
                "assumed_spread_pts": 0.35,
                "spread_diff_pts": 0.01
            }
        },
        {
            "index": 8,
            "event": "SPREAD_OBS",
            "data": {
                "trigger_event": "SIGNAL",
                "ref_id": "SIG-0002",
                "session": "NY AM",
                "realized_spread_pts": 0.42,
                "assumed_spread_pts": 0.35,
                "spread_diff_pts": 0.07
            }
        },
        {"index": 9, "event": "CLOSE", "data": {"signal_id": "SIG-0001", "net_r": 2.0}}
    ]

    tracker = ExecutionTracker(assumed_spread_pts=0.35)
    report = tracker.generate_spread_report(records)

    # 1. Total observations must be exactly 3 (the 3 SPREAD_OBS events, GAP records ignored)
    assert report["total_observations"] == 3
    assert report["by_trigger"]["signal_count"] == 2
    assert report["by_trigger"]["fill_count"] == 1

    # 2. London metrics (0.32 and 0.36)
    lon = report["london"]
    assert lon["count"] == 2
    assert abs(lon["mean"] - 0.34) < 1e-4
    assert abs(lon["median"] - 0.34) < 1e-4
    assert abs(lon["worst"] - 0.36) < 1e-4
    assert abs(lon["diff_vs_assumed"]["mean_diff"] - (-0.01)) < 1e-4

    # 3. NY AM metrics (0.42)
    ny = report["ny_am"]
    assert ny["count"] == 1
    assert abs(ny["mean"] - 0.42) < 1e-4
    assert abs(ny["median"] - 0.42) < 1e-4
    assert abs(ny["worst"] - 0.42) < 1e-4
    assert abs(ny["diff_vs_assumed"]["mean_diff"] - 0.07) < 1e-4

    # 4. Overall metrics (0.32, 0.36, 0.42)
    ovr = report["overall"]
    assert ovr["count"] == 3
    expected_mean = (0.32 + 0.36 + 0.42) / 3.0
    assert abs(ovr["mean"] - round(expected_mean, 4)) < 1e-4
    assert abs(ovr["median"] - 0.36) < 1e-4
    assert abs(ovr["worst"] - 0.42) < 1e-4

    # 5. Format text does not raise and produces structured report
    txt = tracker.format_report_text(report)
    assert "EXECUTION QUALITY REPORT" in txt
    assert "London" in txt
    assert "NY AM" in txt
    assert "Slippage is not measurable" in txt


def test_spread_report_empty_log_handled_cleanly():
    tracker = ExecutionTracker()
    rep = tracker.generate_spread_report([])
    assert rep["total_observations"] == 0
    assert rep["overall"]["count"] == 0
    assert rep["overall"]["mean"] is None
    assert rep["overall"]["worst"] is None
