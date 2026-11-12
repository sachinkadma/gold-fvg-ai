"""
Live / Paper Forward Execution Orchestrator for ICT/SMC Gold Strategy.
Polls MetaTrader 5 on closed 15m candles, enforces duplicate deduplication,
detects downtime gaps, automatically recovers state (pending orders, open positions,
and buffer) from the append-only hash chain on restart, and anchors head hashes to git.
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
from datetime import datetime, timezone, timedelta
import pandas as pd

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from forward.logger import (
    ForwardSignalLogger,
    convert_mt5_time_to_utc,
    verify_chain,
    check_git_remote_reachable,
    MIN_BUFFER_WARMUP
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("run_forward")


def init_mt5(terminal_path=None, login=None, password=None, server=None):
    """Initializes MetaTrader 5 desktop connection."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.error("MetaTrader5 python package not installed.")
        return None

    path = terminal_path or os.environ.get("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    login = login or os.environ.get("MT5_LOGIN", 115395873)
    password = password or os.environ.get("MT5_PASSWORD", "@iVu5gTh")
    server = server or os.environ.get("MT5_SERVER", "MetaQuotes-Demo")

    init_kwargs = {
        "path": path,
        "login": int(login),
        "password": str(password),
        "server": str(server),
        "timeout": 20000
    }
    logger.info(f"Connecting to MT5: server={server}, login={login}...")
    if not mt5.initialize(**init_kwargs):
        err = mt5.last_error()
        logger.error(f"MT5 initialization failed: {err}")
        return None

    term_info = mt5.terminal_info()
    acc_info = mt5.account_info()
    logger.info(f"MT5 Connected! Terminal build: {term_info.build if term_info else 'N/A'}, Account: {acc_info.login if acc_info else 'N/A'}")
    return mt5


def run_dry_run(mt5, symbol, forward_start_date, log_path, days=3, server_tz="EU_EET"):
    """
    Executes a historical dry-run replaying recent closed candles from live MT5 feed.
    No live orders are placed. Verifies resulting hash chain integrity.
    """
    logger.info(f"--- STARTING DRY-RUN REPLAY (Last {days} days on {symbol}) ---")
    num_bars = days * 96  # 96 15m bars per day
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, num_bars)
    if rates is None or len(rates) == 0:
        logger.error(f"Failed to fetch historical rates for dry-run: {mt5.last_error()}")
        return False

    logger.info(f"Fetched {len(rates)} closed bars from MT5 for dry run.")

    sig_logger = ForwardSignalLogger(
        forward_start_date=forward_start_date,
        log_path=log_path,
        check_git=False,
        check_frozen_hashes=False,
        server_tz=server_tz,
        git_push_anchors=False
    )

    for i, r in enumerate(rates):
        sig_logger.on_candle_closed({
            "time": r["time"],
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["tick_volume"]
        }, realized_spread=r["spread"] * 0.01 if "spread" in r.dtype.names else None)

    valid, msg, count = verify_chain(log_path)
    
    # Audit log breakdown
    with open(log_path, "r", encoding="utf-8") as f:
        events = [json.loads(l) for l in f if l.strip()]

    signals = [e for e in events if e["event"] == "SIGNAL"]
    status_counts = {}
    for s in signals:
        st = s["data"].get("filter_status", "UNKNOWN")
        status_counts[st] = status_counts.get(st, 0) + 1

    gaps = [e for e in events if e["event"] == "GAP"]
    gap_counts = {}
    for g in gaps:
        gt = g["data"].get("gap_type", "UNKNOWN")
        gap_counts[gt] = gap_counts.get(gt, 0) + 1

    fills = [e for e in events if e["event"] == "FILL"]
    closes = [e for e in events if e["event"] == "CLOSE"]
    n_trades = len(closes)

    from forward.execution_tracker import ExecutionTracker
    tracker = ExecutionTracker()
    sp_rep = tracker.generate_spread_report(events)

    logger.info("=== DRY-RUN REPLAY SUMMARY ===")
    logger.info(f"Total events logged: {count} | Chain integrity: {valid} ({msg})")
    logger.info(f"Signals by filter_status: {status_counts}")
    logger.info(f"Gaps breakdown: {gap_counts}")
    logger.info(f"Fills: {len(fills)} | Closed trades (N): {n_trades}")
    logger.info(f"Spread Observations: {sp_rep['total_observations']}")
    if sp_rep['total_observations'] > 0:
        logger.info(tracker.format_report_text(sp_rep))
    return valid


def run_forward_loop(
    forward_start_date,
    symbol="XAUUSD",
    timeframe="M15",
    log_path="forward/signals.jsonl",
    anchor_ledger="forward/anchors.json",
    server_tz="EU_EET",
    server_name="MetaQuotes-Demo",
    feed_name="MT5",
    git_push_anchors=False,
    git_remote="origin",
    git_branch="forward-anchors",
    poll_interval=5.0,
    check_git=True,
    check_frozen_hashes=True,
    rehearsal=False
):
    """
    Main forward paper-trading polling loop.
    Rebuilds pending orders, active positions and buffer on restart.
    Deduplicates closed candles and detects downtime gaps.
    """
    mt5 = init_mt5()
    if mt5 is None:
        sys.exit(1)

    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Failed to select symbol '{symbol}' in MT5.")
        mt5.shutdown()
        sys.exit(1)

    # Instantiate ForwardSignalLogger (automatically handles recovery if log_path exists)
    sig_logger = ForwardSignalLogger(
        forward_start_date=forward_start_date,
        log_path=log_path,
        check_git=check_git,
        check_frozen_hashes=check_frozen_hashes,
        external_anchor_path=anchor_ledger,
        server_tz=server_tz,
        server_name=server_name,
        symbol=symbol,
        feed_name=feed_name,
        git_push_anchors=git_push_anchors,
        git_remote=git_remote,
        git_branch=git_branch,
        rehearsal=rehearsal
    )

    # Pre-warm buffer if empty or incomplete
    if len(sig_logger._times_ny) < MIN_BUFFER_WARMUP:
        logger.info(f"Buffer has {len(sig_logger._times_ny)} bars (< {MIN_BUFFER_WARMUP}). Pre-warming from MT5 history...")
        warmup_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, 200)
        if warmup_rates is not None and len(warmup_rates) > 0:
            sig_logger.seed_buffer(warmup_rates)
            logger.info(f"Buffer successfully pre-warmed to {len(sig_logger._times_ny)} bars.")

    logger.info("Forward paper runner initialized and ready.")
    logger.info(f"Active Position: {sig_logger.active_position}")
    logger.info(f"Pending Order: {sig_logger.pending_order}")
    logger.info(f"Last Candle Time UTC: {sig_logger.last_candle_time_utc}")

    running = True

    def signal_handler(signum, frame):
        nonlocal running
        logger.info("Termination signal received. Shutting down gracefully...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while running:
        try:
            # Bar 0 is open/current; Bar 1 is the most recently CLOSED 15m bar
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, 2)
            if rates is not None and len(rates) > 0:
                closed_bar = rates[-1]  # The latest closed candle
                bar_time_utc = convert_mt5_time_to_utc(closed_bar["time"], server_tz=server_tz)

                # Deduplication check
                if sig_logger.last_candle_time_utc is None or bar_time_utc > sig_logger.last_candle_time_utc:
                    # Check for downtime gaps where multiple candles closed while process was stopped
                    if sig_logger.last_candle_time_utc is not None:
                        elapsed = bar_time_utc - sig_logger.last_candle_time_utc
                        if elapsed > timedelta(minutes=15):
                            missed_count = int(elapsed.total_seconds() // 900)
                            logger.warning(f"Downtime detected ({elapsed}): fetching {missed_count} missed bars...")
                            missed_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, min(missed_count + 1, 500))
                            if missed_rates is not None:
                                for mb in missed_rates:
                                    mb_time = convert_mt5_time_to_utc(mb["time"], server_tz=server_tz)
                                    if mb_time > sig_logger.last_candle_time_utc:
                                        spread = mb["spread"] * 0.01 if "spread" in mb.dtype.names else None
                                        sig_logger.on_candle_closed({
                                            "time": mb["time"],
                                            "open": mb["open"],
                                            "high": mb["high"],
                                            "low": mb["low"],
                                            "close": mb["close"],
                                            "volume": mb["tick_volume"]
                                        }, realized_spread=spread)
                                continue

                    # Process the single newly closed candle
                    spread = closed_bar["spread"] * 0.01 if "spread" in closed_bar.dtype.names else None
                    tick = mt5.symbol_info_tick(symbol)
                    tb = float(tick.bid) if tick is not None and getattr(tick, "bid", 0) > 0 else None
                    ta = float(tick.ask) if tick is not None and getattr(tick, "ask", 0) > 0 else None

                    logger.info(f"New closed candle: {bar_time_utc.isoformat()} O={closed_bar['open']} H={closed_bar['high']} L={closed_bar['low']} C={closed_bar['close']}")
                    sig_logger.on_candle_closed({
                        "time": closed_bar["time"],
                        "open": closed_bar["open"],
                        "high": closed_bar["high"],
                        "low": closed_bar["low"],
                        "close": closed_bar["close"],
                        "volume": closed_bar["tick_volume"],
                        "spread": closed_bar["spread"] if "spread" in closed_bar.dtype.names else None,
                        "tick_bid": tb,
                        "tick_ask": ta
                    }, realized_spread=spread)

            time.sleep(poll_interval)
        except Exception as e:
            logger.error(f"Error during polling loop: {e}", exc_info=True)
            time.sleep(poll_interval)

    logger.info("Closing MT5 connection...")
    mt5.shutdown()
    logger.info("Forward runner stopped cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Forward Paper-Trading Runner for XAUUSD 15m")
    parser.add_argument("--forward-start-date", required=True, help="Forward protocol start date (e.g. 2026-11-12T00:00:00Z)")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol to monitor (default: XAUUSD)")
    parser.add_argument("--timeframe", default="M15", help="Timeframe (default: M15)")
    parser.add_argument("--server-tz", default="EU_EET", help="Broker server timezone mode (default: EU_EET for MetaQuotes)")
    parser.add_argument("--server-name", default="MetaQuotes-Demo", help="Broker server name (default: MetaQuotes-Demo)")
    parser.add_argument("--feed-name", default="MT5", help="Data feed name (default: MT5)")
    parser.add_argument("--log-path", default="forward/signals.jsonl", help="Path to append-only JSONL log")
    parser.add_argument("--anchor-ledger", default="forward/anchors.json", help="Path to external anchor JSON ledger")
    parser.add_argument("--no-git-push-anchors", dest="git_push_anchors", action="store_false", help="Disable mandatory git push of anchors to remote")
    parser.set_defaults(git_push_anchors=True)
    parser.add_argument("--git-remote", default="origin", help="Git remote name for anchor pushes")
    parser.add_argument("--git-branch", default="forward-anchors", help="Git branch for anchor commits")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Polling interval in seconds (default: 5.0)")
    parser.add_argument("--dry-run-days", type=int, default=None, help="If set, executes a dry-run on recent MT5 history and exits")
    parser.add_argument("--no-git-check", action="store_true", help="Skip clean git tree check (for testing only)")
    parser.add_argument("--no-hash-check", action="store_true", help="Skip frozen hashes check (for testing only)")
    parser.add_argument("--rehearsal", action="store_true", help="Run in rehearsal mode (isolates log path and forces anchor branch to forward-anchors-rehearsal)")

    args = parser.parse_args()

    if args.rehearsal:
        if args.log_path == "forward/signals.jsonl":
            args.log_path = "forward/rehearsal.jsonl"
        if args.anchor_ledger == "forward/anchors.json":
            args.anchor_ledger = "forward/rehearsal_anchors.json"
        if args.git_branch == "forward-anchors":
            args.git_branch = "forward-anchors-rehearsal"

        prod_log = os.path.abspath(os.path.join(REPO_ROOT, "forward", "signals.jsonl"))
        if os.path.abspath(args.log_path) == prod_log:
            sys.exit("Error: Rehearsal mode strictly prohibits writing to production 'forward/signals.jsonl'.")
        if args.git_branch == "forward-anchors":
            sys.exit("Error: Rehearsal mode strictly prohibits pushing to production 'forward-anchors' branch.")

    if args.dry_run_days is not None:
        mt5 = init_mt5()
        if mt5 is None:
            sys.exit(1)
        ok = run_dry_run(
            mt5=mt5,
            symbol=args.symbol,
            forward_start_date=args.forward_start_date,
            log_path=args.log_path,
            days=args.dry_run_days,
            server_tz=args.server_tz
        )
        mt5.shutdown()
        sys.exit(0 if ok else 1)

    run_forward_loop(
        forward_start_date=args.forward_start_date,
        symbol=args.symbol,
        timeframe=args.timeframe,
        log_path=args.log_path,
        anchor_ledger=args.anchor_ledger,
        server_tz=args.server_tz,
        server_name=args.server_name,
        feed_name=args.feed_name,
        git_push_anchors=args.git_push_anchors,
        git_remote=args.git_remote,
        git_branch=args.git_branch,
        poll_interval=args.poll_interval,
        check_git=not args.no_git_check,
        check_frozen_hashes=not args.no_hash_check,
        rehearsal=args.rehearsal
    )


if __name__ == "__main__":
    main()
