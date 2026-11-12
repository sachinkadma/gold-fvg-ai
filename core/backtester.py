"""
Backtester Engine for XAUUSD 15m ICT Strategy.
Simulates order lifecycle with strict causality, realistic execution costs (spread, slippage, commission),
killzone & bar expiry, conservative SL-first intra-bar priority, and anti-overfitting data splits.
"""

import os
import sys
import argparse
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from core import config
    from core.data_loader import load_and_clean_data
    from core.signals import evaluate_causal_signal_at_bar, check_new_causal_swing_at_bar, is_in_killzone
    from core.metrics import compute_and_save_metrics
except ImportError:
    import config
    from data_loader import load_and_clean_data
    from signals import evaluate_causal_signal_at_bar, check_new_causal_swing_at_bar, is_in_killzone
    from metrics import compute_and_save_metrics


def compute_parameters_hash():
    """Returns a deterministic SHA256 hash of all strategy and execution parameters."""
    param_str = (
        f"SWING_LEN={config.SWING_LEN};"
        f"SL_BUFFER={config.SL_BUFFER};"
        f"RR={config.RR};"
        f"MIN_SL={config.MIN_SL};"
        f"ORDER_EXPIRY_BARS={config.ORDER_EXPIRY_BARS};"
        f"SPREAD={config.SPREAD};"
        f"SLIPPAGE={config.SLIPPAGE};"
        f"COMMISSION={config.COMMISSION_PER_LOT};"
        f"RISK_PERCENT={config.RISK_PERCENT};"
        f"INITIAL_EQUITY={config.INITIAL_EQUITY}"
    )
    return hashlib.sha256(param_str.encode("utf-8")).hexdigest()[:16]


def log_test_run(split_name, trade_count, expectancy_r, win_rate):
    """
    Logs test run with timestamp, param hash, and performance summary to test_runs.log.
    Returns the total number of previous test runs recorded.
    """
    log_file = config.TEST_RUNS_LOG
    prior_runs = 0
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            prior_runs = sum(1 for line in f if "SPLIT: TEST" in line)

    param_hash = compute_parameters_hash()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"[{now_str}] SPLIT: {split_name.upper()} | Run #{prior_runs + 1} | "
        f"Hash: {param_hash} | Trades: {trade_count} | "
        f"Expectancy: {expectancy_r:+.3f}R | WinRate: {win_rate:.1f}%\n"
    )

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

    return prior_runs


def check_position_exit(direction, sl_level, tp_level, cur_hi, cur_lo, spread=None, slippage=None):
    """
    Evaluates intra-candle exit for an active position using conservative SL-first logic.
    Returns: (exit_triggered: bool, exit_price: float or None, exit_reason: str or None)
    """
    spread = spread if spread is not None else config.SPREAD
    slippage = slippage if slippage is not None else config.SLIPPAGE
    exit_triggered = False
    exit_price = None
    exit_reason = None

    if direction == "BUY":
        sl_hit = (cur_lo <= sl_level)
        tp_hit = (cur_hi >= tp_level)
        if sl_hit and tp_hit:
            exit_triggered = True
            exit_price = round(sl_level - slippage, 2)
            exit_reason = "SL (Conservative Same-Candle)"
        elif sl_hit:
            exit_triggered = True
            exit_price = round(sl_level - slippage, 2)
            exit_reason = "SL"
        elif tp_hit:
            exit_triggered = True
            exit_price = round(tp_level, 2)
            exit_reason = "TP"

    elif direction == "SELL":
        sl_hit = (cur_hi >= sl_level)
        tp_hit = (cur_lo <= tp_level)
        if sl_hit and tp_hit:
            exit_triggered = True
            exit_price = round(sl_level + slippage + spread, 2)
            exit_reason = "SL (Conservative Same-Candle)"
        elif sl_hit:
            exit_triggered = True
            exit_price = round(sl_level + slippage + spread, 2)
            exit_reason = "SL"
        elif tp_hit:
            exit_triggered = True
            exit_price = round(tp_level + spread, 2)
            exit_reason = "TP"

    return exit_triggered, exit_price, exit_reason


def check_pending_order(p_dir, p_entry, p_sl, cur_hi, cur_lo, bars_elapsed, spread=None, max_expiry_bars=None, killzone_window_open=True):
    """
    Evaluates a pending limit order on a newly closed candle.
    Returns: (status: str, actual_entry: float or None)
    status is one of: {"INVALIDATED", "EXPIRED", "FILLED", "PENDING"}
    """
    spread = spread if spread is not None else config.SPREAD
    max_expiry_bars = max_expiry_bars if max_expiry_bars is not None else config.ORDER_EXPIRY_BARS

    # A. Invalidation: price hits SL before filling
    if (p_dir == "BUY" and cur_lo <= p_sl) or (p_dir == "SELL" and cur_hi >= p_sl):
        return "INVALIDATED", None

    # B. Expiry by duration or killzone close
    if bars_elapsed > max_expiry_bars or not killzone_window_open:
        return "EXPIRED", None

    # C. Fill check (BID limit order trades THROUGH entry by 0.01 pt)
    if p_dir == "BUY" and cur_lo <= (p_entry - 0.01):
        return "FILLED", round(p_entry + spread, 2)
    elif p_dir == "SELL" and cur_hi >= (p_entry + 0.01):
        return "FILLED", round(p_entry, 2)

    return "PENDING", None


def check_same_candle_sl(p_dir, p_sl, cur_hi, cur_lo, spread=None, slippage=None):
    """
    Conservative Rule: If fill candle itself touches SL, count as SL immediately.
    Returns: (hit: bool, actual_exit: float or None)
    """
    spread = spread if spread is not None else config.SPREAD
    slippage = slippage if slippage is not None else config.SLIPPAGE

    if p_dir == "BUY" and cur_lo <= p_sl:
        return True, round(p_sl - slippage, 2)
    elif p_dir == "SELL" and cur_hi >= p_sl:
        return True, round(p_sl + slippage + spread, 2)
    return False, None


def run_backtest_simulation(df, initial_equity=None, risk_percent=None):
    """
    Executes bar-by-bar causal simulation over the provided DataFrame.

    Returns:
        pd.DataFrame: Completed trades log.
    """
    equity = initial_equity if initial_equity is not None else config.INITIAL_EQUITY
    risk_pct = risk_percent if risk_percent is not None else config.RISK_PERCENT
    spread = config.SPREAD
    slippage = config.SLIPPAGE
    comm_per_lot = config.COMMISSION_PER_LOT

    trades = []
    pending_order = None
    position = None

    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    times = df.index

    swing_highs = []
    swing_lows = []
    s_len = config.SWING_LEN

    for t in range(n):
        cur_time = times[t]
        cur_hi = highs[t]
        cur_lo = lows[t]
        cur_cl = closes[t]

        # Update running causal swings confirmed at bar t
        new_sh, new_sl = check_new_causal_swing_at_bar(highs, lows, t, swing_len=s_len)
        if new_sh is not None:
            swing_highs.append(new_sh)
        if new_sl is not None:
            swing_lows.append(new_sl)

        # ---------------------------------------------------------------------
        # 1. EVALUATE ACTIVE POSITION
        # ---------------------------------------------------------------------
        if position is not None:
            direction = position["direction"]
            sl_level = position["sl"]
            tp_level = position["tp"]
            entry_price = position["entry_price"]
            lot_size = position["lot_size"]
            planned_risk_usd = position["planned_risk_usd"]

            exit_triggered, exit_price, exit_reason = check_position_exit(
                direction=direction,
                sl_level=sl_level,
                tp_level=tp_level,
                cur_hi=cur_hi,
                cur_lo=cur_lo,
                spread=spread,
                slippage=slippage
            )

            if exit_triggered:
                # Calculate PnL
                if direction == "BUY":
                    price_delta = exit_price - entry_price
                else:
                    price_delta = entry_price - exit_price

                gross_usd = price_delta * config.CONTRACT_SIZE * lot_size
                commission_usd = lot_size * comm_per_lot
                net_usd = round(gross_usd - commission_usd, 2)
                net_r = round(net_usd / planned_risk_usd, 3) if planned_risk_usd > 0 else 0.0

                equity_before = position["equity_before"]
                equity_after = round(equity_before + net_usd, 2)
                equity = equity_after

                duration_bars = t - position["fill_bar"]

                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": cur_time,
                    "direction": direction,
                    "killzone": position["killzone"],
                    "entry_target": position["entry_target"],
                    "entry_price": entry_price,
                    "sl_price": sl_level,
                    "tp_price": tp_level,
                    "exit_price": exit_price,
                    "lot_size": lot_size,
                    "planned_risk_usd": planned_risk_usd,
                    "gross_usd": round(gross_usd, 2),
                    "commission_usd": round(commission_usd, 2),
                    "result_usd": net_usd,
                    "result_r": net_r,
                    "exit_reason": exit_reason,
                    "duration_bars": duration_bars,
                    "equity_before": equity_before,
                    "equity_after": equity_after
                })

                position = None

        # ---------------------------------------------------------------------
        # 2. EVALUATE PENDING LIMIT ORDER
        # ---------------------------------------------------------------------
        elif pending_order is not None:
            p_dir = pending_order["direction"]
            p_entry = pending_order["entry"]
            p_sl = pending_order["sl"]
            p_tp = pending_order["tp"]
            p_sig_bar = pending_order["sig_bar"]
            p_kz = pending_order["killzone"]

            kz_window_open = (is_in_killzone(cur_time.time()) == p_kz)
            order_status, actual_entry = check_pending_order(
                p_dir=p_dir,
                p_entry=p_entry,
                p_sl=p_sl,
                cur_hi=cur_hi,
                cur_lo=cur_lo,
                bars_elapsed=(t - p_sig_bar),
                spread=spread,
                max_expiry_bars=config.ORDER_EXPIRY_BARS,
                killzone_window_open=kz_window_open
            )

            if order_status in ("INVALIDATED", "EXPIRED"):
                pending_order = None
            elif order_status == "FILLED":
                filled = True

                if filled:
                    planned_risk_dist = abs(p_entry - p_sl)
                    risk_capital_usd = round(equity * risk_pct, 2)
                    # Sizing: risk_capital / (stop_distance * contract_size)
                    lot_size = round(risk_capital_usd / (planned_risk_dist * config.CONTRACT_SIZE), 2)
                    lot_size = max(0.01, lot_size)

                    fill_candle_sl, actual_exit = check_same_candle_sl(
                        p_dir=p_dir,
                        p_sl=p_sl,
                        cur_hi=cur_hi,
                        cur_lo=cur_lo,
                        spread=spread,
                        slippage=slippage
                    )

                    if fill_candle_sl:
                        if p_dir == "BUY":
                            price_delta = actual_exit - actual_entry
                        else:
                            price_delta = actual_entry - actual_exit

                        gross_usd = price_delta * config.CONTRACT_SIZE * lot_size
                        commission_usd = lot_size * comm_per_lot
                        net_usd = round(gross_usd - commission_usd, 2)
                        net_r = round(net_usd / risk_capital_usd, 3)

                        equity_before = equity
                        equity_after = round(equity + net_usd, 2)
                        equity = equity_after

                        trades.append({
                            "entry_time": cur_time,
                            "exit_time": cur_time,
                            "direction": p_dir,
                            "killzone": p_kz,
                            "entry_target": p_entry,
                            "entry_price": actual_entry,
                            "sl_price": p_sl,
                            "tp_price": p_tp,
                            "exit_price": actual_exit,
                            "lot_size": lot_size,
                            "planned_risk_usd": risk_capital_usd,
                            "gross_usd": round(gross_usd, 2),
                            "commission_usd": round(commission_usd, 2),
                            "result_usd": net_usd,
                            "result_r": net_r,
                            "exit_reason": "SL (Fill Candle Touched SL)",
                            "duration_bars": 0,
                            "equity_before": equity_before,
                            "equity_after": equity_after
                        })
                        pending_order = None
                    else:
                        # Position is now live
                        position = {
                            "direction": p_dir,
                            "entry_time": cur_time,
                            "fill_bar": t,
                            "entry_target": p_entry,
                            "entry_price": actual_entry,
                            "sl": p_sl,
                            "tp": p_tp,
                            "killzone": p_kz,
                            "lot_size": lot_size,
                            "planned_risk_usd": risk_capital_usd,
                            "equity_before": equity
                        }
                        pending_order = None

        # ---------------------------------------------------------------------
        # 3. EVALUATE NEW SIGNAL AT CANDLE CLOSE
        # ---------------------------------------------------------------------
        # Only check if neither a position nor a pending order is active (stateless baseline)
        if position is None and pending_order is None and t >= 30:
            sig = evaluate_causal_signal_at_bar(df, t)
            if sig is not None:
                # Register pending limit order; first chance to fill is t+1
                pending_order = {
                    "direction": sig["direction"],
                    "entry": sig["entry"],
                    "sl": sig["sl"],
                    "tp": sig["tp"],
                    "killzone": sig["killzone"],
                    "sig_bar": t,
                    "sig_time": cur_time
                }

    return pd.DataFrame(trades)


def main():
    parser = argparse.ArgumentParser(description="Institutional XAUUSD ICT Backtester")
    parser.add_argument("--split", choices=["train", "test"], required=True, help="Data split to run ('train' or 'test')")
    parser.add_argument("--final", action="store_true", help="Required flag when running on the locked TEST set")
    parser.add_argument("--data", default=config.DATA_PATH, help="Path to 15m CSV data file")
    parser.add_argument("--data-tz", default=config.DATA_TZ, help="Source timezone of CSV")
    parser.add_argument("--train-start", default=None, help="Override train start date")
    parser.add_argument("--train-end", default=None, help="Override train end date")
    parser.add_argument("--test-start", default=None, help="Override test start date")
    parser.add_argument("--test-end", default=None, help="Override test end date")

    args = parser.parse_args()

    # Enforce anti-overfitting protocol
    if args.split == "test" and not args.final:
        print("\n" + "!" * 70)
        print("ERROR: Execution aborted! The locked TEST set requires the '--final' flag.")
        print("This prevents accidental data snooping and multi-testing p-hacking bias.")
        print("Usage: python backtester.py --split test --final")
        print("!" * 70 + "\n")
        sys.exit(1)

    # 1. Load and clean data
    print(f"\n[BACKTESTER] Loading dataset from: {args.data}...")
    df_all = load_and_clean_data(file_path=args.data, data_tz=args.data_tz, verbose=True)

    data_start = df_all.index[0]
    data_end = df_all.index[-1]

    # 2. Determine split date ranges
    train_start_str = args.train_start or config.TRAIN_START
    train_end_str = args.train_end or config.TRAIN_END
    test_start_str = args.test_start or config.TEST_START
    test_end_str = args.test_end or config.TEST_END

    # Check date coverage in current dataset
    benchmark_train_overlap = (data_start.strftime("%Y-%m-%d") <= train_end_str) and (data_end.strftime("%Y-%m-%d") >= train_start_str)

    if args.split == "train":
        if benchmark_train_overlap:
            split_df = df_all.loc[train_start_str:train_end_str].copy()
            split_label = f"train_{train_start_str[:4]}_{train_end_str[:4]}"
        else:
            # Fallback for recent data extracts (e.g. 2024-2026 MT5 export)
            fallback_train_end = "2025-12-31"
            print("-" * 65)
            print(f"[DATA PARTITION NOTICE]")
            print(f"Configured benchmark TRAIN window ({train_start_str} to {train_end_str})")
            print(f"has no bars in current file (data bounds: {data_start.date()} to {data_end.date()}).")
            print(f"Using available in-sample range for TRAIN: {data_start.date()} to {fallback_train_end}")
            print(f"(leaving 2026 locked for out-of-sample TEST).")
            print("-" * 65)
            split_df = df_all.loc[:fallback_train_end].copy()
            split_label = f"train_insample_{data_start.strftime('%Y%m')}_202512"

    elif args.split == "test":
        if benchmark_train_overlap:
            split_df = df_all.loc[test_start_str:test_end_str].copy()
            split_label = f"test_{test_start_str[:4]}_{test_end_str[:4]}"
        else:
            fallback_test_start = "2026-01-01"
            print("-" * 65)
            print(f"[DATA PARTITION NOTICE]")
            print(f"Using out-of-sample TEST window: {fallback_test_start} to {data_end.date()}")
            print("-" * 65)
            split_df = df_all.loc[fallback_test_start:].copy()
            split_label = f"test_outofsample_2026"

    print(f"\n[BACKTESTER] Running simulation on {args.split.upper()} set ({len(split_df):,} candles)...")
    if split_df.empty:
        print(f"ERROR: No candles found in the selected {args.split} split!")
        sys.exit(1)

    # 3. Run simulation
    trades_df = run_backtest_simulation(split_df)

    cost_assumptions = {
        "spread": config.SPREAD,
        "slippage": config.SLIPPAGE,
        "commission_per_lot": config.COMMISSION_PER_LOT,
        "min_sl": config.MIN_SL,
        "sl_buffer": config.SL_BUFFER,
        "rr": config.RR,
        "swing_len": config.SWING_LEN,
        "order_expiry_bars": config.ORDER_EXPIRY_BARS,
        "risk_percent": config.RISK_PERCENT,
        "initial_equity": config.INITIAL_EQUITY
    }

    # 4. Compute metrics and generate outputs
    report = compute_and_save_metrics(trades_df, split_label=args.split, cost_assumptions=cost_assumptions, split_df=split_df)

    # 5. Anti-overfitting tracking for TEST set
    if args.split == "test":
        prior_test_runs = log_test_run(
            split_name=args.split,
            trade_count=report["overall"]["trade_count"],
            expectancy_r=report["overall"]["expectancy_r"],
            win_rate=report["overall"]["win_rate_pct"]
        )
        print("!" * 68)
        print(f"ANTI-OVERFITTING AUDIT WARNING:")
        print(f"The locked TEST set has now been run {prior_test_runs + 1} time(s).")
        print(f"Logged run parameters and performance to '{config.TEST_RUNS_LOG}'.")
        print("!" * 68 + "\n")


if __name__ == "__main__":
    main()
