"""
Part 1 Diagnostics:
1. Gross vs Net expectancy overall and per-year.
2. Stop distance buckets (3-5, 5-8, 8-12, 12+).
3. BID-side spread handling audit and comparison.
"""

import pandas as pd
import numpy as np
import sys
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from core import config
    from core.data_loader import load_and_clean_data
    from core.backtester import run_backtest_simulation
    from core.signals import evaluate_causal_signal_at_bar, is_in_killzone
except ImportError:
    import config
    from data_loader import load_and_clean_data
    from backtester import run_backtest_simulation
    from signals import evaluate_causal_signal_at_bar, is_in_killzone

def run_diagnostics():
    df_all = load_and_clean_data("data/xauusd_15m_long.csv", data_tz="UTC", target_tz="America/New_York", verbose=False)
    df_train = df_all.loc["2015-01-01":"2021-12-31"].copy()

    # 1. Run baseline as currently implemented
    trades_current = run_backtest_simulation(df_train)
    
    # 2. Re-simulate with exact BID-side spread handling:
    # Buys fill at ask (p_entry + spread), sells at bid (p_entry).
    # Buys exit at bid: TP = tp_level, SL = sl_level - slippage
    # Sells exit at ask: TP = tp_level + spread, SL = sl_level + slippage + spread
    trades_bid_side = run_simulation_bidside(df_train)

    print("=== PART 1: AUDIT & COMPARISON OF SPREAD HANDLING ===")
    print(f"Current backtester (half-spread): {len(trades_current)} trades | Net Expectancy: {trades_current['result_r'].mean():+.3f} R | Net R: {trades_current['result_r'].sum():+.2f} R")
    print(f"Fixed Bid-side (full spread on buy, 0 on sell): {len(trades_bid_side)} trades | Net Expectancy: {trades_bid_side['result_r'].mean():+.3f} R | Net R: {trades_bid_side['result_r'].sum():+.2f} R")

    return trades_current, trades_bid_side

def run_simulation_bidside(df):
    spread = config.SPREAD
    slippage = config.SLIPPAGE
    comm_per_lot = config.COMMISSION_PER_LOT
    risk_pct = config.RISK_PCT

    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    times = df.index

    position = None
    pending_order = None
    trades = []
    equity = config.INITIAL_EQUITY

    for t in range(n):
        cur_time = times[t]
        cur_hi = highs[t]
        cur_lo = lows[t]

        # 1. Position exit check
        if position is not None:
            direction = position["direction"]
            entry_price = position["entry_price"]
            sl_level = position["sl_price"]
            tp_level = position["tp_price"]
            lot_size = position["lot_size"]
            planned_risk_usd = position["planned_risk_usd"]

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
                    exit_price = round(tp_level, 2) # Sells at bid!
                    exit_reason = "TP"
            else: # SELL
                sl_hit = (cur_hi >= sl_level)
                tp_hit = (cur_lo <= tp_level)
                if sl_hit and tp_hit:
                    exit_triggered = True
                    exit_price = round(sl_level + slippage + spread, 2) # Buys to cover at ask!
                    exit_reason = "SL (Conservative Same-Candle)"
                elif sl_hit:
                    exit_triggered = True
                    exit_price = round(sl_level + slippage + spread, 2) # Buys to cover at ask!
                    exit_reason = "SL"
                elif tp_hit:
                    exit_triggered = True
                    exit_price = round(tp_level + spread, 2) # Buys to cover at ask!
                    exit_reason = "TP"

            if exit_triggered:
                if direction == "BUY":
                    price_delta = exit_price - entry_price
                else:
                    price_delta = entry_price - exit_price

                gross_usd = price_delta * config.CONTRACT_SIZE * lot_size
                commission_usd = lot_size * comm_per_lot
                net_usd = round(gross_usd - commission_usd, 2)
                net_r = round(net_usd / planned_risk_usd, 3) if planned_risk_usd > 0 else 0.0

                # Pre-cost Gross R
                gross_r = 2.0 if "TP" in exit_reason else -1.0

                equity_before = position["equity_before"]
                equity_after = round(equity_before + net_usd, 2)
                equity = equity_after

                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": cur_time,
                    "year": cur_time.year,
                    "direction": direction,
                    "killzone": position["killzone"],
                    "entry_target": position["entry_target"],
                    "entry_price": entry_price,
                    "sl_price": sl_level,
                    "tp_price": tp_level,
                    "exit_price": exit_price,
                    "stop_distance": abs(position["entry_target"] - sl_level),
                    "lot_size": lot_size,
                    "planned_risk_usd": planned_risk_usd,
                    "gross_usd": round(gross_usd, 2),
                    "commission_usd": round(commission_usd, 2),
                    "result_usd": net_usd,
                    "result_r": net_r,
                    "gross_r": gross_r,
                    "cost_r": round(gross_r - net_r, 3),
                    "exit_reason": exit_reason,
                    "duration_bars": t - position["fill_bar"],
                    "equity_before": equity_before,
                    "equity_after": equity_after
                })
                position = None

        # 2. Pending limit order check
        elif pending_order is not None:
            p_dir = pending_order["direction"]
            p_entry = pending_order["entry"]
            p_sl = pending_order["sl"]
            p_tp = pending_order["tp"]
            p_sig_bar = pending_order["sig_bar"]
            p_kz = pending_order["killzone"]

            # Invalidation check prior to fill
            invalidated = False
            if p_dir == "BUY" and cur_lo <= p_sl:
                invalidated = True
            elif p_dir == "SELL" and cur_hi >= p_sl:
                invalidated = True

            if invalidated:
                pending_order = None
            elif (t - p_sig_bar) > config.ORDER_EXPIRY_BARS:
                pending_order = None
            elif is_in_killzone(cur_time.time()) != p_kz:
                pending_order = None
            else:
                filled = False
                if p_dir == "BUY" and cur_lo <= (p_entry - 0.01):
                    filled = True
                    actual_entry = round(p_entry + spread, 2) # Buys fill at ask (bid + full spread)
                elif p_dir == "SELL" and cur_hi >= (p_entry + 0.01):
                    filled = True
                    actual_entry = round(p_entry, 2) # Sells fill at bid

                if filled:
                    planned_risk_dist = abs(p_entry - p_sl)
                    risk_capital_usd = round(equity * risk_pct, 2)
                    lot_size = round(risk_capital_usd / (planned_risk_dist * config.CONTRACT_SIZE), 2)
                    lot_size = max(0.01, lot_size)

                    # Same-candle SL check
                    fill_candle_sl = False
                    if p_dir == "BUY" and cur_lo <= p_sl:
                        fill_candle_sl = True
                        actual_exit = round(p_sl - slippage, 2)
                    elif p_dir == "SELL" and cur_hi >= p_sl:
                        fill_candle_sl = True
                        actual_exit = round(p_sl + slippage + spread, 2)

                    if fill_candle_sl:
                        if p_dir == "BUY":
                            p_delta = actual_exit - actual_entry
                        else:
                            p_delta = actual_entry - actual_exit
                        g_usd = p_delta * config.CONTRACT_SIZE * lot_size
                        c_usd = lot_size * comm_per_lot
                        n_usd = round(g_usd - c_usd, 2)
                        n_r = round(n_usd / risk_capital_usd, 3)
                        equity_before = equity
                        equity_after = round(equity + n_usd, 2)
                        equity = equity_after

                        trades.append({
                            "entry_time": cur_time,
                            "exit_time": cur_time,
                            "year": cur_time.year,
                            "direction": p_dir,
                            "killzone": p_kz,
                            "entry_target": p_entry,
                            "entry_price": actual_entry,
                            "sl_price": p_sl,
                            "tp_price": p_tp,
                            "exit_price": actual_exit,
                            "stop_distance": planned_risk_dist,
                            "lot_size": lot_size,
                            "planned_risk_usd": risk_capital_usd,
                            "gross_usd": round(g_usd, 2),
                            "commission_usd": round(c_usd, 2),
                            "result_usd": n_usd,
                            "result_r": n_r,
                            "gross_r": -1.0,
                            "cost_r": round(-1.0 - n_r, 3),
                            "exit_reason": "SL (Conservative Same-Candle)",
                            "duration_bars": 0,
                            "equity_before": equity_before,
                            "equity_after": equity_after
                        })
                        pending_order = None
                    else:
                        position = {
                            "direction": p_dir,
                            "entry_target": p_entry,
                            "entry_price": actual_entry,
                            "fill_bar": t,
                            "entry_time": cur_time,
                            "sl_price": p_sl,
                            "tp_price": p_tp,
                            "killzone": p_kz,
                            "lot_size": lot_size,
                            "planned_risk_usd": risk_capital_usd,
                            "equity_before": equity
                        }
                        pending_order = None

        # 3. New causal signal check
        if position is None and pending_order is None and t >= 30:
            sig = evaluate_causal_signal_at_bar(df, t)
            if sig is not None:
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

if __name__ == "__main__":
    t_curr, t_bid = run_diagnostics()
    # Print diagnostics table
    df_t = t_bid
    print("\n--- a) GROSS VS NET EXPECTANCY OVERALL & PER YEAR ---")
    overall_gross = df_t["gross_r"].mean()
    overall_net = df_t["result_r"].mean()
    print(f"Overall (2015-2021): Gross = {overall_gross:+.3f} R | Net = {overall_net:+.3f} R | Cost Drag = {overall_gross - overall_net:.3f} R")
    print("Year   Trades  Gross Exp (R)  Net Exp (R)  Cost Drag (R)  Win Rate")
    for yr, g in df_t.groupby("year"):
        ge = g["gross_r"].mean()
        ne = g["result_r"].mean()
        wr = (g["result_r"] > 0).mean() * 100.0
        print(f"{yr}   {len(g):<6}  {ge:+.3f} R        {ne:+.3f} R      {ge - ne:.3f} R        {wr:.1f}%")

    print("\n--- b) BUCKET BY STOP DISTANCE ---")
    bins = [3.0, 5.0, 8.0, 12.0, 999.0]
    labels = ["3-5", "5-8", "8-12", "12+"]
    df_t["stop_bucket"] = pd.cut(df_t["stop_distance"], bins=bins, labels=labels, right=False)
    print("Bucket   Trades  Win Rate  Gross Exp (R)  Net Exp (R)  Avg Cost (R)")
    for b in labels:
        g = df_t[df_t["stop_bucket"] == b]
        if len(g) > 0:
            wr = (g["result_r"] > 0).mean() * 100.0
            ge = g["gross_r"].mean()
            ne = g["result_r"].mean()
            cost = g["cost_r"].mean()
            print(f"{b:<8} {len(g):<7} {wr:.1f}%     {ge:+.3f} R        {ne:+.3f} R      {cost:.3f} R")
