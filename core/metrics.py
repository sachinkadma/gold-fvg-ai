"""
Metrics calculation module for XAUUSD ICT Strategy Backtesting Engine.
Computes comprehensive institutional performance metrics, bootstrap confidence intervals,
sub-period breakdowns (per year, per killzone, per direction), buy-and-hold gold benchmark,
and exports trades.csv, metrics.json, and equity_curve.png.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt


def calculate_metrics_for_trades(trades_df):
    """
    Computes performance metrics on a DataFrame of completed trades.
    """
    if trades_df.empty:
        return {
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "average_r": 0.0,
            "expectancy_r": 0.0,
            "expectancy_ci_95": [0.0, 0.0],
            "max_drawdown_r": 0.0,
            "max_drawdown_pct": 0.0,
            "longest_losing_streak": 0,
            "avg_duration_bars": 0.0,
            "total_net_usd": 0.0,
            "total_net_r": 0.0,
            "sample_size_warning": True
        }

    r_vals = trades_df["result_r"].values
    usd_vals = trades_df["result_usd"].values
    n_trades = len(r_vals)

    wins = r_vals[r_vals > 0]
    losses = r_vals[r_vals <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / n_trades) * 100.0 if n_trades > 0 else 0.0

    gross_gains_r = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_losses_r = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = (gross_gains_r / gross_losses_r) if gross_losses_r > 0 else (999.0 if gross_gains_r > 0 else 0.0)

    avg_r = float(np.mean(r_vals))
    expectancy_r = avg_r

    # Longest losing streak
    longest_losing_streak = 0
    current_losing_streak = 0
    for r in r_vals:
        if r <= 0:
            current_losing_streak += 1
            if current_losing_streak > longest_losing_streak:
                longest_losing_streak = current_losing_streak
        else:
            current_losing_streak = 0

    # Max Drawdown in R
    cum_r = np.cumsum(r_vals)
    cum_r_with_zero = np.insert(cum_r, 0, 0.0)
    peak_r = np.maximum.accumulate(cum_r_with_zero)
    drawdowns_r = peak_r - cum_r_with_zero
    max_dd_r = float(np.max(drawdowns_r)) if len(drawdowns_r) > 0 else 0.0

    # Max Drawdown in % of equity
    if "equity_after" in trades_df.columns:
        equities = trades_df["equity_after"].values
        start_eq = float(trades_df["equity_before"].iloc[0]) if "equity_before" in trades_df.columns else 10000.0
        eq_curve = np.insert(equities, 0, start_eq)
        peak_eq = np.maximum.accumulate(eq_curve)
        pct_dd = (peak_eq - eq_curve) / peak_eq * 100.0
        max_dd_pct = float(np.max(pct_dd)) if len(pct_dd) > 0 else 0.0
    else:
        max_dd_pct = 0.0

    # Average trade duration
    avg_duration_bars = float(trades_df["duration_bars"].mean()) if "duration_bars" in trades_df.columns else 0.0

    # Bootstrap 95% Confidence Interval for Expectancy (1,000 resamples)
    if n_trades >= 5:
        rng = np.random.default_rng(42)
        bootstrap_means = [
            float(np.mean(rng.choice(r_vals, size=n_trades, replace=True)))
            for _ in range(1000)
        ]
        ci_lower = float(np.percentile(bootstrap_means, 2.5))
        ci_upper = float(np.percentile(bootstrap_means, 97.5))
    else:
        ci_lower = expectancy_r
        ci_upper = expectancy_r

    return {
        "trade_count": int(n_trades),
        "win_count": int(win_count),
        "loss_count": int(loss_count),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "average_r": round(avg_r, 3),
        "expectancy_r": round(expectancy_r, 3),
        "expectancy_ci_95": [round(ci_lower, 3), round(ci_upper, 3)],
        "max_drawdown_r": round(max_dd_r, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "longest_losing_streak": int(longest_losing_streak),
        "avg_duration_bars": round(avg_duration_bars, 1),
        "total_net_usd": round(float(np.sum(usd_vals)), 2),
        "total_net_r": round(float(np.sum(r_vals)), 2),
        "sample_size_warning": bool(n_trades < 100)
    }


def compute_and_save_metrics(trades_df, split_label, cost_assumptions, output_dir=".", split_df=None):
    """
    Computes overall, annual, direction (long/short), and per-killzone metrics,
    calculates buy-and-hold gold return, writes metrics.json, saves trades.csv,
    and plots equity_curve.png.
    """
    # 1. Overall metrics
    overall_metrics = calculate_metrics_for_trades(trades_df)

    # 2. Per-Year metrics
    per_year_metrics = {}
    if not trades_df.empty and "entry_time" in trades_df.columns:
        trades_df["year"] = pd.to_datetime(trades_df["entry_time"]).dt.year
        for yr, group in trades_df.groupby("year"):
            per_year_metrics[str(yr)] = calculate_metrics_for_trades(group)

    # 3. Per-Killzone metrics
    per_killzone_metrics = {}
    if not trades_df.empty and "killzone" in trades_df.columns:
        for kz, group in trades_df.groupby("killzone"):
            per_killzone_metrics[str(kz)] = calculate_metrics_for_trades(group)

    # 4. Long vs Short breakdown (direction)
    per_direction_metrics = {}
    if not trades_df.empty and "direction" in trades_df.columns:
        for d, group in trades_df.groupby("direction"):
            label = "Long (BUY)" if d == "BUY" else "Short (SELL)"
            per_direction_metrics[label] = calculate_metrics_for_trades(group)

    # 5. Buy-and-Hold Gold Benchmark
    buy_hold_data = None
    if split_df is not None and not split_df.empty and "close" in split_df.columns:
        start_p = float(split_df["close"].iloc[0])
        end_p = float(split_df["close"].iloc[-1])
        ret_pct = ((end_p - start_p) / start_p) * 100.0
        pts_diff = end_p - start_p
        buy_hold_data = {
            "start_date": str(split_df.index[0]),
            "end_date": str(split_df.index[-1]),
            "start_price": round(start_p, 2),
            "end_price": round(end_p, 2),
            "price_change_pts": round(pts_diff, 2),
            "buy_and_hold_return_pct": round(ret_pct, 2)
        }

    # Combine into comprehensive report dict
    full_report = {
        "split": split_label,
        "cost_assumptions": cost_assumptions,
        "overall": overall_metrics,
        "per_direction": per_direction_metrics,
        "per_year": per_year_metrics,
        "per_killzone": per_killzone_metrics,
        "buy_and_hold_benchmark": buy_hold_data
    }

    # 6. Save metrics.json
    metrics_path = f"{output_dir}/metrics.json" if output_dir != "." else "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # 7. Save trades.csv
    trades_path = f"{output_dir}/trades.csv" if output_dir != "." else "trades.csv"
    trades_df.to_csv(trades_path, index=False)

    # 8. Generate and save equity_curve.png
    plot_path = f"{output_dir}/equity_curve.png" if output_dir != "." else "equity_curve.png"
    generate_equity_curve_plot(trades_df, split_label, overall_metrics, plot_path)

    # Print summary to console
    print_metrics_summary(full_report)

    return full_report


def generate_equity_curve_plot(trades_df, split_label, metrics, output_file):
    """
    Renders institutional dark-themed Equity Curve in R-Multiples.
    """
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0F172A")
    ax1.set_facecolor("#1E293B")
    ax2.set_facecolor("#1E293B")

    if trades_df.empty:
        ax1.text(0.5, 0.5, "No Trades Executed in Split Period", horizontalalignment="center", verticalalignment="center", color="#94A3B8", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        return

    r_vals = trades_df["result_r"].values
    cum_r = np.cumsum(r_vals)
    cum_r_with_zero = np.insert(cum_r, 0, 0.0)
    peak_r = np.maximum.accumulate(cum_r_with_zero)
    drawdown_r = peak_r - cum_r_with_zero
    trade_numbers = np.arange(len(cum_r_with_zero))

    # Upper plot: Equity in R
    ax1.plot(trade_numbers, cum_r_with_zero, color="#10B981", linewidth=2.0, label="Realized Edge (R)")
    ax1.plot(trade_numbers, peak_r, color="#64748B", linewidth=1.0, linestyle="--", label="Peak Watermark")
    ax1.axhline(0, color="#475569", linestyle=":", linewidth=1.0)
    ax1.fill_between(trade_numbers, cum_r_with_zero, peak_r, color="#EF4444", alpha=0.15)

    title_str = (
        f"XAUUSD 15m Baseline ICT Strategy -- {split_label.upper()} Split (Conservative Costs)\n"
        f"Trades: {metrics['trade_count']} | Win Rate: {metrics['win_rate_pct']:.1f}% | "
        f"Expectancy: {metrics['expectancy_r']:+.3f}R | Profit Factor: {metrics['profit_factor']:.2f} | "
        f"Max DD: {metrics['max_drawdown_r']:.2f}R ({metrics['max_drawdown_pct']:.1f}%)"
    )
    ax1.set_title(title_str, color="#F8FAFC", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Cumulative Return (R)", color="#F8FAFC", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.3, color="#475569")
    ax1.legend(loc="upper left", framealpha=0.6)

    # Lower plot: Drawdown in R
    ax2.fill_between(trade_numbers, -drawdown_r, 0, color="#EF4444", alpha=0.4, label="Drawdown (R)")
    ax2.plot(trade_numbers, -drawdown_r, color="#EF4444", linewidth=1.0)
    ax2.set_ylabel("Drawdown (R)", color="#F8FAFC", fontsize=11)
    ax2.set_xlabel("Trade Number", color="#F8FAFC", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.3, color="#475569")
    ax2.legend(loc="lower left", framealpha=0.6)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def print_metrics_summary(full_report):
    """
    Prints institutional metrics report to stdout.
    """
    ov = full_report["overall"]
    split = full_report["split"].upper()
    costs = full_report["cost_assumptions"]

    print("\n" + "=" * 68)
    print(f"  QUANT PERFORMANCE REPORT -- {split} SPLIT (AFTER COSTS)")
    print("=" * 68)
    print(f"Cost Assumptions: Spread={costs.get('spread', 0)} pts | Slippage={costs.get('slippage', 0)} pts | Comm=${costs.get('commission_per_lot', 0)}/lot")
    print("-" * 68)
    print(f"Total Trades:           {ov['trade_count']}")
    print(f"Win Rate:               {ov['win_rate_pct']:.1f}% ({ov['win_count']} Wins / {ov['loss_count']} Losses)")
    print(f"Profit Factor:          {ov['profit_factor']:.2f}")
    print(f"Average R:              {ov['average_r']:+.3f} R")
    print(f"Expectancy (R/trade):   {ov['expectancy_r']:+.3f} R")
    print(f"95% Bootstrap CI:       [{ov['expectancy_ci_95'][0]:+.3f} R, {ov['expectancy_ci_95'][1]:+.3f} R]")
    print(f"Max Drawdown (R):       {ov['max_drawdown_r']:.2f} R")
    print(f"Max Drawdown (%):       {ov['max_drawdown_pct']:.2f}%")
    print(f"Longest Losing Streak:  {ov['longest_losing_streak']} trades")
    print(f"Avg Trade Duration:     {ov['avg_duration_bars']:.1f} bars")
    print(f"Total Net Return:       {ov['total_net_r']:+.2f} R  (${ov['total_net_usd']:+,.2f} USD)")

    if ov["sample_size_warning"]:
        print("-" * 68)
        print(">> WARNING: Trade count < 100. Results are statistically weak and")
        print("   exhibit wide confidence intervals.")

    # Buy and Hold Benchmark
    bh = full_report.get("buy_and_hold_benchmark")
    if bh:
        print("-" * 68)
        print("BUY-AND-HOLD GOLD BENCHMARK (FOR CONTEXT):")
        print(f"Period:                 {bh['start_date']} -> {bh['end_date']}")
        print(f"Start / End Gold Spot:  ${bh['start_price']:,.2f} -> ${bh['end_price']:,.2f}")
        print(f"Gold Price Delta:       {bh['price_change_pts']:+,.2f} USD/oz")
        print(f"Buy-and-Hold Return:    {bh['buy_and_hold_return_pct']:+.2f}%")

    # Long vs Short Breakdown
    if full_report.get("per_direction"):
        print("-" * 68)
        print("LONG VS SHORT BREAKDOWN:")
        print(f"{'Direction':<14} {'Trades':<8} {'Win Rate':<12} {'Expectancy':<14} {'95% CI':<22} {'Net R':<10}")
        for dir_name, data in full_report["per_direction"].items():
            ci_s = f"[{data['expectancy_ci_95'][0]:+.3f}, {data['expectancy_ci_95'][1]:+.3f}]"
            print(f"{dir_name:<14} {data['trade_count']:<8} {data['win_rate_pct']:<11.1f}% {data['expectancy_r']:<+13.3f}R {ci_s:<22} {data['total_net_r']:<+9.2f}R")

    # Breakdown per Year
    if full_report.get("per_year"):
        print("-" * 68)
        print("PER-YEAR BREAKDOWN:")
        print(f"{'Year':<8} {'Trades':<8} {'Win Rate':<12} {'Expectancy':<14} {'Profit Factor':<14} {'Net R':<10}")
        for yr, data in full_report["per_year"].items():
            print(f"{yr:<8} {data['trade_count']:<8} {data['win_rate_pct']:<11.1f}% {data['expectancy_r']:<+13.3f}R {data['profit_factor']:<13.2f} {data['total_net_r']:<+9.2f}R")

    # Breakdown per Killzone
    if full_report.get("per_killzone"):
        print("-" * 68)
        print("PER-KILLZONE BREAKDOWN:")
        print(f"{'Killzone':<14} {'Trades':<8} {'Win Rate':<12} {'Expectancy':<14} {'Profit Factor':<14} {'Net R':<10}")
        for kz, data in full_report["per_killzone"].items():
            print(f"{kz:<14} {data['trade_count']:<8} {data['win_rate_pct']:<11.1f}% {data['expectancy_r']:<+13.3f}R {data['profit_factor']:<13.2f} {data['total_net_r']:<+9.2f}R")

    print("=" * 68 + "\n")
