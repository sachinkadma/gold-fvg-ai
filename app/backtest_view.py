"""
Backtest Analytics View Component for Streamlit.
Purely read-only: displays metrics.json, trades.csv, and equity_curve.png.
Zero heavy computation inside Streamlit.
"""

import os
import json
import pandas as pd
import streamlit as st


def render_backtest_tab():
    st.markdown("### 📊 Institutional ICT Backtesting Analytics (Zero Lookahead)")
    st.warning("⚠️ **Visual aid only.** Validated 2015-2021 baseline (684 trades) shows -0.171R net expectancy after costs.")

    def find_artifact(filename):
        candidates = [
            os.path.join("research", "meta_labeling", filename),
            filename,
            os.path.join("..", "research", "meta_labeling", filename),
            os.path.join("..", filename)
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return filename

    metrics_path = find_artifact("metrics.json")
    trades_path = find_artifact("trades.csv")
    curve_path = find_artifact("equity_curve.png")

    if not os.path.exists(metrics_path) or not os.path.exists(trades_path):
        st.warning("⚠️ No historical backtest runs found on disk. Run backtest from CLI:")
        st.code("python backtester.py --split train", language="bash")
        return

    # Load pre-computed artifacts
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        df_trades = pd.read_csv(trades_path)
    except Exception as e:
        st.error(f"Error loading backtest artifacts: {e}")
        return

    split_name = metrics.get("split", "UNKNOWN").upper()
    is_test = "TEST" in split_name
    banner_bg = "linear-gradient(135deg, #1E1B4B, #311042)" if is_test else "linear-gradient(135deg, #064E3B, #0F2E28)"
    badge_col = "#EC4899" if is_test else "#10B981"
    split_type_badge = "LOCKED TEST SET (OUT-OF-SAMPLE)" if is_test else "TRAIN SET (IN-SAMPLE EXPLORATION)"

    costs = metrics.get("cost_assumptions", {})
    spread = costs.get("spread", 0.35)
    slippage = costs.get("slippage", 0.10)
    comm = costs.get("commission_per_lot", 7.0)
    min_sl = costs.get("min_sl", 3.0)
    rr = costs.get("rr", 2.0)
    swing_len = costs.get("swing_len", 5)

    # 1. VISIBLE PARTITION & COST AUDIT BANNER
    st.markdown(f"""
    <div style="background: {banner_bg}; padding: 18px 22px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">
                SPLIT EVALUATION: <span style="color: {badge_col};">{split_type_badge}</span>
            </span>
            <span style="background: {badge_col}; color: #FFFFFF; font-weight: 800; font-size: 0.75rem; padding: 4px 10px; border-radius: 6px;">
                {split_name}
            </span>
        </div>
        <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.6;">
            <b>Rigorous Cost Modeling:</b> Spread: <code>{spread} pts</code> (Buys fill at Ask, Sells at Bid) | 
            Stop Slippage: <code>{slippage} pts</code> | Commission: <code>${comm}/lot round-turn</code> | 
            Min Stop Distance: <code>{min_sl} pts</code> | Target R:R: <code>1:{rr}</code> | Swing Confirmation: <code>{swing_len} bars</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ov = metrics.get("overall", {})

    if ov.get("sample_size_warning"):
        st.warning(f"⚠️ **Sample Size Alert:** Total trades ({ov.get('trade_count', 0)}) < 100. Statistical significance is weak.")

    # 2. EXECUTIVE METRIC KPI TILES
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Trades", f"{ov.get('trade_count', 0)}", f"{ov.get('win_count', 0)}W / {ov.get('loss_count', 0)}L")
    with col2:
        st.metric("Win Rate", f"{ov.get('win_rate_pct', 0.0):.1f}%", f"PF: {ov.get('profit_factor', 0.0):.2f}")
    with col3:
        exp_r = ov.get("expectancy_r", 0.0)
        ci = ov.get("expectancy_ci_95", [0.0, 0.0])
        st.metric("Expectancy (R)", f"{exp_r:+.3f} R", f"95% CI: [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    with col4:
        st.metric("Max Drawdown", f"{ov.get('max_drawdown_r', 0.0):.2f} R", f"-{ov.get('max_drawdown_pct', 0.0):.1f}% Eq")
    with col5:
        st.metric("Net PnL", f"${ov.get('total_net_usd', 0.0):+,.2f}", f"{ov.get('total_net_r', 0.0):+.2f} R")

    st.markdown("---")

    # 3. HIGH RESOLUTION EQUITY CURVE
    col_chart, col_breakdown = st.columns([1.3, 1.0])
    with col_chart:
        st.markdown("#### 📈 Cumulative Realized Edge (in R-multiples)")
        if os.path.exists(curve_path):
            st.image(curve_path, caption="Equity Curve (Conservative Execution Costs & Instant Fill-Candle SL Priority)", use_container_width=True)
        else:
            st.info("Equity curve image not found.")

    with col_breakdown:
        st.markdown("#### 📅 Performance Breakdown")
        breakdown_tab = st.tabs(["By Year", "Long vs Short", "By Killzone"])

        # Per Year Table
        with breakdown_tab[0]:
            py = metrics.get("per_year", {})
            if py:
                py_rows = []
                for yr, data in py.items():
                    ci_str = f"[{data.get('expectancy_ci_95', [0,0])[0]:+.2f}, {data.get('expectancy_ci_95', [0,0])[1]:+.2f}]"
                    py_rows.append({
                        "Year": yr,
                        "Trades": data.get("trade_count", 0),
                        "Win Rate": f"{data.get('win_rate_pct', 0.0):.1f}%",
                        "Profit Factor": f"{data.get('profit_factor', 0.0):.2f}",
                        "Expectancy": f"{data.get('expectancy_r', 0.0):+.3f} R",
                        "95% CI": ci_str,
                        "Net (USD)": f"${data.get('total_net_usd', 0.0):+,.2f}",
                        "Net (R)": f"{data.get('total_net_r', 0.0):+.2f} R"
                    })
                st.dataframe(pd.DataFrame(py_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No yearly data.")

        # Long vs Short Table
        with breakdown_tab[1]:
            pd_dir = metrics.get("per_direction", {})
            if pd_dir:
                dir_rows = []
                for d_label, data in pd_dir.items():
                    ci_str = f"[{data.get('expectancy_ci_95', [0,0])[0]:+.2f}, {data.get('expectancy_ci_95', [0,0])[1]:+.2f}]"
                    dir_rows.append({
                        "Direction": d_label,
                        "Trades": data.get("trade_count", 0),
                        "Win Rate": f"{data.get('win_rate_pct', 0.0):.1f}%",
                        "Profit Factor": f"{data.get('profit_factor', 0.0):.2f}",
                        "Expectancy": f"{data.get('expectancy_r', 0.0):+.3f} R",
                        "95% CI": ci_str,
                        "Net (USD)": f"${data.get('total_net_usd', 0.0):+,.2f}",
                        "Net (R)": f"{data.get('total_net_r', 0.0):+.2f} R"
                    })
                st.dataframe(pd.DataFrame(dir_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No direction data.")

        # Per Killzone Table
        with breakdown_tab[2]:
            pk = metrics.get("per_killzone", {})
            if pk:
                pk_rows = []
                for kz, data in pk.items():
                    ci_str = f"[{data.get('expectancy_ci_95', [0,0])[0]:+.2f}, {data.get('expectancy_ci_95', [0,0])[1]:+.2f}]"
                    pk_rows.append({
                        "Killzone": kz,
                        "Trades": data.get("trade_count", 0),
                        "Win Rate": f"{data.get('win_rate_pct', 0.0):.1f}%",
                        "Profit Factor": f"{data.get('profit_factor', 0.0):.2f}",
                        "Expectancy": f"{data.get('expectancy_r', 0.0):+.3f} R",
                        "95% CI": ci_str,
                        "Net (USD)": f"${data.get('total_net_usd', 0.0):+,.2f}",
                        "Net (R)": f"{data.get('total_net_r', 0.0):+.2f} R"
                    })
                st.dataframe(pd.DataFrame(pk_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No killzone data.")

    # Buy and Hold Benchmark
    bh = metrics.get("buy_and_hold_benchmark")
    if bh:
        st.info(
            f"🪙 **Buy-and-Hold Gold Benchmark (Context):** Over this same TRAIN period, Gold spot moved from "
            f"**${bh.get('start_price', 0):,.2f}** to **${bh.get('end_price', 0):,.2f}** "
            f"({bh.get('price_change_pts', 0):+,.2f} USD/oz, **{bh.get('buy_and_hold_return_pct', 0):+.2f}%** return)."
        )

    st.markdown("---")

    # 4. FILTERABLE HISTORICAL TRADE LOG
    st.markdown("#### 📜 Filterable Institutional Trade Log")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    years_avail = ["All"] + sorted(list(df_trades["year"].astype(str).unique())) if "year" in df_trades.columns else ["All"]
    with f_col1:
        f_yr = st.selectbox("Filter Year", years_avail)

    kz_avail = ["All"] + sorted(list(df_trades["killzone"].astype(str).unique())) if "killzone" in df_trades.columns else ["All"]
    with f_col2:
        f_kz = st.selectbox("Filter Killzone", kz_avail)

    with f_col3:
        f_dir = st.selectbox("Direction", ["All", "BUY", "SELL"])

    with f_col4:
        f_res = st.selectbox("Result Outcome", ["All", "Wins Only", "Losses Only"])

    # Apply filters
    filtered_df = df_trades.copy()
    if f_yr != "All" and "year" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["year"].astype(str) == f_yr]
    if f_kz != "All" and "killzone" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["killzone"] == f_kz]
    if f_dir != "All" and "direction" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["direction"] == f_dir]
    if f_res == "Wins Only":
        filtered_df = filtered_df[filtered_df["result_r"] > 0]
    elif f_res == "Losses Only":
        filtered_df = filtered_df[filtered_df["result_r"] < 0]

    st.caption(f"Showing **{len(filtered_df)}** of **{len(df_trades)}** total historical trades")

    # Format display columns
    display_cols = [c for c in [
        "entry_time", "exit_time", "direction", "killzone",
        "entry_price", "sl_price", "tp_price", "exit_price",
        "lot_size", "result_r", "result_usd", "exit_reason",
        "duration_bars", "equity_after"
    ] if c in filtered_df.columns]

    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True
    )

    # Download CSV button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Trades (CSV)",
        data=csv_bytes,
        file_name=f"xauusd_trades_{split_name.lower()}.csv",
        mime="text/csv"
    )
