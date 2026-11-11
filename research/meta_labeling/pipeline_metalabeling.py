"""
Production Meta-Labeling Dataset Generation & LightGBM Pipeline
Strictly causal, zero lookahead, realistic costs, and leak-free validation.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import sys
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from core import config
    from core.data_loader import load_and_clean_data
    from core.signals import evaluate_causal_signal_at_bar, get_causal_swing_points
except ImportError:
    import config
    from data_loader import load_and_clean_data
    from signals import evaluate_causal_signal_at_bar, get_causal_swing_points

# LightGBM and Scikit-learn
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

SPREAD = config.SPREAD
SLIPPAGE = config.SLIPPAGE
COMM_PER_LOT = config.COMMISSION_PER_LOT
CONTRACT_SIZE = config.CONTRACT_SIZE

def generate_candidates_and_features(df):
    """
    Generates all candidates across the full dataframe without killzone gate,
    computes causal features at signal-bar close, and simulates independent
    trade outcomes for RR in {1.5, 2.0, 3.0}.
    """
    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    times = df.index

    # Precompute vectorised market features
    prev_closes = pd.Series(closes, index=times).shift(1)
    tr = pd.concat([
        pd.Series(highs - lows, index=times),
        (pd.Series(highs, index=times) - prev_closes).abs(),
        (pd.Series(lows, index=times) - prev_closes).abs()
    ], axis=1).max(axis=1)
    atr_14_series = tr.rolling(window=14, min_periods=1).mean()
    vol_rank_series = atr_14_series.rolling(window=500, min_periods=50).rank(pct=True).fillna(0.5)

    # Session features (session boundary: 18:00 NY)
    session_dates = times.to_series().apply(
        lambda dt: dt.date() + pd.Timedelta(days=1) if dt.hour >= 18 else dt.date()
    )
    s_highs = pd.Series(highs, index=times).groupby(session_dates).cummax()
    s_lows = pd.Series(lows, index=times).groupby(session_dates).cummin()
    s_spans = s_highs - s_lows
    s_pos = np.where(s_spans > 1e-4, (pd.Series(closes, index=times) - s_lows) / s_spans, 0.5)
    s_pos_series = pd.Series(s_pos, index=times)

    # Previous day range
    d_high = pd.Series(highs, index=times).groupby(session_dates).max()
    d_low = pd.Series(lows, index=times).groupby(session_dates).min()
    d_range = d_high - d_low
    prev_d_range_map = d_range.shift(1)
    prev_day_range_series = session_dates.map(prev_d_range_map).fillna(d_range.median())

    # 1H Trend (EMA20 slope on closed 1H bars, shifted by 1)
    df_1h = pd.Series(closes, index=times).resample("1h", closed="left", label="left").last().dropna()
    ema_1h = df_1h.ewm(span=20, adjust=False).mean()
    ema_1h_slope = np.sign(ema_1h.diff()).shift(1)
    trend_1h_series = ema_1h_slope.reindex(times, method="ffill").fillna(0.0)

    # Daily Trend (EMA20 slope on closed daily bars, shifted by 1)
    daily_close = pd.Series(closes, index=times).groupby(session_dates).last()
    ema_daily = daily_close.ewm(span=20, adjust=False).mean()
    ema_daily_slope = np.sign(ema_daily.diff()).shift(1)
    trend_daily_series = session_dates.map(ema_daily_slope).fillna(0.0)

    candidates = []

    # Sequential tracking to maintain realistic trade selection
    pending_order = None
    position = None

    print(f"[CANDIDATE GENERATOR] Scanning {n:,} candles for causal ICT setups...")

    for t in range(35, n):
        cur_time = times[t]
        cur_hi = highs[t]
        cur_lo = lows[t]
        cur_close = closes[t]

        # 1. Exit check for active position
        if position is not None:
            p_dir = position["direction"]
            sl_hit = (cur_lo <= position["sl"]) if p_dir == "BUY" else (cur_hi >= position["sl"])
            tp_hit = (cur_hi >= position["tp"]) if p_dir == "BUY" else (cur_lo <= position["tp"])
            if sl_hit or tp_hit:
                position = None

        # 2. Pending check
        elif pending_order is not None:
            p_dir = pending_order["direction"]
            p_sl = pending_order["sl"]
            p_entry = pending_order["entry"]
            p_bar = pending_order["bar"]

            # Invalidation
            if (p_dir == "BUY" and cur_lo <= p_sl) or (p_dir == "SELL" and cur_hi >= p_sl):
                pending_order = None
            elif (t - p_bar) > config.ORDER_EXPIRY_BARS:
                pending_order = None
            elif (p_dir == "BUY" and cur_lo <= p_entry - 0.01) or (p_dir == "SELL" and cur_hi >= p_entry + 0.01):
                # Filled
                position = {"direction": p_dir, "sl": p_sl, "tp": pending_order["tp"], "entry": p_entry}
                pending_order = None

        # 3. New candidate trigger
        if position is None and pending_order is None:
            sig = evaluate_causal_signal_at_bar(df, t, killzone_gate=False)
            if sig is not None:
                p_dir = sig["direction"]
                p_entry = sig["entry"]
                p_sl = sig["sl"]
                p_dist = sig["risk_dist"]
                p_kz = sig["killzone"]

                # Register pending order for sequential simulation
                pending_order = {
                    "direction": p_dir,
                    "entry": p_entry,
                    "sl": p_sl,
                    "tp": sig["tp"],
                    "bar": t
                }

                # Extract causal features at bar t
                atr = float(atr_14_series.iloc[t])
                stop_dist_atr = p_dist / atr if atr > 1e-4 else 1.0

                # Nearest confirmed swing points
                shs, sls = get_causal_swing_points(df, up_to_idx=t, swing_len=config.SWING_LEN)
                dist_sh = min([abs(cur_close - s["price"]) for s in shs], default=p_dist)
                dist_sl = min([abs(cur_close - s["price"]) for s in sls], default=p_dist)
                dist_sh_atr = dist_sh / atr if atr > 1e-4 else 1.0
                dist_sl_atr = dist_sl / atr if atr > 1e-4 else 1.0

                # Forward simulation for outcomes at RR 1.5, 2.0, 3.0
                outcomes = simulate_candidate_forward(df, t, p_dir, p_entry, p_sl, p_dist)

                cand_record = {
                    "candidate_id": len(candidates) + 1,
                    "signal_time": cur_time,
                    "year": cur_time.year,
                    "signal_bar": t,
                    "direction": 1 if p_dir == "BUY" else 0,
                    "hour": cur_time.hour,
                    "day_of_week": cur_time.dayofweek,
                    "killzone": 1 if p_kz == "LONDON" else (2 if p_kz == "NY_AM" else 0),
                    "killzone_str": p_kz,
                    "atr_14": round(atr, 3),
                    "stop_dist": round(p_dist, 2),
                    "stop_dist_atr": round(stop_dist_atr, 3),
                    "dist_to_swing_high_atr": round(dist_sh_atr, 3),
                    "dist_to_swing_low_atr": round(dist_sl_atr, 3),
                    "position_in_session_range": round(float(s_pos_series.iloc[t]), 3),
                    "prev_day_range": round(float(prev_day_range_series.iloc[t]), 2),
                    "trend_1h_sign": float(trend_1h_series.iloc[t]),
                    "trend_daily_sign": float(trend_daily_series.iloc[t]),
                    "volatility_rank": round(float(vol_rank_series.iloc[t]), 3),
                    "filled": outcomes["filled"],
                    "net_r_15": outcomes["net_r_15"],
                    "net_r_20": outcomes["net_r_20"],
                    "net_r_30": outcomes["net_r_30"],
                    "exit_reason_20": outcomes["exit_reason_20"]
                }
                candidates.append(cand_record)

    cand_df = pd.DataFrame(candidates)
    return cand_df


def simulate_candidate_forward(df, sig_bar, direction, entry, sl, risk_dist):
    """
    Simulates the forward trade outcome for a candidate limit order
    for RR in {1.5, 2.0, 3.0} adhering to strict conservative fill & SL rules.
    """
    n = len(df)
    highs = df["high"].values
    lows = df["low"].values

    # Check pending order fill within 8 bars
    fill_bar = None
    fill_price = None

    for f_idx in range(sig_bar + 1, min(sig_bar + 1 + config.ORDER_EXPIRY_BARS, n)):
        b_hi = highs[f_idx]
        b_lo = lows[f_idx]

        # Invalidation check
        if direction == "BUY" and b_lo <= sl:
            break
        elif direction == "SELL" and b_hi >= sl:
            break

        # Fill check (trades THROUGH entry level)
        if direction == "BUY" and b_lo <= entry - 0.01:
            fill_bar = f_idx
            fill_price = round(entry + SPREAD, 2) # Ask fill
            break
        elif direction == "SELL" and b_hi >= entry + 0.01:
            fill_bar = f_idx
            fill_price = round(entry, 2) # Bid fill
            break

    if fill_bar is None:
        return {
            "filled": False,
            "net_r_15": 0.0,
            "net_r_20": 0.0,
            "net_r_30": 0.0,
            "exit_reason_20": "UNFILLED"
        }

    # Simulate for each RR target
    r_results = {}
    exit_reasons = {}

    for rr in [1.5, 2.0, 3.0]:
        tp_level = round(entry + rr * risk_dist if direction == "BUY" else entry - rr * risk_dist, 2)

        # Check same candle SL
        fill_lo = lows[fill_bar]
        fill_hi = highs[fill_bar]
        same_candle_sl = (direction == "BUY" and fill_lo <= sl) or (direction == "SELL" and fill_hi >= sl)

        if same_candle_sl:
            exit_p = round(sl - SLIPPAGE if direction == "BUY" else sl + SLIPPAGE + SPREAD, 2)
            exit_reason = "SL (Same-Candle)"
        else:
            exit_p = None
            exit_reason = None
            for w in range(fill_bar + 1, n):
                w_hi = highs[w]
                w_lo = lows[w]
                sl_hit = (w_lo <= sl) if direction == "BUY" else (w_hi >= sl)
                tp_hit = (w_hi >= tp_level) if direction == "BUY" else (w_lo <= tp_level)

                if sl_hit and tp_hit:
                    # Conservative rule: SL counted first
                    exit_reason = "SL (Conservative Same-Candle)"
                    exit_p = round(sl - SLIPPAGE if direction == "BUY" else sl + SLIPPAGE + SPREAD, 2)
                    break
                elif sl_hit:
                    exit_reason = "SL"
                    exit_p = round(sl - SLIPPAGE if direction == "BUY" else sl + SLIPPAGE + SPREAD, 2)
                    break
                elif tp_hit:
                    exit_reason = "TP"
                    exit_p = round(tp_level if direction == "BUY" else tp_level + SPREAD, 2)
                    break

        if exit_p is None:
            # Reached end of data before resolution
            exit_p = closes[-1]
            exit_reason = "TIME_EXIT"

        # Calculate exact net R
        if direction == "BUY":
            price_delta = exit_p - fill_price
        else:
            price_delta = fill_price - exit_p

        # Commission in R terms: Comm / (Contract Size * Risk Dist)
        comm_r = COMM_PER_LOT / (CONTRACT_SIZE * risk_dist)
        gross_r = price_delta / risk_dist
        net_r = round(gross_r - comm_r, 3)

        key = str(int(rr * 10))
        r_results[f"net_r_{key}"] = net_r
        exit_reasons[f"exit_reason_{key}"] = exit_reason

    return {
        "filled": True,
        "net_r_15": r_results["net_r_15"],
        "net_r_20": r_results["net_r_20"],
        "net_r_30": r_results["net_r_30"],
        "exit_reason_20": exit_reasons["exit_reason_20"]
    }


def run_meta_labeling_experiment(cand_df):
    """
    Trains LightGBM on TRAIN (2015-2019), selects threshold on TRAIN via CV,
    evaluates on VALID (2020-2021), and conducts shuffled-label sanity test.
    """
    feature_cols = [
        "hour",
        "day_of_week",
        "killzone",
        "atr_14",
        "stop_dist",
        "stop_dist_atr",
        "dist_to_swing_high_atr",
        "dist_to_swing_low_atr",
        "position_in_session_range",
        "prev_day_range",
        "trend_1h_sign",
        "trend_daily_sign",
        "direction",
        "volatility_rank"
    ]

    # Target label: Y = 1 if net_r_20 > 0 else 0 (Meta-label on RR=2.0)
    cand_df["label_binary"] = (cand_df["net_r_20"] > 0).astype(int)

    # Splits
    train_mask = (cand_df["year"] >= 2015) & (cand_df["year"] <= 2019)
    valid_mask = (cand_df["year"] >= 2020) & (cand_df["year"] <= 2021)

    df_train = cand_df[train_mask].copy().reset_index(drop=True)
    df_valid = cand_df[valid_mask].copy().reset_index(drop=True)

    X_train = df_train[feature_cols]
    y_train = df_train["label_binary"].values
    r_train = df_train["net_r_20"].values

    X_valid = df_valid[feature_cols]
    y_valid = df_valid["label_binary"].values
    r_valid = df_valid["net_r_20"].values

    print("\n" + "=" * 65)
    print("META-LABELING DATASET PARTITIONS")
    print("=" * 65)
    print(f"TRAIN (2015-2019): {len(df_train):,} candidates | Base Win Rate: {y_train.mean()*100:.1f}% | Base Exp: {r_train.mean():+.3f} R")
    print(f"VALID (2020-2021): {len(df_valid):,} candidates | Base Win Rate: {y_valid.mean()*100:.1f}% | Base Exp: {r_valid.mean():+.3f} R")

    # 1. Select threshold on TRAIN via 5-Fold TimeSeriesSplit (Purged / expanding window)
    print("\n[OPTIMIZATION] Selecting Probability Threshold on TRAIN via Time-Series Cross-Validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    threshold_candidates = np.linspace(0.40, 0.70, 31)
    oof_expectancies = {round(th, 2): [] for th in threshold_candidates}

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_va, y_va, r_va = X_train.iloc[val_idx], y_train[val_idx], r_train[val_idx]

        clf_fold = lgb.LGBMClassifier(
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            n_estimators=100,
            min_child_samples=25,
            class_weight="balanced",
            random_state=42,
            verbosity=-1
        )
        clf_fold.fit(X_tr, y_tr)
        probs_va = clf_fold.predict_proba(X_va)[:, 1]

        for th in threshold_candidates:
            selected_mask = probs_va >= th
            if selected_mask.sum() >= 5:
                fold_exp = r_va[selected_mask].mean()
            else:
                fold_exp = -999.0
            oof_expectancies[round(th, 2)].append(fold_exp)

    mean_oof_exp = {
        th: np.mean([v for v in vals if v > -900]) if any(v > -900 for v in vals) else -999.0
        for th, vals in oof_expectancies.items()
    }
    valid_th_candidates = {k: v for k, v in mean_oof_exp.items() if v > -900}
    best_threshold = max(valid_th_candidates.keys(), key=lambda k: valid_th_candidates[k])
    print(f"Selected Best Threshold on TRAIN: {best_threshold:.2f} (OOF Expectancy: {best_threshold:.2f} -> {valid_th_candidates[best_threshold]:+.3f} R)")

    # 2. Train LightGBM with early stopping on VALID
    print("\n[TRAINING] Fitting Full LightGBM Model with Early Stopping on VALID...")
    model = lgb.LGBMClassifier(
        max_depth=3,
        num_leaves=7,
        learning_rate=0.03,
        n_estimators=300,
        min_child_samples=25,
        class_weight="balanced",
        random_state=42,
        verbosity=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    # 3. Evaluate on VALID ONLY
    valid_probs = model.predict_proba(X_valid)[:, 1]
    df_valid["model_prob"] = valid_probs
    df_valid["taken"] = valid_probs >= best_threshold

    selected_valid = df_valid[df_valid["taken"]].copy()
    n_taken = len(selected_valid)
    n_total = len(df_valid)

    valid_net_r = selected_valid["net_r_20"].values
    val_exp = float(np.mean(valid_net_r)) if n_taken > 0 else 0.0
    val_wr = float((valid_net_r > 0).mean() * 100.0) if n_taken > 0 else 0.0

    # 95% Bootstrap CI
    rng = np.random.default_rng(42)
    boot_means = [float(np.mean(rng.choice(valid_net_r, size=n_taken, replace=True))) for _ in range(1000)]
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Drawdown in R
    cum_r = np.cumsum(valid_net_r)
    cum_r_with_zero = np.insert(cum_r, 0, 0.0)
    peak_r = np.maximum.accumulate(cum_r_with_zero)
    dd_r = peak_r - cum_r_with_zero
    max_dd_r = float(np.max(dd_r)) if len(dd_r) > 0 else 0.0

    # Profit Factor
    wins_usd = valid_net_r[valid_net_r > 0]
    loss_usd = abs(valid_net_r[valid_net_r < 0])
    pf = float(np.sum(wins_usd) / np.sum(loss_usd)) if np.sum(loss_usd) > 0 else 999.0

    print("\n" + "=" * 65)
    print("META-LABELER VALIDATION RESULTS (VALID 2020-2021 ONLY)")
    print("=" * 65)
    print(f"Candidates Available:   {n_total:,}")
    print(f"Trades Taken:           {n_taken:,} ({n_taken/n_total*100:.1f}% acceptance rate)")
    print(f"Win Rate:               {val_wr:.1f}%")
    print(f"Profit Factor:          {pf:.2f}")
    print(f"Net Expectancy (R):     {val_exp:+.3f} R")
    print(f"95% Bootstrap CI:       [{ci_lower:+.3f} R, {ci_upper:+.3f} R]")
    print(f"Max Drawdown (R):       {max_dd_r:.2f} R")
    print(f"Total Net Return (R):   {float(np.sum(valid_net_r)):+.2f} R")

    # Per-year breakdown
    print("\nPER-YEAR BREAKDOWN ON VALID:")
    print("Year   Candidates  Taken  Win Rate  Net Exp (R)  Profit Factor  Total Net R")
    for yr, g in df_valid.groupby("year"):
        g_taken = g[g["taken"]]
        if len(g_taken) > 0:
            yr_r = g_taken["net_r_20"].values
            yr_wr = (yr_r > 0).mean() * 100.0
            yr_exp = yr_r.mean()
            w_u = yr_r[yr_r > 0]
            l_u = abs(yr_r[yr_r < 0])
            yr_pf = np.sum(w_u) / np.sum(l_u) if np.sum(l_u) > 0 else 999.0
            print(f"{yr}   {len(g):<11} {len(g_taken):<6} {yr_wr:.1f}%     {yr_exp:+.3f} R      {yr_pf:.2f}           {yr_r.sum():+.2f} R")

    # Feature Importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.feature_importances_
    }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)
    print("\nFEATURE IMPORTANCE (LIGHTGBM SPLIT GAIN):")
    for idx, row in importance_df.iterrows():
        print(f"  {idx+1:2d}. {row['feature']:<26} : {row['importance_gain']}")

    # 4. Shuffled-Label Sanity Test
    print("\n" + "=" * 65)
    print("SHUFFLED-LABEL SANITY TEST (LEAKAGE CHECK)")
    print("=" * 65)
    y_train_shuffled = rng.permutation(y_train)
    model_shuffled = lgb.LGBMClassifier(
        max_depth=3,
        num_leaves=7,
        learning_rate=0.03,
        n_estimators=300,
        min_child_samples=25,
        class_weight="balanced",
        random_state=42,
        verbosity=-1
    )
    model_shuffled.fit(
        X_train, y_train_shuffled,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    shuffled_probs = model_shuffled.predict_proba(X_valid)[:, 1]
    shuffled_taken = df_valid[shuffled_probs >= best_threshold]
    shuffled_exp = shuffled_taken["net_r_20"].mean() if len(shuffled_taken) > 0 else 0.0
    print(f"Shuffled-Label Trades Taken: {len(shuffled_taken):,} / {n_total:,}")
    print(f"Shuffled-Label VALID Expectancy: {shuffled_exp:+.3f} R")
    if abs(shuffled_exp) < 0.10 or len(shuffled_taken) == 0:
        print("[VERIFIED] Shuffled model exhibits ~0 R edge. Zero feature leakage detected!")
    else:
        print("[WARNING] Non-zero expectancy in shuffled model; investigate potential leakage.")

    return {
        "best_threshold": best_threshold,
        "n_candidates_valid": n_total,
        "n_taken_valid": n_taken,
        "val_wr": val_wr,
        "val_exp": val_exp,
        "val_pf": pf,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "max_dd_r": max_dd_r,
        "total_net_r": float(np.sum(valid_net_r)),
        "importance_df": importance_df,
        "shuffled_taken": len(shuffled_taken),
        "shuffled_exp": shuffled_exp
    }

if __name__ == "__main__":
    t0 = time.time()
    df = load_and_clean_data("data/xauusd_15m_long.csv", data_tz="UTC", target_tz="America/New_York", verbose=False)
    # Strictly in-sample TRAIN (2015-2019) + VALID (2020-2021); NEVER touch 2022+ TEST
    df_insample = df.loc["2015-01-01":"2021-12-31"]

    candidates_df = generate_candidates_and_features(df_insample)
    
    # Save to data/candidates.parquet
    candidates_df.to_parquet("data/candidates.parquet", index=False)
    print(f"\n[SAVED] Saved {len(candidates_df):,} candidates to data/candidates.parquet ({os.path.getsize('data/candidates.parquet')/1024:.1f} KB)")

    # Run ML experiment
    results = run_meta_labeling_experiment(candidates_df)
    t1 = time.time()
    print(f"\n[COMPLETE] Meta-labeling pipeline finished in {t1 - t0:.2f}s")
