import pandas as pd
import numpy as np
try:
    from core import config
except ImportError:
    import config

def get_killzone(dt_ny):
    """
    Returns the active killzone string ('LONDON', 'NY_AM') or None.
    dt_ny must be an America/New_York localized datetime or timestamp.
    """
    hour = dt_ny.hour
    minute = dt_ny.minute

    # London: 02:00 to 05:00 NY
    if (hour == 2 and minute >= 0) or (3 <= hour < 5) or (hour == 5 and minute == 0):
        return "LONDON"

    # NY AM: 07:00 to 10:00 NY
    if (hour == 7 and minute >= 0) or (8 <= hour < 10) or (hour == 10 and minute == 0):
        return "NY_AM"

    return None

def find_causal_swings(highs, lows, current_idx, swing_len=config.SWING_LEN):
    """
    Finds swing highs and lows strictly known as of current_idx.
    A swing at index i is confirmed only after swing_len candles have closed,
    i.e., i + swing_len <= current_idx.
    """
    swing_highs = []
    swing_lows = []
    max_pivot_idx = current_idx - swing_len

    if max_pivot_idx < swing_len:
        return swing_highs, swing_lows

    # Only need pivots that can participate in sweeps within lookback (max 60 bars ago)
    start_pivot_idx = max(swing_len, current_idx - 80)
    for i in range(start_pivot_idx, max_pivot_idx + 1):
        cur_hi = highs[i]
        cur_lo = lows[i]

        # Check if i is strictly the highest within [i - swing_len, i + swing_len]
        left_hi = highs[i - swing_len : i]
        right_hi = highs[i + 1 : i + swing_len + 1]
        if cur_hi > np.max(left_hi) and cur_hi >= np.max(right_hi):
            swing_highs.append({"index": i, "price": cur_hi})

        # Check if i is strictly the lowest within [i - swing_len, i + swing_len]
        left_lo = lows[i - swing_len : i]
        right_lo = lows[i + 1 : i + swing_len + 1]
        if cur_lo < np.min(left_lo) and cur_lo <= np.min(right_lo):
            swing_lows.append({"index": i, "price": cur_lo})

    return swing_highs, swing_lows

def evaluate_signal_at_bar(df, current_idx, swing_len=config.SWING_LEN, sl_buffer=config.SL_BUFFER, rr=config.RR, min_gap_pts=config.MIN_GAP_PTS):
    """
    Evaluates whether an ICT setup triggers at the CLOSE of candle current_idx.
    Strictly causal: uses ONLY df.iloc[0 : current_idx + 1].
    """
    if current_idx < 30:
        return None

    dt_ny = df.index[current_idx]
    kz = get_killzone(dt_ny)
    if kz is None:
        return None

    # Slice strictly up to current_idx
    sub_df = df.iloc[: current_idx + 1]
    highs = sub_df["high"].values
    lows = sub_df["low"].values
    closes = sub_df["close"].values
    opens = sub_df["open"].values
    n = len(sub_df)

    # 1. Causal Swing Points (confirmed <= current_idx)
    swing_highs, swing_lows = find_causal_swings(highs, lows, current_idx, swing_len=swing_len)
    if not swing_highs or not swing_lows:
        return None

    # 2. Look for Liquidity Sweeps in recent candles (e.g. past 25 candles)
    lookback_sweep = max(swing_len, n - 25)
    
    # ----------------- BULLISH SETUP EVALUATION -----------------
    # Check if a recent candle swept a confirmed swing low
    for s_idx in range(n - 1, lookback_sweep - 1, -1):
        s_low = lows[s_idx]
        s_close = closes[s_idx]

        for sl in swing_lows:
            if sl["index"] < s_idx and (s_idx - sl["index"]) <= 35:
                p_level = sl["price"]
                # Sweep condition: traded below p_level, closed at or above
                if s_low < p_level and s_close >= p_level:
                    sweep_low = s_low
                    
                    # Next: Check for MSS (close above prior swing high) between s_idx and current_idx
                    relevant_shs = [sh for sh in swing_highs if sh["index"] <= s_idx]
                    if not relevant_shs:
                        continue
                    pivot_sh = relevant_shs[-1]["price"]

                    mss_confirmed = False
                    mss_idx = None
                    for k in range(s_idx, n):
                        if closes[k] > pivot_sh:
                            mss_confirmed = True
                            mss_idx = k
                            break

                    if mss_confirmed and mss_idx is not None:
                        # Next: Check for Bullish 3-candle FVG formed during displacement (between s_idx and current_idx)
                        for c_idx in range(s_idx + 2, n):
                            c1_high = highs[c_idx - 2]
                            c3_low = lows[c_idx]
                            gap = c3_low - c1_high

                            if gap >= min_gap_pts:
                                fvg_top = c3_low
                                fvg_bot = c1_high
                                entry_ce = round((fvg_top + fvg_bot) / 2.0, 2)
                                
                                # Check mitigation: has price already filled past entry_ce before current_idx?
                                already_mitigated = False
                                if c_idx + 1 < n:
                                    if np.min(lows[c_idx + 1 : n]) <= entry_ce:
                                        already_mitigated = True

                                if not already_mitigated:
                                    # Setup confirmed at close of candle n-1
                                    stop_loss = round(sweep_low - sl_buffer, 2)
                                    risk_dist = abs(entry_ce - stop_loss)
                                    if risk_dist < config.MIN_SL:
                                        continue

                                    take_profit = round(entry_ce + rr * risk_dist, 2)
                                    return {
                                        "signal_bar_idx": current_idx,
                                        "signal_time": dt_ny,
                                        "killzone": kz,
                                        "direction": "BUY",
                                        "entry_price": entry_ce,
                                        "stop_loss": stop_loss,
                                        "take_profit": take_profit,
                                        "invalidation_price": sweep_low,
                                        "risk_distance": risk_dist,
                                        "fvg_top": fvg_top,
                                        "fvg_bot": fvg_bot,
                                        "sweep_idx": s_idx
                                    }

    # ----------------- BEARISH SETUP EVALUATION -----------------
    for s_idx in range(n - 1, lookback_sweep - 1, -1):
        s_high = highs[s_idx]
        s_close = closes[s_idx]

        for sh in swing_highs:
            if sh["index"] < s_idx and (s_idx - sh["index"]) <= 35:
                p_level = sh["price"]
                # Sweep condition: traded above p_level, closed at or below
                if s_high > p_level and s_close <= p_level:
                    sweep_high = s_high
                    
                    # Next: Check for MSS (close below prior swing low) between s_idx and current_idx
                    relevant_sls = [sl for sl in swing_lows if sl["index"] <= s_idx]
                    if not relevant_sls:
                        continue
                    pivot_sl = relevant_sls[-1]["price"]

                    mss_confirmed = False
                    mss_idx = None
                    for k in range(s_idx, n):
                        if closes[k] < pivot_sl:
                            mss_confirmed = True
                            mss_idx = k
                            break

                    if mss_confirmed and mss_idx is not None:
                        # Next: Check for Bearish 3-candle FVG formed during displacement
                        for c_idx in range(s_idx + 2, n):
                            c1_low = lows[c_idx - 2]
                            c3_high = highs[c_idx]
                            gap = c1_low - c3_high

                            if gap >= min_gap_pts:
                                fvg_top = c1_low
                                fvg_bot = c3_high
                                entry_ce = round((fvg_top + fvg_bot) / 2.0, 2)

                                already_mitigated = False
                                if c_idx + 1 < n:
                                    if np.max(highs[c_idx + 1 : n]) >= entry_ce:
                                        already_mitigated = True

                                if not already_mitigated:
                                    stop_loss = round(sweep_high + sl_buffer, 2)
                                    risk_dist = abs(stop_loss - entry_ce)
                                    if risk_dist < config.MIN_SL:
                                        continue

                                    take_profit = round(entry_ce - rr * risk_dist, 2)
                                    return {
                                        "signal_bar_idx": current_idx,
                                        "signal_time": dt_ny,
                                        "killzone": kz,
                                        "direction": "SELL",
                                        "entry_price": entry_ce,
                                        "stop_loss": stop_loss,
                                        "take_profit": take_profit,
                                        "invalidation_price": sweep_high,
                                        "risk_distance": risk_dist,
                                        "fvg_top": fvg_top,
                                        "fvg_bot": fvg_bot,
                                        "sweep_idx": s_idx
                                    }

    return None

def generate_signals(df):
    """
    Iterates through historical bars and yields causal signals.
    """
    signals = []
    n = len(df)
    for i in range(30, n):
        sig = evaluate_signal_at_bar(df, i)
        if sig is not None:
            signals.append(sig)
    return signals
