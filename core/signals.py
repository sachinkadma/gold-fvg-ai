"""
Causal Strategy Signals module for XAUUSD 15m ICT Strategy.
Strictly causal: no future data is accessed. Swings at index i are usable
only after i + SWING_LEN candles have closed. Signals are evaluated at candle close.
"""

from datetime import time
import numpy as np
import pandas as pd
try:
    from core import config
except ImportError:
    import config


def is_in_killzone(ts_time, killzones=None):
    """
    Checks if a timestamp time falls inside any defined killzone.
    London: 02:00 <= t < 05:00 NY
    NY AM:  07:00 <= t < 10:00 NY
    """
    kz_dict = killzones or config.KILLZONES
    for kz_name, (kz_start, kz_end) in kz_dict.items():
        if kz_start <= ts_time < kz_end:
            return kz_name
    return None


def get_causal_swing_points(df, up_to_idx, swing_len=None):
    """
    Returns swing highs and lows that are confirmed and usable at or before up_to_idx.
    A swing at index i is confirmed only after i + swing_len candles have closed,
    i.e., i <= up_to_idx - swing_len.
    """
    s_len = swing_len or config.SWING_LEN
    max_confirmed_i = up_to_idx - s_len
    if max_confirmed_i < s_len:
        return [], []

    highs = df["high"].values
    lows = df["low"].values

    swing_highs = []
    swing_lows = []

    # Only evaluate candidate swings within relevant lookback up to max_confirmed_i
    start_candidate = max(s_len, max_confirmed_i - 120)
    for i in range(start_candidate, max_confirmed_i + 1):
        cur_hi = highs[i]
        cur_lo = lows[i]

        # Check left and right windows
        left_hi = highs[i - s_len : i]
        right_hi = highs[i + 1 : i + s_len + 1]

        left_lo = lows[i - s_len : i]
        right_lo = lows[i + 1 : i + s_len + 1]

        if cur_hi > np.max(left_hi) and cur_hi >= np.max(right_hi):
            swing_highs.append({
                "index": i,
                "price": round(float(cur_hi), 2)
            })

        if cur_lo < np.min(left_lo) and cur_lo <= np.min(right_lo):
            swing_lows.append({
                "index": i,
                "price": round(float(cur_lo), 2)
            })

    return swing_highs, swing_lows


def check_new_causal_swing_at_bar(highs, lows, t, swing_len=5):
    """
    Checks if candidate candle i = t - swing_len is confirmed as a swing high/low at bar t.
    Strictly causal: uses only candles in [i - swing_len, t].
    Returns: (new_sh, new_sl)
    """
    i = t - swing_len
    if i < swing_len:
        return None, None

    cur_hi = highs[i]
    cur_lo = lows[i]

    new_sh = None
    new_sl = None

    if cur_hi > np.max(highs[i - swing_len : i]) and cur_hi >= np.max(highs[i + 1 : t + 1]):
        new_sh = {"index": i, "price": round(float(cur_hi), 2)}

    if cur_lo < np.min(lows[i - swing_len : i]) and cur_lo <= np.min(lows[i + 1 : t + 1]):
        new_sl = {"index": i, "price": round(float(cur_lo), 2)}

    return new_sh, new_sl


def evaluate_causal_signal_at_bar(
    df,
    t,
    swing_len=None,
    sl_buffer=None,
    rr=None,
    min_sl=None,
    fvg_min_gap=None,
    swing_highs=None,
    swing_lows=None,
    killzone_gate=True
):
    """
    Evaluates whether a valid ICT trade setup is triggered at the close of candle t.
    Uses ONLY candles up to index t (inclusive).

    Returns:
        dict or None: Signal dict with entry, SL, TP, direction, killzone, etc., or None.
    """
    s_len = swing_len or config.SWING_LEN
    buffer = sl_buffer if sl_buffer is not None else config.SL_BUFFER
    rr_ratio = rr if rr is not None else config.RR
    min_stop = min_sl if min_sl is not None else config.MIN_SL
    min_gap = fvg_min_gap if fvg_min_gap is not None else config.FVG_MIN_GAP

    # 1. Killzone gate
    bar_time = df.index[t].time()
    kz_name = is_in_killzone(bar_time)
    if killzone_gate and not kz_name:
        return None
    kz_name = kz_name or "OTHER"

    # Need enough lookback history
    min_history = s_len * 2 + 10
    if t < min_history:
        return None

    # 2. Get causal swing points confirmed by bar t
    if swing_highs is None or swing_lows is None:
        swing_highs, swing_lows = get_causal_swing_points(df, up_to_idx=t, swing_len=s_len)

    if not swing_highs and not swing_lows:
        return None

    # 3. Detect sweeps in recent candles
    # Look back up to 30 candles from t, but not before the earliest confirmed swing
    recent_lookback = min(35, t)
    start_search = max(0, t - recent_lookback)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values

    bullish_sweep = None
    bearish_sweep = None

    # Check SSL sweep (bullish setup)
    if swing_lows:
        for sl in swing_lows:
            sl_idx = sl["index"]
            sl_level = sl["price"]
            if sl_idx < start_search:
                # Still check if swept recently
                pass
            # Look for sweep between max(start_search, sl_idx + 1) and t
            for s in range(max(start_search, sl_idx + 1), t):
                # Candle dipped below swing low
                if lows[s] < sl_level and closes[s] >= sl_level - 1.0: # Sweep with rejection
                    # Found an SSL sweep
                    bullish_sweep = {
                        "sweep_idx": s,
                        "level": sl_level,
                        "sweep_extreme": float(np.min(lows[s : t + 1]))
                    }

    # Check BSL sweep (bearish setup)
    if swing_highs:
        for sh in swing_highs:
            sh_idx = sh["index"]
            sh_level = sh["price"]
            for s in range(max(start_search, sh_idx + 1), t):
                # Candle pierced above swing high
                if highs[s] > sh_level and closes[s] <= sh_level + 1.0: # Sweep with rejection
                    bearish_sweep = {
                        "sweep_idx": s,
                        "level": sh_level,
                        "sweep_extreme": float(np.max(highs[s : t + 1]))
                    }

    # 4. Check for Market Structure Shift (MSS) and Displacement FVG
    # Priority A: Bullish Setup
    if bullish_sweep is not None:
        sw_idx = bullish_sweep["sweep_idx"]
        # Find intermediate high between sweep and current bar
        if sw_idx < t:
            # Pivot high to break: max high between sweep bar and t-1
            # Or the closest confirmed swing high before/at the sweep
            prior_shs = [sh for sh in swing_highs if sh["index"] <= sw_idx]
            pivot_level = prior_shs[-1]["price"] if prior_shs else np.max(highs[max(0, sw_idx - 5) : sw_idx + 1])

            # Check if an MSS candle closed above pivot_level between sw_idx and t
            mss_idx = None
            for k in range(sw_idx + 1, t + 1):
                if closes[k] > pivot_level:
                    mss_idx = k
                    break

            if mss_idx is not None:
                # Look for an unmitigated Bullish FVG created during this displacement leg
                # An FVG at bar k requires: lows[k] > highs[k-2]
                for k in range(sw_idx + 2, t + 1):
                    c1_hi = highs[k - 2]
                    c3_lo = lows[k]
                    gap = c3_lo - c1_hi
                    if gap >= min_gap:
                        # Check displacement candle body ratio
                        disp_body = abs(closes[k - 1] - opens[k - 1])
                        disp_range = highs[k - 1] - lows[k - 1]
                        body_ratio = disp_body / disp_range if disp_range > 0 else 0

                        if body_ratio >= config.DISPLACEMENT_BODY_RATIO:
                            fvg_top = round(float(c3_lo), 2)
                            fvg_bot = round(float(c1_hi), 2)
                            ce = round((fvg_top + fvg_bot) / 2.0, 2)

                            # Verify FVG has NOT been mitigated past CE prior to or at bar t
                            is_mitigated = False
                            if k < t:
                                if np.min(lows[k + 1 : t + 1]) <= ce:
                                    is_mitigated = True

                            if not is_mitigated:
                                entry = ce
                                sl = round(bullish_sweep["sweep_extreme"] - buffer, 2)
                                risk_dist = round(abs(entry - sl), 2)
                                if risk_dist >= min_stop:
                                    tp = round(entry + rr_ratio * risk_dist, 2)
                                    return {
                                        "bar_idx": t,
                                        "time": df.index[t],
                                        "direction": "BUY",
                                        "entry": entry,
                                        "sl": sl,
                                        "tp": tp,
                                        "risk_dist": risk_dist,
                                        "killzone": kz_name,
                                        "sweep_level": bullish_sweep["level"],
                                        "sweep_extreme": bullish_sweep["sweep_extreme"],
                                        "fvg_top": fvg_top,
                                        "fvg_bot": fvg_bot,
                                        "ce": ce
                                    }

    # Priority B: Bearish Setup
    if bearish_sweep is not None:
        sw_idx = bearish_sweep["sweep_idx"]
        if sw_idx < t:
            prior_sls = [sl for sl in swing_lows if sl["index"] <= sw_idx]
            pivot_level = prior_sls[-1]["price"] if prior_sls else np.min(lows[max(0, sw_idx - 5) : sw_idx + 1])

            # Check MSS candle close below pivot_level
            mss_idx = None
            for k in range(sw_idx + 1, t + 1):
                if closes[k] < pivot_level:
                    mss_idx = k
                    break

            if mss_idx is not None:
                # Look for unmitigated Bearish FVG created during displacement leg
                # highs[k] < lows[k-2]
                for k in range(sw_idx + 2, t + 1):
                    c1_lo = lows[k - 2]
                    c3_hi = highs[k]
                    gap = c1_lo - c3_hi
                    if gap >= min_gap:
                        disp_body = abs(closes[k - 1] - opens[k - 1])
                        disp_range = highs[k - 1] - lows[k - 1]
                        body_ratio = disp_body / disp_range if disp_range > 0 else 0

                        if body_ratio >= config.DISPLACEMENT_BODY_RATIO:
                            fvg_top = round(float(c1_lo), 2)
                            fvg_bot = round(float(c3_hi), 2)
                            ce = round((fvg_top + fvg_bot) / 2.0, 2)

                            # Verify unmitigated
                            is_mitigated = False
                            if k < t:
                                if np.max(highs[k + 1 : t + 1]) >= ce:
                                    is_mitigated = True

                            if not is_mitigated:
                                entry = ce
                                sl = round(bearish_sweep["sweep_extreme"] + buffer, 2)
                                risk_dist = round(abs(sl - entry), 2)
                                if risk_dist >= min_stop:
                                    tp = round(entry - rr_ratio * risk_dist, 2)
                                    return {
                                        "bar_idx": t,
                                        "time": df.index[t],
                                        "direction": "SELL",
                                        "entry": entry,
                                        "sl": sl,
                                        "tp": tp,
                                        "risk_dist": risk_dist,
                                        "killzone": kz_name,
                                        "sweep_level": bearish_sweep["level"],
                                        "sweep_extreme": bearish_sweep["sweep_extreme"],
                                        "fvg_top": fvg_top,
                                        "fvg_bot": fvg_bot,
                                        "ce": ce
                                    }

    return None


def generate_all_signals(df):
    """
    Computes signals across the entire DataFrame in a strictly causal manner.
    Returns a dictionary mapping bar index to signal dict (or empty if no signal).
    """
    signals = {}
    n = len(df)
    for t in range(n):
        sig = evaluate_causal_signal_at_bar(df, t)
        if sig is not None:
            signals[t] = sig
    return signals
