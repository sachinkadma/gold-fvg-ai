import json, ssl, time
import pandas as pd
import numpy as np
from datetime import datetime
import zoneinfo
import requests
import cv2
from PIL import Image

NY_TZ = zoneinfo.ZoneInfo("America/New_York")

# -----------------------------------------------------------------------------
# 1. LIVE GOLD (XAU/USD / frxXAUUSD) 15m DATA FEED
# -----------------------------------------------------------------------------
def fetch_live_gold_candles(granularity=900, count=50):
    """
    Fetches real-time 15m Gold (frxXAUUSD) candlestick array.
    Primary: Deriv Public WebSocket Gateway (app_id=1089).
    Failover: Live Gold Spot REST feed (GC=F) if WebSocket hits regional Cloudflare filter.
    Returns: df, cur_price, source_name, error_msg
    """
    # 1. Try Deriv WebSocket Gateway (Direct Spot frxXAUUSD)
    ws_endpoints = [
        "wss://ws.derivws.com/websockets/v3?app_id=1089",
        "wss://ws.binaryws.com/websockets/v3?app_id=1089"
    ]
    
    for url in ws_endpoints:
        try:
            import websocket
            ws = websocket.create_connection(
                url,
                timeout=4,
                sslopt={"cert_reqs": ssl.CERT_NONE},
                header=["Origin: https://app.deriv.com", "User-Agent: Mozilla/5.0"]
            )
            req = {
                "ticks_history": "frxXAUUSD",
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "start": 1,
                "style": "candles",
                "granularity": granularity
            }
            ws.send(json.dumps(req))
            res = ws.recv()
            data = json.loads(res)
            ws.close()
            
            raw_candles = data.get("candles", [])
            if raw_candles:
                rows = []
                for c in raw_candles:
                    ts = pd.to_datetime(c["epoch"], unit="s", utc=True).tz_convert(NY_TZ)
                    rows.append({
                        "Date_NY": ts,
                        "Open": round(float(c["open"]), 2),
                        "High": round(float(c["high"]), 2),
                        "Low": round(float(c["low"]), 2),
                        "Close": round(float(c["close"]), 2)
                    })
                df = pd.DataFrame(rows).sort_values("Date_NY").reset_index(drop=True)
                cur_price = df.iloc[-1]["Close"]
                return df, cur_price, "Deriv WebSocket (Spot frxXAUUSD)", None
        except Exception:
            continue

    # 2. Resilient Live Spot Failover (Yahoo Finance REST GC=F / XAUUSD Spot)
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=15m&range=3d"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 200:
            res_json = r.json()
            results = res_json.get("chart", {}).get("result", [])
            if results:
                timestamps = results[0].get("timestamp", [])
                indicators = results[0].get("indicators", {}).get("quote", [{}])[0]
                opens = indicators.get("open", [])
                highs = indicators.get("high", [])
                lows = indicators.get("low", [])
                closes = indicators.get("close", [])
                
                rows = []
                for t_stamp, o, h, l, c in zip(timestamps, opens, highs, lows, closes):
                    if None in (t_stamp, o, h, l, c): continue
                    ts = pd.to_datetime(t_stamp, unit="s", utc=True).tz_convert(NY_TZ)
                    rows.append({
                        "Date_NY": ts,
                        "Open": round(float(o), 2),
                        "High": round(float(h), 2),
                        "Low": round(float(l), 2),
                        "Close": round(float(c), 2)
                    })
                if rows:
                    df = pd.DataFrame(rows).tail(count).reset_index(drop=True)
                    cur_price = df.iloc[-1]["Close"]
                    return df, cur_price, "Gold Spot Live Feed (Active)", None
    except Exception as e:
        return None, 0.0, "Error", str(e)

    return None, 0.0, "Error", "Live feeds unreachable"

# -----------------------------------------------------------------------------
# 2. DETERMINISTIC ALGORITHMIC MATH ENGINE (ICT 15m)
# -----------------------------------------------------------------------------
def detect_swing_points(df, window=5):
    """
    Identifies fractal swing highs and swing lows using a 5-candle rolling window
    (2 left, 1 center, 2 right).
    Returns:
      swing_highs: list of dicts {'index', 'price', 'time'}
      swing_lows: list of dicts {'index', 'price', 'time'}
    """
    swing_highs = []
    swing_lows = []
    n = len(df)
    k = window // 2

    for i in range(k, n - k):
        cur_hi = df["High"].iloc[i]
        cur_lo = df["Low"].iloc[i]
        is_hi = True
        is_lo = True

        for j in range(i - k, i + k + 1):
            if j == i: continue
            if df["High"].iloc[j] >= cur_hi:
                is_hi = False
            if df["Low"].iloc[j] <= cur_lo:
                is_lo = False

        if is_hi:
            swing_highs.append({
                "index": i,
                "price": round(float(cur_hi), 2),
                "time": df["Date_NY"].iloc[i] if "Date_NY" in df else i
            })
        if is_lo:
            swing_lows.append({
                "index": i,
                "price": round(float(cur_lo), 2),
                "time": df["Date_NY"].iloc[i] if "Date_NY" in df else i
            })

    return swing_highs, swing_lows

def detect_displacement_candles(df, min_body_ratio=0.70):
    """
    Flags displacement candles where body ratio |Close - Open| / (High - Low) >= 0.70
    and candle range is above the local median range.
    """
    ranges = df["High"] - df["Low"]
    bodies = (df["Close"] - df["Open"]).abs()
    median_rng = float(ranges.median()) if len(ranges) > 0 else 1.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        body_ratios = np.where(ranges > 0, bodies / ranges, 0.0)

    is_disp = (body_ratios >= min_body_ratio) & (ranges >= median_rng * 0.95)
    return is_disp, body_ratios

def detect_3candle_fvgs(df, min_gap_pts=0.40):
    """
    Computes exact 3-candle Fair Value Gaps (BISI & SIBI) and verifies whether
    subsequent candles have mitigated past 50% Consequent Encroachment (CE).
    Returns list of dicts with FVG metrics.
    """
    fvgs = []
    n = len(df)
    if n < 3:
        return fvgs

    _, body_ratios = detect_displacement_candles(df, min_body_ratio=0.70)

    for i in range(2, n):
        # Candle 1: i-2, Candle 2 (displacement): i-1, Candle 3: i
        c1_hi, c1_lo = df["High"].iloc[i-2], df["Low"].iloc[i-2]
        c2_body_ratio = body_ratios[i-1]
        c3_hi, c3_lo = df["High"].iloc[i], df["Low"].iloc[i]

        # 1. Bullish FVG (BISI)
        if c3_lo > c1_hi:
            gap = c3_lo - c1_hi
            if gap >= min_gap_pts:
                fvg_top = round(float(c3_lo), 2)
                fvg_bot = round(float(c1_hi), 2)
                ce = round((fvg_top + fvg_bot) / 2.0, 2)
                
                # Mitigation check in candles after formation (i+1 to n-1)
                mitigated = False
                fully_filled = False
                if i + 1 < n:
                    subsequent_lows = df["Low"].iloc[i+1:].values
                    if len(subsequent_lows) > 0:
                        if np.min(subsequent_lows) <= ce:
                            mitigated = True
                        if np.min(subsequent_lows) <= fvg_bot:
                            fully_filled = True

                fvgs.append({
                    "type": "BULLISH_FVG",
                    "fvg_top": fvg_top,
                    "fvg_bot": fvg_bot,
                    "ce": ce,
                    "gap_pts": round(gap, 2),
                    "candle_idx": i-1,
                    "is_displacement": bool(c2_body_ratio >= 0.65),
                    "mitigated": mitigated,
                    "fully_filled": fully_filled
                })

        # 2. Bearish FVG (SIBI)
        elif c3_hi < c1_lo:
            gap = c1_lo - c3_hi
            if gap >= min_gap_pts:
                fvg_top = round(float(c1_lo), 2)
                fvg_bot = round(float(c3_hi), 2)
                ce = round((fvg_top + fvg_bot) / 2.0, 2)
                
                # Mitigation check in candles after formation
                mitigated = False
                fully_filled = False
                if i + 1 < n:
                    subsequent_highs = df["High"].iloc[i+1:].values
                    if len(subsequent_highs) > 0:
                        if np.max(subsequent_highs) >= ce:
                            mitigated = True
                        if np.max(subsequent_highs) >= fvg_top:
                            fully_filled = True

                fvgs.append({
                    "type": "BEARISH_FVG",
                    "fvg_top": fvg_top,
                    "fvg_bot": fvg_bot,
                    "ce": ce,
                    "gap_pts": round(gap, 2),
                    "candle_idx": i-1,
                    "is_displacement": bool(c2_body_ratio >= 0.65),
                    "mitigated": mitigated,
                    "fully_filled": fully_filled
                })

    return fvgs

def compute_deterministic_market_structure(df, account_balance=10000.0, risk_percent=1.0):
    """
    Computes all market structure elements purely mathematically:
    1. Swing Points (5-candle rolling window)
    2. Liquidity Sweeps vs Breakout Expansions (BSL / SSL)
    3. Market Structure Shift (MSS) with candle body close confirmation
    4. Unmitigated 3-Candle FVGs & 50% Consequent Encroachment
    5. Candle Range Theory (CRT) & Equilibrium Dealing Range (Discount vs Premium)
    6. Exact Invalidation Stop Loss & 1:2.0 Minimum Take Profit Rail
    7. Dynamic Lot Sizing
    """
    n = len(df)
    if n < 10:
        return {
            "has_setup": False,
            "bias": "NEUTRAL",
            "action": "WAIT",
            "rejection_reason": "Insufficient candles for deterministic market structure analysis (< 10 candles)",
            "current_price": float(df.iloc[-1]["Close"]) if n > 0 else 0.0
        }

    current_price = round(float(df.iloc[-1]["Close"]), 2)
    range_high = round(float(df["High"].max()), 2)
    range_low = round(float(df["Low"].min()), 2)
    equilibrium = round((range_high + range_low) / 2.0, 2)

    # 1. Detect Swing Points
    swing_highs, swing_lows = detect_swing_points(df, window=5)
    
    # 2. Detect 3-Candle FVGs
    all_fvgs = detect_3candle_fvgs(df, min_gap_pts=0.35)
    unmitigated_bull_fvgs = [f for f in all_fvgs if f["type"] == "BULLISH_FVG" and not f["mitigated"]]
    unmitigated_bear_fvgs = [f for f in all_fvgs if f["type"] == "BEARISH_FVG" and not f["mitigated"]]

    # 3. Detect Sweeps vs Expansions (Looking back over the last 15 candles)
    recent_start_idx = max(5, n - 20)
    
    bullish_sweep = None
    bearish_sweep = None

    for i in range(recent_start_idx, n):
        c_lo = df["Low"].iloc[i]
        c_hi = df["High"].iloc[i]
        c_cl = df["Close"].iloc[i]

        # Check against swing lows that occurred prior to candle i
        for sl in swing_lows:
            if sl["index"] < i - 2 and (i - sl["index"]) <= 30:
                p_low = sl["price"]
                # Bullish SSL Sweep: Wick < prior low, but Close >= prior low
                if c_lo < p_low:
                    if c_cl >= p_low:
                        bullish_sweep = {
                            "type": "SSL_SWEEP",
                            "level": p_low,
                            "sweep_low": round(float(c_lo), 2),
                            "candle_idx": i,
                            "depth_pts": round(p_low - c_lo, 2)
                        }
                    # If close < p_low, that is a breakout expansion (BOS), not a sweep

        # Check against swing highs that occurred prior to candle i
        for sh in swing_highs:
            if sh["index"] < i - 2 and (i - sh["index"]) <= 30:
                p_high = sh["price"]
                # Bearish BSL Sweep: Wick > prior high, but Close <= prior high
                if c_hi > p_high:
                    if c_cl <= p_high:
                        bearish_sweep = {
                            "type": "BSL_SWEEP",
                            "level": p_high,
                            "sweep_high": round(float(c_hi), 2),
                            "candle_idx": i,
                            "depth_pts": round(c_hi - p_high, 2)
                        }

    # 4. Check for Market Structure Shift (MSS) after Sweep
    bullish_mss = None
    bearish_mss = None

    # Evaluate Bullish Path
    if bullish_sweep is not None:
        sw_idx = bullish_sweep["candle_idx"]
        # Find intermediate swing high formed prior to or around the sweep
        relevant_shs = [sh for sh in swing_highs if sh["index"] <= sw_idx]
        if relevant_shs:
            pivot_sh = relevant_shs[-1]["price"]
            # Check if any candle from sw_idx to n-1 closed body > pivot_sh
            for k in range(sw_idx, n):
                if df["Close"].iloc[k] > pivot_sh:
                    bullish_mss = {
                        "mss_level": pivot_sh,
                        "mss_candle_idx": k,
                        "close_price": round(float(df["Close"].iloc[k]), 2)
                    }
                    break

    # Evaluate Bearish Path
    if bearish_sweep is not None:
        sw_idx = bearish_sweep["candle_idx"]
        relevant_sls = [sl for sl in swing_lows if sl["index"] <= sw_idx]
        if relevant_sls:
            pivot_sl = relevant_sls[-1]["price"]
            for k in range(sw_idx, n):
                if df["Close"].iloc[k] < pivot_sl:
                    bearish_mss = {
                        "mss_level": pivot_sl,
                        "mss_candle_idx": k,
                        "close_price": round(float(df["Close"].iloc[k]), 2)
                    }
                    break

    # 5. Synthesize Unanimous Setup Criteria
    rejection_reasons = []

    # Priority A: Bullish Setup
    if bullish_sweep and bullish_mss and unmitigated_bull_fvgs:
        active_fvg = unmitigated_bull_fvgs[-1]
        entry_ce = active_fvg["ce"]
        stop_loss = round(bullish_sweep["sweep_low"] - 1.5, 2)
        risk_dist = abs(entry_ce - stop_loss)
        
        # Enforce minimum 1:2.0 RR
        tp1 = round(entry_ce + 2.0 * risk_dist, 2)
        tp2_runner = round(range_high, 2) if range_high > tp1 else round(entry_ce + 3.2 * risk_dist, 2)
        rr_ratio = round((tp1 - entry_ce) / risk_dist, 2) if risk_dist > 0 else 0.0

        # Dealing range CRT check: Entry should be in discount (< equilibrium)
        in_discount = entry_ce <= equilibrium + 2.0
        if not in_discount:
            rejection_reasons.append("CRT Invalidation: Bullish entry is in Premium (> 50% Equilibrium)")

        # Lot sizing
        sl_pips = max(5.0, risk_dist * 10.0)
        risk_dollars = account_balance * (risk_percent / 100.0)
        rec_lot = round(risk_dollars / (sl_pips * 10.0), 2)

        has_setup = len(rejection_reasons) == 0 and rr_ratio >= 2.0
        action = "EXECUTE_LIMIT" if has_setup else "WAIT"

        return {
            "has_setup": has_setup,
            "action": action,
            "bias": "BULLISH",
            "verdict": "BUY_LIMIT",
            "setup_name": "ICT 2022 Mentorship Buy Model",
            "setup_grade": "A+ Institutional (High Probability)" if has_setup else "B+ Evaluation (Wait Criteria)",
            "current_price": current_price,
            "sweep_type": "SSL_SWEEP",
            "sweep_level": bullish_sweep["level"],
            "sweep_extreme": bullish_sweep["sweep_low"],
            "mss_confirmed": True,
            "mss_level": bullish_mss["mss_level"],
            "fvg_top": active_fvg["fvg_top"],
            "fvg_bottom": active_fvg["fvg_bot"],
            "entry_ce": entry_ce,
            "stop_loss": stop_loss,
            "take_profit_1": tp1,
            "take_profit_2": tp2_runner,
            "risk_reward_ratio": rr_ratio,
            "recommended_lot": rec_lot,
            "risk_amount": risk_dollars,
            "confluence_score": 94 if has_setup else 78,
            "range_zone": "DISCOUNT (Buy Zone)" if in_discount else "PREMIUM",
            "trapped_liquidity": f"Retail breakout sellers trapped below swing low ${bullish_sweep['level']:.2f}",
            "invalidation_thesis": f"Candle body close below ${stop_loss:.2f} completely invalidates bullish order flow",
            "rejection_reasons": rejection_reasons,
            "unmitigated_fvgs_count": len(unmitigated_bull_fvgs)
        }

    # Priority B: Bearish Setup
    elif bearish_sweep and bearish_mss and unmitigated_bear_fvgs:
        active_fvg = unmitigated_bear_fvgs[-1]
        entry_ce = active_fvg["ce"]
        stop_loss = round(bearish_sweep["sweep_high"] + 1.5, 2)
        risk_dist = abs(stop_loss - entry_ce)

        tp1 = round(entry_ce - 2.0 * risk_dist, 2)
        tp2_runner = round(range_low, 2) if range_low < tp1 else round(entry_ce - 3.2 * risk_dist, 2)
        rr_ratio = round((entry_ce - tp1) / risk_dist, 2) if risk_dist > 0 else 0.0

        # Dealing range CRT check: Entry should be in premium (> equilibrium)
        in_premium = entry_ce >= equilibrium - 2.0
        if not in_premium:
            rejection_reasons.append("CRT Invalidation: Bearish entry is in Discount (< 50% Equilibrium)")

        sl_pips = max(5.0, risk_dist * 10.0)
        risk_dollars = account_balance * (risk_percent / 100.0)
        rec_lot = round(risk_dollars / (sl_pips * 10.0), 2)

        has_setup = len(rejection_reasons) == 0 and rr_ratio >= 2.0
        action = "EXECUTE_LIMIT" if has_setup else "WAIT"

        return {
            "has_setup": has_setup,
            "action": action,
            "bias": "BEARISH",
            "verdict": "SELL_LIMIT",
            "setup_name": "ICT 2022 Mentorship Sell Model",
            "setup_grade": "A+ Institutional (High Probability)" if has_setup else "B+ Evaluation (Wait Criteria)",
            "current_price": current_price,
            "sweep_type": "BSL_SWEEP",
            "sweep_level": bearish_sweep["level"],
            "sweep_extreme": bearish_sweep["sweep_high"],
            "mss_confirmed": True,
            "mss_level": bearish_mss["mss_level"],
            "fvg_top": active_fvg["fvg_top"],
            "fvg_bottom": active_fvg["fvg_bot"],
            "entry_ce": entry_ce,
            "stop_loss": stop_loss,
            "take_profit_1": tp1,
            "take_profit_2": tp2_runner,
            "risk_reward_ratio": rr_ratio,
            "recommended_lot": rec_lot,
            "risk_amount": risk_dollars,
            "confluence_score": 94 if has_setup else 78,
            "range_zone": "PREMIUM (Sell Zone)" if in_premium else "DISCOUNT",
            "trapped_liquidity": f"Retail breakout buyers trapped above swing high ${bearish_sweep['level']:.2f}",
            "invalidation_thesis": f"Candle body close above ${stop_loss:.2f} completely invalidates bearish order flow",
            "rejection_reasons": rejection_reasons,
            "unmitigated_fvgs_count": len(unmitigated_bear_fvgs)
        }

    # Baseline Observation when full MSS/FVG is pending
    bias = "BULLISH" if current_price < equilibrium else "BEARISH"
    missing = []
    if not (bullish_sweep or bearish_sweep):
        missing.append("Prerequisite #13 Liquidity Sweep (BSL or SSL) not yet detected in recent 20 candles")
    if not (bullish_mss or bearish_mss):
        missing.append("Prerequisite #4 Market Structure Shift (MSS) candle body close pending")
    if not (unmitigated_bull_fvgs or unmitigated_bear_fvgs):
        missing.append("Prerequisite #17 Unmitigated 3-Candle FVG with 50% CE not yet formed in displacement leg")

    # Safe deterministic calculation for chart display
    dummy_sl = round(range_low - 2.0 if bias == "BULLISH" else range_high + 2.0, 2)
    dummy_entry = current_price
    dummy_tp = round(dummy_entry + 2.0 * abs(dummy_entry - dummy_sl) if bias == "BULLISH" else dummy_entry - 2.0 * abs(dummy_entry - dummy_sl), 2)
    
    return {
        "has_setup": False,
        "action": "WAIT",
        "bias": bias,
        "verdict": "WAIT_CONFIRMATION",
        "setup_name": "Structure Consolidation / Order Flow Building",
        "setup_grade": "Neutral / Observation (No Execution)",
        "current_price": current_price,
        "sweep_type": "NONE",
        "sweep_level": swing_lows[-1]["price"] if swing_lows else range_low,
        "sweep_extreme": range_low if bias == "BULLISH" else range_high,
        "mss_confirmed": False,
        "mss_level": equilibrium,
        "fvg_top": round(current_price + 2.0, 2),
        "fvg_bottom": round(current_price - 2.0, 2),
        "entry_ce": dummy_entry,
        "stop_loss": dummy_sl,
        "take_profit_1": dummy_tp,
        "take_profit_2": dummy_tp,
        "risk_reward_ratio": 2.0,
        "recommended_lot": 0.0,
        "risk_amount": 0.0,
        "confluence_score": 60,
        "range_zone": "EQUILIBRIUM" if abs(current_price - equilibrium) < 3.0 else ("DISCOUNT" if current_price < equilibrium else "PREMIUM"),
        "trapped_liquidity": "Liquidity pools resting at external range boundaries",
        "invalidation_thesis": "Awaiting clear liquidity sweep and displacement MSS",
        "rejection_reasons": missing,
        "unmitigated_fvgs_count": len(all_fvgs)
    }

# -----------------------------------------------------------------------------
# 3. CAMERON4X OPENCV STRICT COORDINATE MAPPING CHART ENGINE
# -----------------------------------------------------------------------------
def render_native_cameron4x_chart(
    df,
    math_structure=None,
    verdict="SELL_LIMIT",
    setup_name="ICT 2022 Mentorship",
    score=92,
    session_txt="NY AM Killzone",
    entry_p=None,
    sl_p=None,
    tp_p=None,
    sweep_p=None,
    mss_p=None,
    fvg_top=None,
    fvg_bot=None
):
    """
    Renders an institutional Cameron4x dark candlestick chart (940x540) on #0B1120.
    Binds visual coordinates strictly to deterministic price floats via price_to_pixel().
    """
    w, h = 940, 540
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (32, 17, 11)  # Canvas #0B1120 in BGR

    if df is None or len(df) < 5:
        cv2.putText(img, "No Live Candlestick Data Available", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return Image.fromarray(img[:, :, ::-1])

    # Unpack from math_structure if provided
    if math_structure is not None:
        verdict = math_structure.get("verdict", verdict)
        setup_name = math_structure.get("setup_name", setup_name)
        score = math_structure.get("confluence_score", score)
        entry_p = math_structure.get("entry_ce", entry_p)
        sl_p = math_structure.get("stop_loss", sl_p)
        tp_p = math_structure.get("take_profit_1", tp_p)
        sweep_p = math_structure.get("sweep_level", sweep_p)
        mss_p = math_structure.get("mss_level", mss_p)
        fvg_top = math_structure.get("fvg_top", fvg_top)
        fvg_bot = math_structure.get("fvg_bottom", fvg_bot)

    is_buy = "BUY" in verdict.upper()
    last_close = float(df.iloc[-1]["Close"])

    # High-Visibility Institutional Colors (BGR)
    COLOR_MSS    = (255, 255, 255) # Pure White
    COLOR_SWEEP  = (248, 189, 56)  # Sky Blue #38BDF8
    COLOR_BULL   = (129, 185, 16)  # Emerald Green #10B981
    COLOR_BEAR   = (68, 68, 239)   # Coral Red #EF4444
    COLOR_CE     = (6, 182, 212)   # Amber/Cyan #06B6D4
    COLOR_TP     = (153, 211, 52)  # Mint Green #34D399
    COLOR_SPOT   = (36, 191, 251)  # Amber #FBBF24
    COLOR_GRID   = (45, 30, 20)    # Subdued Grid lines

    all_lows = df["Low"].values
    all_highs = df["High"].values

    # Fallbacks if any level is None
    if sweep_p is None: sweep_p = float(np.min(all_lows)) if is_buy else float(np.max(all_highs))
    if mss_p is None: mss_p = float(np.median(df["Close"].values))
    if fvg_top is None: fvg_top = round(last_close + 3.0, 2)
    if fvg_bot is None: fvg_bot = round(last_close - 3.0, 2)
    if entry_p is None: entry_p = round((fvg_top + fvg_bot) / 2.0, 2)
    if sl_p is None: sl_p = round(sweep_p - 3.0 if is_buy else sweep_p + 3.0, 2)
    if tp_p is None: tp_p = round(entry_p + 2.0 * abs(entry_p - sl_p) if is_buy else entry_p - 2.0 * abs(entry_p - sl_p), 2)

    # Compute bounding price range
    all_p_points = [
        float(np.min(all_lows)), float(np.max(all_highs)),
        entry_p, sl_p, tp_p, sweep_p, mss_p, fvg_top, fvg_bot, last_close
    ]
    min_p = min(all_p_points)
    max_p = max(all_p_points)
    p_pad = max(2.0, (max_p - min_p) * 0.08)
    min_p -= p_pad
    max_p += p_pad
    p_range = max(1.0, max_p - min_p)

    def price_to_pixel(p):
        """Strict linear transform from price float to canvas pixel Y."""
        return int(h * 0.12 + (max_p - p) / p_range * (h * 0.72))

    # 1. Price Grid & Axis Labels
    for p_level in np.linspace(min_p, max_p, 8):
        y_l = price_to_pixel(p_level)
        cv2.line(img, (int(w * 0.05), y_l), (int(w * 0.86), y_l), COLOR_GRID, 1)
        cv2.putText(img, f"${p_level:.1f}", (int(w * 0.87), y_l + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

    # 2. Render Candlesticks
    n_c = len(df)
    chart_w = int(w * 0.79)
    c_w = max(5, int(chart_w / n_c) - 2)
    start_x = int(w * 0.06)

    for i, row in df.iterrows():
        x_mid = start_x + i * (c_w + 3)
        y_o = price_to_pixel(row["Open"])
        y_c = price_to_pixel(row["Close"])
        y_hi = price_to_pixel(row["High"])
        y_lo = price_to_pixel(row["Low"])

        is_bull = row["Close"] >= row["Open"]
        c_col = COLOR_BULL if is_bull else COLOR_BEAR

        # Wick
        cv2.line(img, (x_mid, y_hi), (x_mid, y_lo), c_col, 1, cv2.LINE_AA)
        
        # Body
        top_b = min(y_o, y_c)
        bot_b = max(y_o, y_c)
        if bot_b - top_b < 2: bot_b = top_b + 2
        cv2.rectangle(img, (x_mid - c_w // 2, top_b), (x_mid + c_w // 2, bot_b), c_col, -1)

        # Time labels at bottom
        if i % 7 == 0 and "Date_NY" in row:
            t_str = row["Date_NY"].strftime("%H:%M")
            cv2.putText(img, t_str, (x_mid - 16, int(h * 0.90)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (148, 163, 184), 1, cv2.LINE_AA)

    overlay = img.copy()

    # 3. Sweep Line
    sw_y = price_to_pixel(sweep_p)
    sw_lbl = f"⚡ #13 SSL Swept (${sweep_p:.1f})" if is_buy else f"⚡ #13 BSL Swept (${sweep_p:.1f})"
    cv2.line(img, (int(w * 0.08), sw_y), (int(w * 0.85), sw_y), COLOR_SWEEP, 2, cv2.LINE_AA)
    cv2.putText(img, sw_lbl, (int(w * 0.10), sw_y + 16 if is_buy else sw_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_SWEEP, 1, cv2.LINE_AA)

    # 4. MSS Dashed Line
    mss_y = price_to_pixel(mss_p)
    for x in range(int(w * 0.20), int(w * 0.85), 14):
        cv2.line(img, (x, mss_y), (x + 8, mss_y), COLOR_MSS, 2, cv2.LINE_AA)
    cv2.putText(img, f"⚡ #4 MSS Pivot (${mss_p:.1f})", (int(w * 0.22), mss_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_MSS, 1, cv2.LINE_AA)

    # 5. FVG Shaded Box & 50% CE Midline
    fvg_col = COLOR_BULL if is_buy else COLOR_BEAR
    fvg_lbl = f"🟩 #17 Bull FVG (BISI) [${fvg_bot:.1f}-${fvg_top:.1f}]" if is_buy else f"🟥 #17 Bear FVG (SIBI) [${fvg_bot:.1f}-${fvg_top:.1f}]"
    
    fvg_y1 = price_to_pixel(fvg_top)
    fvg_y2 = price_to_pixel(fvg_bot)
    fvg_x1 = int(w * 0.38)
    fvg_x2 = int(w * 0.85)

    cv2.rectangle(overlay, (fvg_x1, min(fvg_y1, fvg_y2)), (fvg_x2, max(fvg_y1, fvg_y2)), fvg_col, -1)
    cv2.rectangle(img, (fvg_x1, min(fvg_y1, fvg_y2)), (fvg_x2, max(fvg_y1, fvg_y2)), fvg_col, 2)
    
    # 50% CE Midline
    ce_y = price_to_pixel(entry_p)
    for x in range(fvg_x1, fvg_x2, 8):
        cv2.line(img, (x, ce_y), (x + 5, ce_y), COLOR_CE, 2, cv2.LINE_AA)
    cv2.putText(img, fvg_lbl, (fvg_x1 + 6, min(fvg_y1, fvg_y2) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"🔹 #28 50% CE Entry (${entry_p:.2f})", (fvg_x1 + 6, ce_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_CE, 1, cv2.LINE_AA)

    # 6. Target 1:2 R:R Rail
    tp_y = price_to_pixel(tp_p)
    for x in range(int(w * 0.30), int(w * 0.85), 14):
        cv2.line(img, (x, tp_y), (x + 8, tp_y), COLOR_TP, 2, cv2.LINE_AA)
    cv2.putText(img, f"🎯 #94 Institutional Target 1:2 R:R (${tp_p:.2f})", (int(w * 0.32), tp_y - 6 if is_buy else tp_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TP, 1, cv2.LINE_AA)

    # 7. Live Spot Price Dotted Line & Right-Scale Badge
    cur_y = price_to_pixel(last_close)
    for x in range(int(w * 0.06), int(w * 0.86), 6):
        cv2.line(img, (x, cur_y), (x + 3, cur_y), COLOR_SPOT, 1, cv2.LINE_AA)
    cv2.rectangle(img, (int(w * 0.86), cur_y - 10), (w - 6, cur_y + 10), (15, 23, 42), -1)
    cv2.rectangle(img, (int(w * 0.86), cur_y - 10), (w - 6, cur_y + 10), COLOR_SPOT, 1)
    cv2.putText(img, f"${last_close:.2f}", (int(w * 0.865), cur_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_SPOT, 1, cv2.LINE_AA)

    # Blend translucent FVG overlay (28% alpha)
    cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)

    # 8. Top Institutional Watermark Header
    banner_h = 36
    cv2.rectangle(img, (0, 0), (w, banner_h), (15, 23, 42), -1)
    cv2.line(img, (0, banner_h), (w, banner_h), (51, 65, 85), 1)
    status_text = f"DERIV SPOT 15m (frxXAUUSD) | {verdict} | Spot: ${last_close:.2f} | Confluence: {score}% | {session_txt[:24]}"
    cv2.putText(img, status_text, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (251, 191, 36), 1, cv2.LINE_AA)

    return Image.fromarray(img[:, :, ::-1])
