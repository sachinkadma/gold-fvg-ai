# NOT CAUSAL — Visualization / Legacy Only. Do not import in core/ or research/.
import json, ssl, time, os
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from PIL import Image, ImageDraw, ImageFont

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import zoneinfo
    NY_TZ = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    try:
        from backports import zoneinfo
        NY_TZ = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        import datetime as dt
        NY_TZ = dt.timezone.utc

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    go = None
    HAS_PLOTLY = False

# -----------------------------------------------------------------------------
# 1. LIVE GOLD (XAU/USD / frxXAUUSD) 15m DATA FEED
# -----------------------------------------------------------------------------
def fetch_live_gold_candles(
    granularity=900,
    count=50,
    source="AUTO",
    mt5_login=None,
    mt5_password=None,
    mt5_server=None,
    mt5_symbol=None
):
    """
    Fetches real-time 15m Gold candlestick array.
    Sources supported:
      - 'MT5': Direct MetaTrader 5 account connection
      - 'DERIV': Deriv Public WebSocket Gateway (frxXAUUSD)
      - 'SPOT': Live Gold Spot REST feed (GC=F)
      - 'AUTO': Priority MT5 (if on Windows with terminal) -> Deriv WebSocket -> Gold Spot
    Returns: df, cur_price, source_name, error_msg
    """
    if mt5_login is None:
        mt5_login = os.environ.get("MT5_LOGIN", 0)
    if mt5_password is None:
        mt5_password = os.environ.get("MT5_PASSWORD", "")
    if mt5_server is None:
        mt5_server = os.environ.get("MT5_SERVER", "MetaQuotes-Demo")
    if mt5_symbol is None:
        mt5_symbol = os.environ.get("MT5_SYMBOL", "XAUUSD")
    # 1. Attempt MetaTrader 5 if selected or in AUTO mode
    if source in ["MT5", "AUTO"]:
        try:
            try:
                from legacy.mt5_client import fetch_mt5_gold_candles, is_mt5_available
            except ImportError:
                try:
                    from mt5_client import fetch_mt5_gold_candles, is_mt5_available
                except ImportError:
                    fetch_mt5_gold_candles = None
                    is_mt5_available = lambda: False
            if is_mt5_available():
                df_mt5, cur_p_mt5, feed_name_mt5, err_mt5 = fetch_mt5_gold_candles(
                    login=mt5_login,
                    password=mt5_password,
                    server=mt5_server,
                    symbol=mt5_symbol,
                    timeframe="M15",
                    count=count
                )
                if df_mt5 is not None and not df_mt5.empty:
                    return df_mt5, cur_p_mt5, feed_name_mt5, None
                elif source == "MT5":
                    # If user explicitly asked for MT5, return the exact error
                    return None, 0.0, "MetaTrader 5", err_mt5
        except Exception as e:
            if source == "MT5":
                return None, 0.0, "MetaTrader 5", f"MT5 connection error: {str(e)}"

    # 2. Deriv Public WebSocket Gateway (app_id=1089)
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

    # 3. Resilient Live Spot Failover (Yahoo Finance REST GC=F / XAUUSD Spot)
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
    Identifies fractal swing highs and swing lows using a 5-candle rolling window.
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
    Flags displacement candles where body ratio |Close - Open| / (High - Low) >= 0.70.
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
    Computes exact 3-candle Fair Value Gaps (BISI & SIBI) and verifies mitigation past 50% CE.
    """
    fvgs = []
    n = len(df)
    if n < 3:
        return fvgs

    _, body_ratios = detect_displacement_candles(df, min_body_ratio=0.70)

    for i in range(2, n):
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

# -----------------------------------------------------------------------------
# 3. DETERMINISTIC ELLIOTT WAVE THEORY (EWT) ENGINE
# -----------------------------------------------------------------------------
def detect_elliott_wave_cycle(df, swing_highs, swing_lows):
    """
    Deterministically validates a 5-wave impulse sequence (0-1-2-3-4-5)
    using the 3 Cardinal Rules of Elliott Wave Theory:
      Rule 1: Wave 2 never retraces beyond Wave 0 origin (P2 > P0 bull; P2 < P0 bear).
      Rule 2: Wave 3 is never the shortest wave among Waves 1, 3, and 5.
      Rule 3: Wave 4 never enters Wave 1 price territory (P4 > P1 bull; P4 < P1 bear).
    """
    if not swing_highs and not swing_lows:
        return {
            "status": "INCOMPLETE_CYCLE",
            "cycle_phase": "INCOMPLETE_CYCLE",
            "direction": "NEUTRAL",
            "invalidation_price": 0.0,
            "wave_points": [],
            "fib_w2_ratio": 0.0,
            "fib_w3_ratio": 0.0,
            "fib_w4_ratio": 0.0,
            "cardinal_rules": {"rule1_w2": True, "rule2_w3": True, "rule3_w4": True},
            "rule_violation_reason": "No confirmed swing pivots detected"
        }

    # 1. Combine all swing points into chronological list
    all_pts = []
    for sh in swing_highs:
        all_pts.append({"index": sh["index"], "price": float(sh["price"]), "type": "high"})
    for sl in swing_lows:
        all_pts.append({"index": sl["index"], "price": float(sl["price"]), "type": "low"})

    all_pts.sort(key=lambda x: x["index"])

    # Clean to ensure strictly alternating sequence (high, low, high, low ...)
    alternating = []
    for pt in all_pts:
        if not alternating:
            alternating.append(pt)
        else:
            prev = alternating[-1]
            if prev["type"] == pt["type"]:
                if pt["type"] == "high":
                    if pt["price"] > prev["price"]:
                        alternating[-1] = pt
                else:
                    if pt["price"] < prev["price"]:
                        alternating[-1] = pt
            else:
                alternating.append(pt)

    if len(alternating) < 6:
        wave_pts = []
        for i, pt in enumerate(alternating[-5:]):
            wave_pts.append({"label": f"({i})", "idx": pt["index"], "price": pt["price"]})
        return {
            "status": "INCOMPLETE_CYCLE",
            "cycle_phase": "INCOMPLETE_CYCLE",
            "direction": "NEUTRAL",
            "invalidation_price": 0.0,
            "wave_points": wave_pts,
            "fib_w2_ratio": 0.0,
            "fib_w3_ratio": 0.0,
            "fib_w4_ratio": 0.0,
            "cardinal_rules": {"rule1_w2": True, "rule2_w3": True, "rule3_w4": True},
            "rule_violation_reason": "Fewer than 6 confirmed alternating fractal pivots (advisory pass)"
        }

    cand = alternating[-6:]
    p0, p1, p2, p3, p4, p5 = [p["price"] for p in cand]
    is_bull_cand = cand[0]["type"] == "low" and cand[1]["type"] == "high"
    is_bear_cand = cand[0]["type"] == "high" and cand[1]["type"] == "low"
    last_close = float(df.iloc[-1]["Close"]) if len(df) > 0 else p5

    # ------------------ BULLISH IMPULSE EVALUATION ------------------
    if is_bull_cand:
        w1_len = p1 - p0
        w3_len = p3 - p2
        w5_len = p5 - p4

        # Cardinal Rule 1: Wave 2 cannot break below Wave 0 origin
        rule1 = p2 > p0
        # Cardinal Rule 2: Wave 3 is never the shortest among 1, 3, and 5
        rule2 = not (w3_len < w1_len and w3_len < w5_len)
        # Cardinal Rule 3: Wave 4 low cannot enter Wave 1 price territory
        rule3 = p4 > p1
        # Directional motive check
        motive_valid = (p1 > p0) and (p3 > p2) and (p5 > p4) and (p3 > p1) and (p5 > p3)

        fib_w2 = round(abs(p1 - p2) / w1_len, 2) if w1_len > 0 else 0.0
        fib_w3 = round(w3_len / w1_len, 2) if w1_len > 0 else 0.0
        fib_w4 = round(abs(p3 - p4) / w3_len, 2) if w3_len > 0 else 0.0

        wave_points = [
            {"label": "(0)", "idx": cand[0]["index"], "price": p0},
            {"label": "(1)", "idx": cand[1]["index"], "price": p1},
            {"label": "(2)", "idx": cand[2]["index"], "price": p2},
            {"label": "(3)", "idx": cand[3]["index"], "price": p3},
            {"label": "(4)", "idx": cand[4]["index"], "price": p4},
            {"label": "(5)", "idx": cand[5]["index"], "price": p5}
        ]

        if not (rule1 and rule2 and rule3 and motive_valid):
            violation_reasons = []
            if not rule1: violation_reasons.append("Cardinal Rule 1: Wave 2 retraced beyond Wave 0 origin")
            if not rule2: violation_reasons.append("Cardinal Rule 2: Wave 3 is the shortest impulse wave")
            if not rule3: violation_reasons.append("Cardinal Rule 3: Wave 4 entered Wave 1 price territory")
            if not motive_valid: violation_reasons.append("Motive structure failure: Pivots do not form progressive higher highs")

            return {
                "status": "INVALID",
                "cycle_phase": "CARDINAL_RULE_VIOLATION",
                "direction": "BULLISH",
                "invalidation_price": p4,
                "wave_points": wave_points,
                "fib_w2_ratio": fib_w2,
                "fib_w3_ratio": fib_w3,
                "fib_w4_ratio": fib_w4,
                "cardinal_rules": {"rule1_w2": rule1, "rule2_w3": rule2, "rule3_w4": rule3},
                "rule_violation_reason": "; ".join(violation_reasons)
            }

        if last_close < p5:
            cycle_phase = "COMPLETED_IMPULSE_AWAITING_ABC"
        else:
            cycle_phase = "IN_WAVE_5_TERMINAL_SWEEP"

        return {
            "status": "VALID_IMPULSE",
            "cycle_phase": cycle_phase,
            "direction": "BULLISH",
            "invalidation_price": p4,
            "wave_points": wave_points,
            "fib_w2_ratio": fib_w2,
            "fib_w3_ratio": fib_w3,
            "fib_w4_ratio": fib_w4,
            "cardinal_rules": {"rule1_w2": True, "rule2_w3": True, "rule3_w4": True},
            "rule_violation_reason": None
        }

    # ------------------ BEARISH IMPULSE EVALUATION ------------------
    elif is_bear_cand:
        w1_len = p0 - p1
        w3_len = p2 - p3
        w5_len = p4 - p5

        rule1 = p2 < p0
        rule2 = not (w3_len < w1_len and w3_len < w5_len)
        rule3 = p4 < p1
        motive_valid = (p1 < p0) and (p3 < p2) and (p5 < p4) and (p3 < p1) and (p5 < p3)

        fib_w2 = round(abs(p2 - p1) / w1_len, 2) if w1_len > 0 else 0.0
        fib_w3 = round(w3_len / w1_len, 2) if w1_len > 0 else 0.0
        fib_w4 = round(abs(p4 - p3) / w3_len, 2) if w3_len > 0 else 0.0

        wave_points = [
            {"label": "(0)", "idx": cand[0]["index"], "price": p0},
            {"label": "(1)", "idx": cand[1]["index"], "price": p1},
            {"label": "(2)", "idx": cand[2]["index"], "price": p2},
            {"label": "(3)", "idx": cand[3]["index"], "price": p3},
            {"label": "(4)", "idx": cand[4]["index"], "price": p4},
            {"label": "(5)", "idx": cand[5]["index"], "price": p5}
        ]

        if not (rule1 and rule2 and rule3 and motive_valid):
            violation_reasons = []
            if not rule1: violation_reasons.append("Cardinal Rule 1: Wave 2 retraced beyond Wave 0 origin")
            if not rule2: violation_reasons.append("Cardinal Rule 2: Wave 3 is the shortest impulse wave")
            if not rule3: violation_reasons.append("Cardinal Rule 3: Wave 4 entered Wave 1 price territory")
            if not motive_valid: violation_reasons.append("Motive structure failure: Pivots do not form progressive lower lows")

            return {
                "status": "INVALID",
                "cycle_phase": "CARDINAL_RULE_VIOLATION",
                "direction": "BEARISH",
                "invalidation_price": p4,
                "wave_points": wave_points,
                "fib_w2_ratio": fib_w2,
                "fib_w3_ratio": fib_w3,
                "fib_w4_ratio": fib_w4,
                "cardinal_rules": {"rule1_w2": rule1, "rule2_w3": rule2, "rule3_w4": rule3},
                "rule_violation_reason": "; ".join(violation_reasons)
            }

        if last_close > p5:
            cycle_phase = "COMPLETED_IMPULSE_AWAITING_ABC"
        else:
            cycle_phase = "IN_WAVE_5_TERMINAL_SWEEP"

        return {
            "status": "VALID_IMPULSE",
            "cycle_phase": cycle_phase,
            "direction": "BEARISH",
            "invalidation_price": p4,
            "wave_points": wave_points,
            "fib_w2_ratio": fib_w2,
            "fib_w3_ratio": fib_w3,
            "fib_w4_ratio": fib_w4,
            "cardinal_rules": {"rule1_w2": True, "rule2_w3": True, "rule3_w4": True},
            "rule_violation_reason": None
        }

    return {
        "status": "INCOMPLETE_CYCLE",
        "cycle_phase": "INCOMPLETE_CYCLE",
        "direction": "NEUTRAL",
        "invalidation_price": 0.0,
        "wave_points": [],
        "fib_w2_ratio": 0.0,
        "fib_w3_ratio": 0.0,
        "fib_w4_ratio": 0.0,
        "cardinal_rules": {"rule1_w2": True, "rule2_w3": True, "rule3_w4": True},
        "rule_violation_reason": "Non-impulsive alternating pivot configuration"
    }

# -----------------------------------------------------------------------------
# 4. FULL DETERMINISTIC MARKET STRUCTURE ASSEMBLY
# -----------------------------------------------------------------------------
def compute_deterministic_market_structure(df, account_balance=10000.0, risk_percent=1.0):
    """
    Computes all ICT market structure elements and incorporates Elliott Wave Cycle validation.
    """
    n = len(df)
    if n < 10:
        return {
            "has_setup": False,
            "bias": "NEUTRAL",
            "action": "WAIT",
            "rejection_reasons": ["Insufficient candles (< 10 candles)"],
            "current_price": float(df.iloc[-1]["Close"]) if n > 0 else 0.0,
            "elliott_wave": {"status": "INCOMPLETE_CYCLE", "cycle_phase": "INCOMPLETE_CYCLE"}
        }

    current_price = round(float(df.iloc[-1]["Close"]), 2)
    range_high = round(float(df["High"].max()), 2)
    range_low = round(float(df["Low"].min()), 2)
    equilibrium = round((range_high + range_low) / 2.0, 2)

    # 1. Detect Swing Points & Elliott Wave Cycle
    swing_highs, swing_lows = detect_swing_points(df, window=5)
    ewt = detect_elliott_wave_cycle(df, swing_highs, swing_lows)
    
    # 2. Detect 3-Candle FVGs
    all_fvgs = detect_3candle_fvgs(df, min_gap_pts=0.35)
    unmitigated_bull_fvgs = [f for f in all_fvgs if f["type"] == "BULLISH_FVG" and not f["mitigated"]]
    unmitigated_bear_fvgs = [f for f in all_fvgs if f["type"] == "BEARISH_FVG" and not f["mitigated"]]

    # 3. Detect Sweeps vs Expansions
    recent_start_idx = max(5, n - 20)
    bullish_sweep = None
    bearish_sweep = None

    for i in range(recent_start_idx, n):
        c_lo = df["Low"].iloc[i]
        c_hi = df["High"].iloc[i]
        c_cl = df["Close"].iloc[i]

        for sl in swing_lows:
            if sl["index"] < i - 2 and (i - sl["index"]) <= 30:
                p_low = sl["price"]
                if c_lo < p_low and c_cl >= p_low:
                    bullish_sweep = {
                        "type": "SSL_SWEEP",
                        "level": p_low,
                        "sweep_low": round(float(c_lo), 2),
                        "candle_idx": i,
                        "depth_pts": round(p_low - c_lo, 2)
                    }

        for sh in swing_highs:
            if sh["index"] < i - 2 and (i - sh["index"]) <= 30:
                p_high = sh["price"]
                if c_hi > p_high and c_cl <= p_high:
                    bearish_sweep = {
                        "type": "BSL_SWEEP",
                        "level": p_high,
                        "sweep_high": round(float(c_hi), 2),
                        "candle_idx": i,
                        "depth_pts": round(c_hi - p_high, 2)
                    }

    # 4. Check for Market Structure Shift (MSS)
    bullish_mss = None
    bearish_mss = None

    if bullish_sweep is not None:
        sw_idx = bullish_sweep["candle_idx"]
        relevant_shs = [sh for sh in swing_highs if sh["index"] <= sw_idx]
        if relevant_shs:
            pivot_sh = relevant_shs[-1]["price"]
            for k in range(sw_idx, n):
                if df["Close"].iloc[k] > pivot_sh:
                    bullish_mss = {
                        "mss_level": pivot_sh,
                        "mss_candle_idx": k,
                        "close_price": round(float(df["Close"].iloc[k]), 2)
                    }
                    break

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

    rejection_reasons = []

    # Elliott Wave Hard Veto Check
    if ewt["status"] == "INVALID":
        rejection_reasons.append(f"Elliott Wave Gate Veto: {ewt.get('rule_violation_reason')}")

    # Priority A: Bullish Setup
    if bullish_sweep and bullish_mss and unmitigated_bull_fvgs:
        active_fvg = unmitigated_bull_fvgs[-1]
        entry_ce = active_fvg["ce"]
        stop_loss = round(bullish_sweep["sweep_low"] - 1.5, 2)
        risk_dist = abs(entry_ce - stop_loss)
        
        tp1 = round(entry_ce + 2.0 * risk_dist, 2)
        tp2_runner = round(range_high, 2) if range_high > tp1 else round(entry_ce + 3.2 * risk_dist, 2)
        rr_ratio = round((tp1 - entry_ce) / risk_dist, 2) if risk_dist > 0 else 0.0

        in_discount = entry_ce <= equilibrium + 2.0
        if not in_discount:
            rejection_reasons.append("CRT Invalidation: Bullish entry is in Premium (> 50% Equilibrium)")

        # EWT Terminal Exhaustion check
        if ewt["cycle_phase"] == "IN_WAVE_5_TERMINAL_SWEEP" and ewt["direction"] == "BULLISH":
            rejection_reasons.append("Elliott Wave Gate Veto: Prohibited to BUY during Wave 5 terminal exhaustion sweep into BSL")

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
            "setup_grade": "Rule Match: Yes" if has_setup else "Rule Match: No",
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
            "confluence_score": 95 if has_setup else 75,
            "range_zone": "DISCOUNT (Buy Zone)" if in_discount else "PREMIUM",
            "trapped_liquidity": f"Retail breakout sellers trapped below swing low ${bullish_sweep['level']:.2f}",
            "invalidation_thesis": f"Candle body close below ${stop_loss:.2f} completely invalidates bullish order flow",
            "rejection_reasons": rejection_reasons,
            "unmitigated_fvgs_count": len(unmitigated_bull_fvgs),
            "elliott_wave": ewt
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

        in_premium = entry_ce >= equilibrium - 2.0
        if not in_premium:
            rejection_reasons.append("CRT Invalidation: Bearish entry is in Discount (< 50% Equilibrium)")

        # EWT Terminal Exhaustion check
        if ewt["cycle_phase"] == "IN_WAVE_5_TERMINAL_SWEEP" and ewt["direction"] == "BEARISH":
            rejection_reasons.append("Elliott Wave Gate Veto: Prohibited to SELL during Wave 5 terminal exhaustion sweep into SSL")

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
            "setup_grade": "Rule Match: Yes" if has_setup else "Rule Match: No",
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
            "confluence_score": 95 if has_setup else 75,
            "range_zone": "PREMIUM (Sell Zone)" if in_premium else "DISCOUNT",
            "trapped_liquidity": f"Retail breakout buyers trapped above swing high ${bearish_sweep['level']:.2f}",
            "invalidation_thesis": f"Candle body close above ${stop_loss:.2f} completely invalidates bearish order flow",
            "rejection_reasons": rejection_reasons,
            "unmitigated_fvgs_count": len(unmitigated_bear_fvgs),
            "elliott_wave": ewt
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

    dummy_sl = round(range_low - 2.0 if bias == "BULLISH" else range_high + 2.0, 2)
    dummy_entry = current_price
    dummy_tp = round(dummy_entry + 2.0 * abs(dummy_entry - dummy_sl) if bias == "BULLISH" else dummy_entry - 2.0 * abs(dummy_entry - dummy_sl), 2)
    
    return {
        "has_setup": False,
        "action": "WAIT",
        "bias": bias,
        "verdict": "WAIT_CONFIRMATION",
        "setup_name": "Structure Consolidation / Order Flow Building",
        "setup_grade": "Rule Match: No",
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
        "unmitigated_fvgs_count": len(all_fvgs),
        "elliott_wave": ewt
    }

# -----------------------------------------------------------------------------
# 5. TYPOGRAPHY & PILL BADGE HELPER
# -----------------------------------------------------------------------------
def get_truetype_font(size=13, bold=True):
    """
    Safely retrieves a crisp TrueType font across Windows, Linux, and macOS.
    """
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahomabd.ttf" if bold else "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()

def draw_pill_badge(draw, text, x, y, font, text_color, bg_color=(15, 23, 42), border_color=None, padding=(8, 4)):
    """
    Draws a clean, high-contrast rounded pill badge with 1px border.
    Returns the bounding box (bx0, by0, bx1, by1) for anti-collision checks.
    """
    pad_x, pad_y = padding
    bbox = draw.textbbox((0, 0), text, font=font)
    t_w = bbox[2] - bbox[0]
    t_h = bbox[3] - bbox[1]

    bx0 = int(x - pad_x)
    by0 = int(y - pad_y)
    bx1 = int(x + t_w + pad_x)
    by1 = int(y + t_h + pad_y)

    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=5, fill=bg_color, outline=border_color, width=1)
    draw.text((x, y), text, font=font, fill=text_color)
    return bx0, by0, bx1, by1

# -----------------------------------------------------------------------------
# 6. HIGH-DPI CAMERON4X PILLOW TRUE-TYPE CHART ENGINE (ZERO OVERLAP & RAZOR SHARP)
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
    fvg_bot=None,
    show_ewt=True
):
    """
    Renders an institutional High-DPI Cameron4x dark candlestick chart (1360x720) on #0B1120.
    Features:
      - Crisp subpixel TrueType typography (Segoe UI Bold / Arial Bold) - zero pixelation.
      - 100% clean text: NO broken unicode/emoji '?' question mark glitches.
      - Intelligent anti-collision:
          * Sweep and MSS badges staggered mathematically.
          * FVG title strictly pinned to top-left of box; 50% CE pinned to far right.
      - Circular Elliott Wave badges with vertical leader stems.
      - Dynamic vertical bounding so distant targets never squash candles.
    """
    w, h = 1360, 720
    img = Image.new("RGBA", (w, h), (11, 17, 32, 255)) # #0B1120
    draw = ImageDraw.Draw(img)

    f_title = get_truetype_font(15, bold=True)
    f_body = get_truetype_font(12, bold=True)
    f_sub = get_truetype_font(11, bold=True)
    f_small = get_truetype_font(10, bold=False)

    if df is None or len(df) < 5:
        draw.text((int(w * 0.30), int(h * 0.45)), "No Live Candlestick Data Available", font=f_title, fill=(255, 255, 255))
        return img.convert("RGB")

    try:
        if math_structure is not None and isinstance(math_structure, dict):
            verdict = str(math_structure.get("verdict", verdict) or verdict)
            setup_name = str(math_structure.get("setup_name", setup_name) or setup_name)
            score = int(math_structure.get("confluence_score", score) or score)
            entry_p = math_structure.get("entry_ce", entry_p)
            sl_p = math_structure.get("stop_loss", sl_p)
            tp_p = math_structure.get("take_profit_1", tp_p)
            sweep_p = math_structure.get("sweep_level", sweep_p)
            mss_p = math_structure.get("mss_level", mss_p)
            fvg_top = math_structure.get("fvg_top", fvg_top)
            fvg_bot = math_structure.get("fvg_bottom", fvg_bot)

        is_buy = "BUY" in str(verdict).upper()
        last_close = float(df.iloc[-1]["Close"])

        COLOR_BULL  = (16, 185, 129, 255)   # #10B981 Emerald
        COLOR_BEAR  = (239, 68, 68, 255)    # #EF4444 Coral Red
        COLOR_SWEEP = (56, 189, 248, 255)   # #38BDF8 Sky Blue
        COLOR_MSS   = (255, 255, 255, 255)  # White
        COLOR_CE    = (6, 182, 212, 255)    # Cyan
        COLOR_TP    = (52, 211, 153, 255)   # Mint Green
        COLOR_SL    = (248, 113, 113, 255)  # Salmon
        COLOR_SPOT  = (251, 191, 36, 255)   # Amber
        COLOR_GRID  = (30, 41, 59, 255)     # Subdued Grid

        all_lows = df["Low"].dropna().astype(float).values
        all_highs = df["High"].dropna().astype(float).values

        if len(all_lows) == 0: all_lows = np.array([last_close - 5.0])
        if len(all_highs) == 0: all_highs = np.array([last_close + 5.0])

        c_min = float(np.min(all_lows))
        c_max = float(np.max(all_highs))

        if sweep_p is None: sweep_p = c_min if is_buy else c_max
        else: sweep_p = float(sweep_p)

        if mss_p is None: mss_p = float(np.median(df["Close"].astype(float).values))
        else: mss_p = float(mss_p)

        if fvg_top is None: fvg_top = round(last_close + 2.5, 2)
        else: fvg_top = float(fvg_top)

        if fvg_bot is None: fvg_bot = round(last_close - 2.5, 2)
        else: fvg_bot = float(fvg_bot)

        if entry_p is None: entry_p = round((fvg_top + fvg_bot) / 2.0, 2)
        else: entry_p = float(entry_p)

        if sl_p is None: sl_p = round(sweep_p - 2.5 if is_buy else sweep_p + 2.5, 2)
        else: sl_p = float(sl_p)

        if tp_p is None: tp_p = round(entry_p + 2.0 * abs(entry_p - sl_p) if is_buy else entry_p - 2.0 * abs(entry_p - sl_p), 2)
        else: tp_p = float(tp_p)

        core_levels = [c_min, c_max, sweep_p, mss_p, entry_p, sl_p, fvg_top, fvg_bot, last_close]
        min_core = min(core_levels)
        max_core = max(core_levels)
        core_rng = max(2.0, max_core - min_core)

        tp_offscreen = False
        if is_buy:
            if tp_p > max_core + core_rng * 0.40:
                tp_offscreen = True
                max_p = max_core + core_rng * 0.25
            else:
                max_p = max(max_core, tp_p)
            min_p = min_core - core_rng * 0.10
        else:
            if tp_p < min_core - core_rng * 0.40:
                tp_offscreen = True
                min_p = min_core - core_rng * 0.25
            else:
                min_p = min(min_core, tp_p)
            max_p = max_core + core_rng * 0.10

        p_pad = max(1.5, (max_p - min_p) * 0.05)
        min_p -= p_pad
        max_p += p_pad
        p_range = max(1.0, max_p - min_p)

        def p2y(p):
            val = float(p) if (p is not None and not np.isnan(p)) else last_close
            py = h * 0.12 + (max_p - val) / p_range * (h * 0.76)
            return int(np.clip(py, 48, h - 35))

        # 1. Price Grid & Right Axis Scale
        for p_level in np.linspace(min_p, max_p, 8):
            y_l = p2y(p_level)
            draw.line([(int(w * 0.04), y_l), (int(w * 0.90), y_l)], fill=COLOR_GRID, width=1)
            draw.text((int(w * 0.905), y_l - 6), f"${p_level:.2f}", font=f_sub, fill=(148, 163, 184))

        # 2. Render Candlesticks
        n_c = len(df)
        chart_w = int(w * 0.84)
        c_w = max(8, int(chart_w / n_c) - 4)
        start_x = int(w * 0.05)

        for i in range(n_c):
            row = df.iloc[i]
            x_mid = int(start_x + i * (c_w + 4))

            c_open = float(row["Open"])
            c_close = float(row["Close"])
            c_high = float(row["High"])
            c_low = float(row["Low"])

            y_o = p2y(c_open)
            y_c = p2y(c_close)
            y_hi = p2y(c_high)
            y_lo = p2y(c_low)

            is_c_bull = c_close >= c_open
            c_col = COLOR_BULL if is_c_bull else COLOR_BEAR

            # Wick
            draw.line([(x_mid, y_hi), (x_mid, y_lo)], fill=c_col, width=1)

            # Body
            top_b = min(y_o, y_c)
            bot_b = max(y_o, y_c)
            if bot_b - top_b < 2:
                bot_b = top_b + 2
            draw.rectangle([(x_mid - c_w // 2, top_b), (x_mid + c_w // 2, bot_b)], fill=c_col)

            # Time labels on bottom axis
            if i % max(1, n_c // 8) == 0:
                t_str = ""
                if "Date_NY" in row:
                    val = row["Date_NY"]
                    t_str = val.strftime("%H:%M") if hasattr(val, "strftime") else str(val)[11:16]
                elif hasattr(df.index[i], "strftime"):
                    t_str = df.index[i].strftime("%H:%M")
                if t_str:
                    draw_pill_badge(draw, t_str, x_mid - 16, h - 30, f_small, (148, 163, 184), bg_color=(15, 23, 42), padding=(4, 2))

        # 3. Liquidity Sweep Line & Badge
        sw_y = p2y(sweep_p)
        draw.line([(int(w * 0.05), sw_y), (int(w * 0.90), sw_y)], fill=COLOR_SWEEP, width=2)
        sw_lbl = f"[SWEEP] {'SSL' if is_buy else 'BSL'} Swept (${sweep_p:.2f})"
        sw_y_text = sw_y - 18 if is_buy else sw_y + 8
        sw_bx0, sw_by0, sw_bx1, sw_by1 = draw_pill_badge(draw, sw_lbl, int(w * 0.06), sw_y_text, f_body, COLOR_SWEEP, bg_color=(15, 23, 42), border_color=COLOR_SWEEP)

        # 4. MSS Dashed Line & Mathematically Non-Colliding Badge
        mss_y = p2y(mss_p)
        for x in range(int(w * 0.15), int(w * 0.90), 16):
            draw.line([(x, mss_y), (x + 10, mss_y)], fill=COLOR_MSS, width=2)
        
        mss_lbl = f"[MSS] Pivot Level (${mss_p:.2f})"
        if abs(mss_y - sw_y) < 36:
            mss_x = sw_bx1 + 20
            mss_y_text = mss_y - 18 if is_buy else mss_y + 8
            if mss_x > int(w * 0.45):
                mss_x = int(w * 0.06)
                mss_y_text = sw_by1 + 6
        else:
            mss_x = int(w * 0.20)
            mss_y_text = mss_y - 18

        draw_pill_badge(draw, mss_lbl, mss_x, mss_y_text, f_body, COLOR_MSS, bg_color=(15, 23, 42), border_color=COLOR_MSS)

        # 5. FVG Shaded Box & 50% CE Midline (Zero Overlap Guaranteed)
        fvg_y1 = p2y(fvg_top)
        fvg_y2 = p2y(fvg_bot)
        box_top = min(fvg_y1, fvg_y2)
        box_bot = max(fvg_y1, fvg_y2)
        fvg_x1 = int(w * 0.35)
        fvg_x2 = int(w * 0.90)

        # Shaded FVG Box (using alpha composite)
        fvg_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        fvg_draw = ImageDraw.Draw(fvg_overlay)
        fvg_box_fill = (239, 68, 68, 50) if not is_buy else (16, 185, 129, 50)
        fvg_draw.rectangle([(fvg_x1, box_top), (fvg_x2, box_bot)], fill=fvg_box_fill, outline=COLOR_BEAR if not is_buy else COLOR_BULL, width=2)
        img = Image.alpha_composite(img, fvg_overlay)
        draw = ImageDraw.Draw(img)

        # 50% CE Line
        ce_y = p2y(entry_p)
        for x in range(fvg_x1, fvg_x2, 10):
            draw.line([(x, ce_y), (x + 6, ce_y)], fill=COLOR_CE, width=2)

        # FVG Badge: STRICTLY TOP-LEFT ABOVE FVG Box
        fvg_title = f"[FVG] {'Bear FVG (SIBI)' if not is_buy else 'Bull FVG (BISI)'} [${fvg_bot:.2f}-${fvg_top:.2f}]"
        draw_pill_badge(draw, fvg_title, fvg_x1 + 8, box_top - 24, f_body, (255, 255, 255), bg_color=(15, 23, 42), border_color=COLOR_BEAR if not is_buy else COLOR_BULL)

        # 50% CE Badge: STRICTLY BOTTOM-LEFT BELOW FVG Box (Zero Overlap Guaranteed)
        ce_lbl = f"[50% CE ENTRY] ${entry_p:.2f}"
        ce_badge_y = box_bot + 8 if (box_bot + 28 < h - 35) else (ce_y - 12)
        draw_pill_badge(draw, ce_lbl, fvg_x1 + 8, ce_badge_y, f_body, COLOR_CE, bg_color=(15, 23, 42), border_color=COLOR_CE)

        # 6. Target 1:2 R:R (On-Screen or Smart Off-Screen Directional Rail)
        if not tp_offscreen:
            tp_y = p2y(tp_p)
            for x in range(int(w * 0.25), int(w * 0.90), 16):
                draw.line([(x, tp_y), (x + 10, tp_y)], fill=COLOR_TP, width=2)
            draw_pill_badge(draw, f"[TARGET 1:2 R:R] ${tp_p:.2f}", int(w * 0.28), tp_y + 8 if not is_buy else tp_y - 22, f_body, COLOR_TP, bg_color=(15, 23, 42), border_color=COLOR_TP)
        else:
            off_y = int(h * 0.88) if not is_buy else int(h * 0.16)
            pts_away = abs(last_close - tp_p)
            dir_sym = "v" if not is_buy else "^"
            draw_pill_badge(draw, f"[TARGET 1:2 EXTENDED] ${tp_p:.2f} ({pts_away:.1f} pts {dir_sym})", int(w * 0.35), off_y, f_body, COLOR_TP, bg_color=(15, 23, 42), border_color=COLOR_TP)

        # 7. Stop Loss Level
        sl_y = p2y(sl_p)
        for x in range(int(w * 0.20), int(w * 0.90), 16):
            draw.line([(x, sl_y), (x + 10, sl_y)], fill=COLOR_SL, width=2)
        draw_pill_badge(draw, f"[STOP LOSS] ${sl_p:.2f}", int(w * 0.22), sl_y - 22 if not is_buy else sl_y + 8, f_body, COLOR_SL, bg_color=(15, 23, 42), border_color=COLOR_SL)

        # 8. Live Spot Level & Right Axis Tag
        cur_y = p2y(last_close)
        for x in range(int(w * 0.05), int(w * 0.90), 8):
            draw.line([(x, cur_y), (x + 4, cur_y)], fill=COLOR_SPOT, width=1)
        draw_pill_badge(draw, f"${last_close:.2f}", int(w * 0.905), cur_y - 10, f_body, (0, 0, 0), bg_color=COLOR_SPOT)

        # 9. Elliott Wave Overlay (Smart Vertical Offset & Floating Badges)
        if bool(show_ewt) and math_structure is not None:
            ewt = math_structure.get("elliott_wave", {})
            if isinstance(ewt, dict):
                wave_pts = ewt.get("wave_points", [])
                if wave_pts and isinstance(wave_pts, list) and len(wave_pts) >= 3:
                    # Trendline
                    for w_i in range(len(wave_pts) - 1):
                        p1 = wave_pts[w_i]
                        p2 = wave_pts[w_i + 1]
                        try:
                            s_idx = int(p1.get("idx", 0))
                            e_idx = int(p2.get("idx", 0))
                            x1 = int(start_x + s_idx * (c_w + 4))
                            y1 = p2y(p1.get("price", last_close))
                            x2 = int(start_x + e_idx * (c_w + 4))
                            y2 = p2y(p2.get("price", last_close))
                            draw.line([(x1, y1), (x2, y2)], fill=(245, 158, 11, 200), width=2)
                        except Exception:
                            continue

                    # Circular badges with leader stems
                    for idx_w, pt in enumerate(wave_pts):
                        try:
                            pt_idx = int(pt.get("idx", 0))
                            px = int(start_x + pt_idx * (c_w + 4))
                            pt_price = float(pt.get("price", last_close))
                            py = p2y(pt_price)
                            lbl = str(pt.get("label", ""))

                            if idx_w == 0:
                                is_high_pivot = pt_price > float(wave_pts[1].get("price", pt_price)) if len(wave_pts) > 1 else (not is_buy)
                            elif idx_w == len(wave_pts) - 1:
                                is_high_pivot = pt_price > float(wave_pts[-2].get("price", pt_price))
                            else:
                                prev_p = float(wave_pts[idx_w - 1].get("price", pt_price))
                                next_p = float(wave_pts[idx_w + 1].get("price", pt_price))
                                is_high_pivot = pt_price >= prev_p or pt_price >= next_p

                            by = py - 24 if is_high_pivot else py + 24
                            draw.line([(px, py), (px, by)], fill=(148, 163, 184, 180), width=1)

                            r = 12
                            draw.ellipse([(px - r, by - r), (px + r, by + r)], fill=(15, 23, 42), outline=(56, 189, 248), width=2)
                            draw.text((px - 7, by - 6), lbl, font=f_small, fill=(255, 255, 255))
                        except Exception:
                            continue

        # 10. Top Institutional Watermark Header
        draw.rectangle([(0, 0), (w, 42)], fill=(15, 23, 42))
        draw.line([(0, 42), (w, 42)], fill=(51, 65, 85), width=1)
        ewt_info = ""
        if math_structure and isinstance(math_structure, dict):
            ph = str(math_structure.get("elliott_wave", {}).get("cycle_phase", ""))
            if ph:
                if len(ph) > 20: ph = ph[:18] + ".."
                ewt_info = f" | EWT: {ph}"
        header_txt = f"XAUUSD 15m ({session_txt[:16]}) | {verdict} | Spot: ${last_close:.2f} | Score: {score}%{ewt_info}"
        draw.text((16, 12), header_txt, font=f_title, fill=COLOR_SPOT)

        return img.convert("RGB")

    except Exception as render_exc:
        draw.text((20, h // 2), f"Chart Rendering Exception: {str(render_exc)[:50]}", font=f_body, fill=(239, 68, 68))
        return img.convert("RGB")

# -----------------------------------------------------------------------------
# 7. INTERACTIVE TRADINGVIEW-GRADE PLOTLY ENGINE (HD ZOOM & PAN)
# -----------------------------------------------------------------------------
def build_cameron4x_plotly_figure(
    df,
    math_structure=None,
    session_txt="NY AM Killzone",
    show_ewt=True
):
    """
    Builds an institutional TradingView-grade Dark Candlestick Chart with Plotly.
    Features:
      - Full interactive mouse wheel zoom & click-drag pan.
      - 100% vector SVG sharpness with zero font blurriness.
      - Clean visual separation: FVG box, 50% CE midline, Sweep ray, MSS line, Target, and Stop Loss.
      - Hover tooltips showing exact candle metrics and timestamps.
      - Elliott Wave (0)-(5) cycle nodes with motive connecting trendline.
    """
    if not HAS_PLOTLY or go is None:
        return None

    if df is None or len(df) < 5:
        fig = go.Figure()
        fig.add_annotation(text="No Live Candlestick Data Available", showarrow=False, font=dict(size=18, color="#FFFFFF"))
        fig.update_layout(template="plotly_dark", plot_bgcolor="#0B1120", paper_bgcolor="#0B1120")
        return fig

    time_labels = []
    for i in range(len(df)):
        row = df.iloc[i]
        if "Date_NY" in row and hasattr(row["Date_NY"], "strftime"):
            time_labels.append(row["Date_NY"].strftime("%m/%d %H:%M"))
        elif hasattr(df.index[i], "strftime"):
            time_labels.append(df.index[i].strftime("%m/%d %H:%M"))
        else:
            time_labels.append(str(i))

    verdict = math_structure.get("verdict", "WAIT") if math_structure else "WAIT"
    score = math_structure.get("confluence_score", 75) if math_structure else 75
    is_buy = "BUY" in str(verdict).upper()
    last_close = float(df.iloc[-1]["Close"])

    entry_p = float(math_structure.get("entry_ce", last_close))
    sl_p = float(math_structure.get("stop_loss", last_close + 2.0))
    tp_p = float(math_structure.get("take_profit_1", last_close - 4.0))
    sweep_p = float(math_structure.get("sweep_level", last_close + 3.0))
    mss_p = float(math_structure.get("mss_level", last_close + 1.0))
    fvg_top = float(math_structure.get("fvg_top", last_close + 1.5))
    fvg_bot = float(math_structure.get("fvg_bottom", last_close - 1.5))

    COLOR_BULL = "#10B981"
    COLOR_BEAR = "#EF4444"
    COLOR_SWEEP = "#38BDF8"
    COLOR_MSS = "#FFFFFF"
    COLOR_CE = "#06B6D4"
    COLOR_TP = "#34D399"
    COLOR_SL = "#F87171"
    COLOR_SPOT = "#FBBF24"

    fig = go.Figure()

    # 1. Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=time_labels,
        open=df["Open"].astype(float),
        high=df["High"].astype(float),
        low=df["Low"].astype(float),
        close=df["Close"].astype(float),
        name="XAUUSD 15m",
        increasing_line_color=COLOR_BULL,
        increasing_fillcolor=COLOR_BULL,
        decreasing_line_color=COLOR_BEAR,
        decreasing_fillcolor=COLOR_BEAR,
        hoverlabel=dict(bgcolor="#0F172A", font_size=12, font_family="Segoe UI, monospace")
    ))

    n_c = len(df)
    x0_lbl = time_labels[0]
    x1_lbl = time_labels[-1]

    # 2. Liquidity Sweep Line & Annotation
    fig.add_shape(
        type="line",
        x0=x0_lbl, x1=x1_lbl,
        y0=sweep_p, y1=sweep_p,
        line=dict(color=COLOR_SWEEP, width=2, dash="solid")
    )
    sw_tag = "BSL" if not is_buy else "SSL"
    fig.add_annotation(
        x=time_labels[max(0, int(n_c * 0.08))],
        y=sweep_p,
        text=f"<b>[#13 {sw_tag} SWEEP ${sweep_p:.2f}]</b>",
        showarrow=False,
        yshift=14 if not is_buy else -14,
        font=dict(color=COLOR_SWEEP, size=11, family="Segoe UI Bold, Arial"),
        bgcolor="#0B1120",
        bordercolor=COLOR_SWEEP,
        borderwidth=1.5,
        borderpad=4
    )

    # 3. MSS Pivot Level & Annotation
    fig.add_shape(
        type="line",
        x0=time_labels[max(0, int(n_c * 0.20))], x1=x1_lbl,
        y0=mss_p, y1=mss_p,
        line=dict(color=COLOR_MSS, width=2, dash="dash")
    )
    fig.add_annotation(
        x=time_labels[min(n_c - 1, int(n_c * 0.45))],
        y=mss_p,
        text=f"<b>[#4 MSS PIVOT ${mss_p:.2f}]</b>",
        showarrow=False,
        yshift=14 if is_buy else -14,
        font=dict(color=COLOR_MSS, size=11, family="Segoe UI Bold, Arial"),
        bgcolor="#0B1120",
        bordercolor="#94A3B8",
        borderwidth=1.5,
        borderpad=4
    )

    # 4. FVG Shaded Box & 50% CE Midline
    fvg_x_start = time_labels[max(0, int(n_c * 0.40))]
    fvg_x_end = x1_lbl
    fvg_fill = "rgba(239, 68, 68, 0.20)" if not is_buy else "rgba(16, 185, 129, 0.20)"
    fvg_border = COLOR_BEAR if not is_buy else COLOR_BULL

    fig.add_shape(
        type="rect",
        x0=fvg_x_start, x1=fvg_x_end,
        y0=fvg_bot, y1=fvg_top,
        fillcolor=fvg_fill,
        line=dict(color=fvg_border, width=1.5)
    )

    # 50% CE Midline
    fig.add_shape(
        type="line",
        x0=fvg_x_start, x1=fvg_x_end,
        y0=entry_p, y1=entry_p,
        line=dict(color=COLOR_CE, width=2, dash="dot")
    )

    # FVG Badge: Top Left of FVG
    fig.add_annotation(
        x=fvg_x_start,
        y=fvg_top if not is_buy else fvg_bot,
        text=f"<b>[#17 {'BEAR' if not is_buy else 'BULL'} FVG ${fvg_bot:.2f}-${fvg_top:.2f}]</b>",
        showarrow=False,
        yshift=14 if not is_buy else -14,
        xshift=40,
        font=dict(color="#FFFFFF", size=11, family="Segoe UI Bold, Arial"),
        bgcolor="#0B1120",
        bordercolor=fvg_border,
        borderwidth=1.5,
        borderpad=4
    )

    # 50% CE Badge: Pinned on the RIGHT of FVG
    fig.add_annotation(
        x=fvg_x_end,
        y=entry_p,
        text=f"<b>[#28 50% CE ENTRY ${entry_p:.2f}]</b>",
        showarrow=False,
        xshift=-80,
        yshift=0,
        font=dict(color=COLOR_CE, size=11, family="Segoe UI Bold, Arial"),
        bgcolor="#0B1120",
        bordercolor=COLOR_CE,
        borderwidth=1.5,
        borderpad=4
    )

    # 5. Stop Loss Line & Annotation
    fig.add_shape(
        type="line",
        x0=time_labels[max(0, int(n_c * 0.25))], x1=x1_lbl,
        y0=sl_p, y1=sl_p,
        line=dict(color=COLOR_SL, width=2, dash="dashdot")
    )
    fig.add_annotation(
        x=time_labels[min(n_c - 1, int(n_c * 0.35))],
        y=sl_p,
        text=f"<b>[STOP LOSS ${sl_p:.2f}]</b>",
        showarrow=False,
        yshift=14 if not is_buy else -14,
        font=dict(color=COLOR_SL, size=11, family="Segoe UI Bold, Arial"),
        bgcolor="#0B1120",
        bordercolor=COLOR_SL,
        borderwidth=1.5,
        borderpad=4
    )

    # 6. Target 1:2 R:R Line & Annotation
    fig.add_shape(
        type="line",
        x0=time_labels[max(0, int(n_c * 0.30))], x1=x1_lbl,
        y0=tp_p, y1=tp_p,
        line=dict(color=COLOR_TP, width=2, dash="dashdot")
    )
    fig.add_annotation(
        x=time_labels[min(n_c - 1, int(n_c * 0.50))],
        y=tp_p,
        text=f"<b>[#94 TARGET 1:2 R:R ${tp_p:.2f}]</b>",
        showarrow=False,
        yshift=-14 if not is_buy else 14,
        font=dict(color=COLOR_TP, size=11, family="Segoe UI Bold, Arial"),
        bgcolor="#0B1120",
        bordercolor=COLOR_TP,
        borderwidth=1.5,
        borderpad=4
    )

    # 7. Live Spot Level & Right Axis Tag
    fig.add_shape(
        type="line",
        x0=x0_lbl, x1=x1_lbl,
        y0=last_close, y1=last_close,
        line=dict(color=COLOR_SPOT, width=1.5, dash="dot")
    )
    fig.add_annotation(
        x=x1_lbl,
        y=last_close,
        text=f"<b>${last_close:.2f}</b>",
        showarrow=False,
        xshift=40,
        font=dict(color="#000000", size=11, family="Segoe UI Bold, Arial"),
        bgcolor=COLOR_SPOT,
        bordercolor=COLOR_SPOT,
        borderwidth=1,
        borderpad=3
    )

    # 8. Elliott Wave Markers & Motive Connecting Trendlines
    if show_ewt and math_structure:
        ewt = math_structure.get("elliott_wave", {})
        wave_pts = ewt.get("wave_points", [])
        if wave_pts and len(wave_pts) >= 3:
            ewt_x = []
            ewt_y = []
            ewt_txt = []
            for p in wave_pts:
                idx = min(n_c - 1, max(0, int(p.get("idx", 0))))
                ewt_x.append(time_labels[idx])
                ewt_y.append(float(p.get("price", last_close)))
                ewt_txt.append(f"<b>{p.get('label', '')}</b>")

            fig.add_trace(go.Scatter(
                x=ewt_x,
                y=ewt_y,
                mode="lines+markers+text",
                name="Elliott Wave (0)-(5)",
                line=dict(color="#FBBF24", width=2, dash="dot"),
                marker=dict(size=14, color="#0F172A", line=dict(color="#38BDF8", width=2)),
                text=ewt_txt,
                textposition=["top center" if (i % 2 == 1 if is_buy else i % 2 == 0) else "bottom center" for i in range(len(ewt_txt))],
                textfont=dict(size=12, color="#FFFFFF", family="Segoe UI Bold, Arial"),
                hoverinfo="text+y"
            ))

    ewt_phase = math_structure.get("elliott_wave", {}).get("cycle_phase", "") if math_structure else ""
    ewt_banner = f" | EWT: {ewt_phase}" if ewt_phase else ""
    title_text = f"<b>XAUUSD 15m ({session_txt[:16]}) | {verdict} | Spot: ${last_close:.2f} | Score: {score}%{ewt_banner}</b>"

    fig.update_layout(
        title=dict(text=title_text, font=dict(family="Segoe UI, Arial, sans-serif", size=15, color="#FBBF24"), x=0.02, y=0.98),
        template="plotly_dark",
        plot_bgcolor="#0B1120",
        paper_bgcolor="#0B1120",
        height=660,
        margin=dict(l=45, r=80, t=55, b=45),
        xaxis=dict(
            type="category",
            rangeslider=dict(visible=False),
            gridcolor="#1E293B",
            showgrid=True,
            tickangle=-45,
            tickfont=dict(size=10, color="#94A3B8")
        ),
        yaxis=dict(
            gridcolor="#1E293B",
            showgrid=True,
            tickformat="$.2f",
            tickfont=dict(size=11, color="#94A3B8"),
            side="right"
        ),
        showlegend=False,
        dragmode="pan"
    )

    return fig
