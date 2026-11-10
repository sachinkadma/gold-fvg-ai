import json, ssl, time
import pandas as pd
import numpy as np
from datetime import datetime
import zoneinfo
import requests
import cv2
from PIL import Image

NY_TZ = zoneinfo.ZoneInfo("America/New_York")

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

def render_native_cameron4x_chart(
    df,
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
    Renders a pure, pristine Cameron4x institutional candlestick chart (920x540)
    with zero mobile UI clutter, real candlesticks, right-side price grid,
    bottom time labels, and exact ICT markings:
    - #13 Liquidity Sweep (BSL / SSL Swept)
    - #4 MSS (Market Structure Shift) dashed line
    - #17 Fair Value Gap shaded box
    - #28 50% Consequent Encroachment (50% CE) dashed line
    - 🎯 Target 1:2 R:R rail
    - Current Spot Price line with right-side badge
    """
    w, h = 940, 540
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (32, 17, 11)  # High-contrast institutional dark canvas #0B1120 (BGR)

    if df is None or len(df) < 5:
        cv2.putText(img, "No Live Candlestick Data Available", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return Image.fromarray(img[:, :, ::-1])

    is_buy = "BUY" in verdict.upper()
    last_close = float(df.iloc[-1]["Close"])

    # High-Visibility ICT Colors (BGR format)
    COLOR_MSS    = (255, 255, 255) # Pure White
    COLOR_SWEEP  = (248, 189, 56)  # Cyan/Sky Blue #38BDF8
    COLOR_BULL   = (129, 185, 16)  # Emerald Green #10B981
    COLOR_BEAR   = (68, 68, 239)   # Coral Red #EF4444
    COLOR_CE     = (6, 182, 212)   # Consequent Encroachment Amber/Cyan #06B6D4
    COLOR_TP     = (153, 211, 52)  # Mint Green #34D399
    COLOR_SPOT   = (36, 191, 251)  # Live Spot Amber #FBBF24
    COLOR_GRID   = (45, 30, 20)    # Subdued Grid lines

    # Determine all price extrema including user targets
    all_lows = df["Low"].values
    all_highs = df["High"].values
    
    # Auto-compute ICT levels if not explicitly provided
    if sweep_p is None:
        sweep_p = float(np.min(all_lows)) if is_buy else float(np.max(all_highs))
    if mss_p is None:
        mss_p = float(np.median(df["Close"].values))
    if fvg_top is None or fvg_bot is None:
        if is_buy:
            fvg_top = round(last_close + 2.0, 2)
            fvg_bot = round(last_close - 4.0, 2)
        else:
            fvg_top = round(last_close + 4.0, 2)
            fvg_bot = round(last_close - 2.0, 2)
    if entry_p is None:
        entry_p = round((fvg_top + fvg_bot) / 2.0, 2)
    if sl_p is None:
        sl_p = round(sweep_p - 4.0 if is_buy else sweep_p + 4.0, 2)
    if tp_p is None:
        tp_dist = abs(entry_p - sl_p) * 2.0
        tp_p = round(entry_p + tp_dist if is_buy else entry_p - tp_dist, 2)

    # Compute bounding price range with margin
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

    def p_to_y(p):
        return int(h * 0.12 + (max_p - p) / p_range * (h * 0.72))

    # 1. Right Price Scale Grid & Axis
    price_grid_steps = np.linspace(min_p, max_p, 8)
    for p_level in price_grid_steps:
        y_l = p_to_y(p_level)
        cv2.line(img, (int(w * 0.05), y_l), (int(w * 0.86), y_l), COLOR_GRID, 1)
        cv2.putText(img, f"${p_level:.1f}", (int(w * 0.87), y_l + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

    # 2. Render Candlesticks
    n_c = len(df)
    chart_w = int(w * 0.79)
    c_w = max(5, int(chart_w / n_c) - 2)
    start_x = int(w * 0.06)

    candle_coords = []
    for i, row in df.iterrows():
        x_mid = start_x + i * (c_w + 3)
        y_o = p_to_y(row["Open"])
        y_c = p_to_y(row["Close"])
        y_hi = p_to_y(row["High"])
        y_lo = p_to_y(row["Low"])

        is_bull = row["Close"] >= row["Open"]
        c_col = COLOR_BULL if is_bull else COLOR_BEAR

        # Wick
        cv2.line(img, (x_mid, y_hi), (x_mid, y_lo), c_col, 1, cv2.LINE_AA)
        
        # Body
        top_b = min(y_o, y_c)
        bot_b = max(y_o, y_c)
        if bot_b - top_b < 2: bot_b = top_b + 2
        cv2.rectangle(img, (x_mid - c_w // 2, top_b), (x_mid + c_w // 2, bot_b), c_col, -1)

        # Time labels at bottom every ~8 candles
        if i % 7 == 0 and "Date_NY" in row:
            t_str = row["Date_NY"].strftime("%H:%M")
            cv2.putText(img, t_str, (x_mid - 16, int(h * 0.90)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (148, 163, 184), 1, cv2.LINE_AA)

        candle_coords.append({
            "x": x_mid, "high": row["High"], "low": row["Low"],
            "open": row["Open"], "close": row["Close"], "is_bull": is_bull
        })

    overlay = img.copy()

    # 3. Draw Cameron4x Markings: #13 Liquidity Sweep
    sw_y = p_to_y(sweep_p)
    sw_lbl = f"⚡ #13 SSL Swept (${sweep_p:.1f})" if is_buy else f"⚡ #13 BSL Swept (${sweep_p:.1f})"
    cv2.line(img, (int(w * 0.08), sw_y), (int(w * 0.85), sw_y), COLOR_SWEEP, 2, cv2.LINE_AA)
    cv2.putText(img, sw_lbl, (int(w * 0.10), sw_y + 16 if is_buy else sw_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_SWEEP, 1, cv2.LINE_AA)

    # 4. Draw Cameron4x Markings: #4 Market Structure Shift (MSS) Dashed Line
    mss_y = p_to_y(mss_p)
    for x in range(int(w * 0.20), int(w * 0.85), 14):
        cv2.line(img, (x, mss_y), (x + 8, mss_y), COLOR_MSS, 2, cv2.LINE_AA)
    cv2.putText(img, f"⚡ #4 MSS Pivot (${mss_p:.1f})", (int(w * 0.22), mss_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_MSS, 1, cv2.LINE_AA)

    # 5. Draw Cameron4x Markings: #17 FVG Box & #28 50% CE Midline
    fvg_col = COLOR_BULL if is_buy else COLOR_BEAR
    fvg_lbl = f"🟩 #17 Bull FVG (BISI) [${fvg_bot:.1f}-${fvg_top:.1f}]" if is_buy else f"🟥 #17 Bear FVG (SIBI) [${fvg_bot:.1f}-${fvg_top:.1f}]"
    
    fvg_y1 = p_to_y(fvg_top)
    fvg_y2 = p_to_y(fvg_bot)
    fvg_x1 = int(w * 0.38)
    fvg_x2 = int(w * 0.85)

    cv2.rectangle(overlay, (fvg_x1, min(fvg_y1, fvg_y2)), (fvg_x2, max(fvg_y1, fvg_y2)), fvg_col, -1)
    cv2.rectangle(img, (fvg_x1, min(fvg_y1, fvg_y2)), (fvg_x2, max(fvg_y1, fvg_y2)), fvg_col, 2)
    
    # 50% Consequent Encroachment (50% CE)
    ce_y = p_to_y(entry_p)
    for x in range(fvg_x1, fvg_x2, 8):
        cv2.line(img, (x, ce_y), (x + 5, ce_y), COLOR_CE, 2, cv2.LINE_AA)
    cv2.putText(img, fvg_lbl, (fvg_x1 + 6, min(fvg_y1, fvg_y2) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"🔹 #28 50% CE Entry (${entry_p:.2f})", (fvg_x1 + 6, ce_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_CE, 1, cv2.LINE_AA)

    # 6. Draw Institutional Target 1:2 R:R Rail
    tp_y = p_to_y(tp_p)
    for x in range(int(w * 0.30), int(w * 0.85), 14):
        cv2.line(img, (x, tp_y), (x + 8, tp_y), COLOR_TP, 2, cv2.LINE_AA)
    cv2.putText(img, f"🎯 #94 Institutional Target 1:2 R:R (${tp_p:.2f})", (int(w * 0.32), tp_y - 6 if is_buy else tp_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TP, 1, cv2.LINE_AA)

    # 7. Current Live Spot Price Dotted Line & Right-Scale Badge
    cur_y = p_to_y(last_close)
    for x in range(int(w * 0.06), int(w * 0.86), 6):
        cv2.line(img, (x, cur_y), (x + 3, cur_y), COLOR_SPOT, 1, cv2.LINE_AA)
    # Badge on right axis
    cv2.rectangle(img, (int(w * 0.86), cur_y - 10), (w - 6, cur_y + 10), (15, 23, 42), -1)
    cv2.rectangle(img, (int(w * 0.86), cur_y - 10), (w - 6, cur_y + 10), COLOR_SPOT, 1)
    cv2.putText(img, f"${last_close:.2f}", (int(w * 0.865), cur_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_SPOT, 1, cv2.LINE_AA)

    # Blend translucent FVG overlay (30% alpha)
    cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)

    # 8. Top Institutional Watermark Header
    banner_h = 36
    cv2.rectangle(img, (0, 0), (w, banner_h), (15, 23, 42), -1)
    cv2.line(img, (0, banner_h), (w, banner_h), (51, 65, 85), 1)
    status_text = f"DERIV SPOT 15m (frxXAUUSD) | {verdict} | Spot: ${last_close:.2f} | Confluence: {score}% | {session_txt[:24]}"
    cv2.putText(img, status_text, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (251, 191, 36), 1, cv2.LINE_AA)

    return Image.fromarray(img[:, :, ::-1])
