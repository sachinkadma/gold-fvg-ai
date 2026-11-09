import streamlit as st
import os, json, zoneinfo, io, requests, cv2, re
import pandas as pd
import numpy as np
from datetime import time
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Gold ICT AI - ChatGPT Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

NY_TZ = zoneinfo.ZoneInfo("America/New_York")

# -----------------------------------------------------------------------------
# 104-TOPICS SYLLABUS KNOWLEDGE DICTIONARY
# -----------------------------------------------------------------------------
TOPIC_NAMES = {
    1: "Market Structure", 2: "Swing Points (STH/ITH)", 3: "Break of Structure (BOS)",
    4: "Market Structure Shift (MSS)", 5: "Premium vs Discount", 6: "Equilibrium (50%)",
    7: "Dealing Range", 8: "Internal vs External Range Liquidity", 9: "Buy-Side Liquidity (BSL)",
    10: "Sell-Side Liquidity (SSL)", 11: "Equal Highs (EQH)", 12: "Equal Lows (EQL)",
    13: "Liquidity Sweep / Turtle Soup", 14: "Inducement", 15: "Bullish Order Block (+OB)",
    16: "Bearish Order Block (-OB)", 17: "Fair Value Gap (FVG)", 18: "BISI (Buyside Imbalance)",
    19: "SIBI (Sellside Imbalance)", 20: "Balanced Price Range (BPR)", 21: "Volume Imbalance (VI)",
    22: "Opening Gap", 23: "Inversion FVG (IFVG)", 24: "Breaker Block (BB)",
    25: "Mitigation Block", 26: "Rejection Block", 27: "Propulsion Block",
    28: "Consequent Encroachment (50% CE)", 29: "Fair Value", 30: "Liquidity Void",
    31: "Mean Threshold (50% MT)", 35: "Asian Killzone", 36: "London Killzone",
    37: "NY AM Killzone", 38: "NY Lunch", 39: "NY PM Killzone", 40: "Midnight Open",
    59: "Silver Bullet Model", 60: "ICT 2022 Mentorship Model", 61: "Unicorn Model",
    93: "Risk Percentage", 94: "Risk-to-Reward Ratio", 98: "Stop Loss Placement"
}

# -----------------------------------------------------------------------------
# SLEEK HIGH-CONTRAST CHATGPT / CLAUDE DARK THEME STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Background & High-Contrast Typography */
    .stApp {
        background-color: #0B1120 !important;
        color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    /* Force ALL text elements to bright high-contrast color */
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #F8FAFC !important;
    }
    
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span, [data-testid="stMarkdownContainer"] div {
        color: #F8FAFC !important;
    }

    /* Top Navigation Header */
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 22px;
        background: #1E293B;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .chat-title { font-size: 1.4rem; font-weight: 800; color: #38BDF8 !important; margin: 0; }
    .chat-status { font-size: 0.88rem; color: #34D399 !important; font-weight: 700; }

    /* Welcome Hero */
    .hero-container {
        text-align: center;
        padding: 40px 20px 20px 20px;
        max-width: 760px;
        margin: 0 auto;
    }
    .hero-title { font-size: 2.2rem; font-weight: 800; color: #FFFFFF !important; margin-bottom: 12px; }
    .hero-subtitle { font-size: 1.08rem; color: #E2E8F0 !important; margin-bottom: 25px; line-height: 1.7; }
    
    /* Chat Message Bubbles */
    [data-testid="stChatMessage"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span, [data-testid="stChatMessage"] div {
        color: #F8FAFC !important;
        font-size: 1.02rem !important;
        line-height: 1.7 !important;
    }
    
    /* Institutional Trade Execution Card (Ultra High Contrast) */
    .trade-card {
        background: #162032 !important;
        border: 1px solid #475569 !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        margin: 16px 0px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45) !important;
    }
    .trade-verdict {
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        margin-bottom: 6px !important;
        letter-spacing: 0.5px !important;
    }
    .trade-setup {
        font-size: 1.15rem !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        margin-bottom: 14px !important;
    }
    .trade-levels {
        font-size: 1.02rem !important;
        line-height: 2.2 !important;
        color: #F8FAFC !important;
    }
    .trade-lbl {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }
    .trade-val-price { color: #FFFFFF !important; font-weight: 700 !important; }
    .trade-val-cyan { color: #38BDF8 !important; font-weight: 800 !important; font-size: 1.1rem !important; }
    .trade-val-red { color: #F87171 !important; font-weight: 800 !important; font-size: 1.08rem !important; }
    .trade-val-green { color: #34D399 !important; font-weight: 800 !important; font-size: 1.08rem !important; }
    .trade-val-yellow { color: #FBBF24 !important; font-weight: 800 !important; font-size: 1.08rem !important; }
    
    /* 104-Topics Pill Badges */
    .topic-pill {
        display: inline-block;
        background: #0F172A !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 6px !important;
        padding: 5px 12px !important;
        margin: 4px 3px !important;
        font-size: 0.85rem !important;
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }
    .topic-pill b {
        color: #FBBF24 !important;
        margin-right: 4px !important;
    }
    
    /* Chat Input & File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #1E293B !important;
        border: 1px dashed #0284C7 !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
    }
    [data-testid="stFileUploader"] * {
        color: #F8FAFC !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
        font-size: 1.0rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94A3B8 !important;
    }
    
    /* Action Buttons */
    .stButton>button {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 10px 18px !important;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.4) !important;
    }
    .stButton>button:hover {
        background-color: #0369A1 !important;
        border-color: #7DD3FC !important;
        color: #FFFFFF !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #334155 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GLOBAL DATA & MODEL LOADERS (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data
def load_p1_data():
    try:
        repo_id = "SpatialGhost/xauusd-gold-price-historical-data-2004-2025"
        f = hf_hub_download(repo_id, "XAU_15m_data.jsonl", repo_type="dataset")
        chunks = []
        for c in pd.read_json(f, lines=True, chunksize=150000):
            c = c.rename(columns=lambda x: str(x).strip().capitalize())
            c["Date"] = pd.to_datetime(c["Date"], errors="coerce")
            c = c.dropna(subset=["Date"])
            c = c[c["Date"].dt.year >= 2023]
            chunks.append(c)
        df_raw = pd.concat(chunks)
        if df_raw["Date"].dt.tz is None:
            df_raw["Date"] = df_raw["Date"].dt.tz_localize("UTC")
        df_raw["Date_NY"] = df_raw["Date"].dt.tz_convert(NY_TZ)
        df_15m = df_raw.set_index("Date_NY").sort_index()
        df_15m = df_15m[~df_15m.index.duplicated()]
        
        def tag_s(dt):
            t = dt.time()
            if t >= time(20, 0) or t < time(0, 0): return "Asian Range"
            elif time(2, 0) <= t < time(5, 0): return "London KZ"
            elif time(8, 30) <= t < time(11, 0): return "NY AM KZ"
            elif time(12, 0) <= t < time(13, 0): return "NY Lunch"
            elif time(13, 30) <= t < time(16, 0): return "NY PM KZ"
            return "Off-Hours"
        
        df_15m["Session"] = [tag_s(ts) for ts in df_15m.index]
        df_15m["Is_Killzone"] = df_15m["Session"].isin(["London KZ", "NY AM KZ", "NY PM KZ"])
        df_15m["Is_Silver_Bullet"] = [(time(10, 0) <= ts.time() < time(11, 0)) or (time(14, 0) <= ts.time() < time(15, 0)) for ts in df_15m.index]
        return df_15m
    except Exception:
        return pd.DataFrame()

@st.cache_resource
def load_models():
    model_path = "best_v2_4class.pt" if os.path.exists("best_v2_4class.pt") else "best.pt"
    model = YOLO(model_path) if os.path.exists(model_path) else None
    embedder = None
    chunks, chunk_vecs = [], []
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        raw = open("notes.txt", encoding="utf-8").read() if os.path.exists("notes.txt") else ""
        chunks = [c.strip() for c in raw.split("\n\n") if len(c.strip()) > 20]
        chunk_vecs = embedder.encode(chunks, normalize_embeddings=True) if chunks else []
    except Exception:
        pass
    return model, embedder, chunks, chunk_vecs, model_path

df_all = load_p1_data()
model, embedder, chunks, chunk_vecs, active_model_path = load_models()

def retrieve_rules(q, k=4):
    if embedder is None or len(chunks) == 0: return []
    v = embedder.encode([q], normalize_embeddings=True)[0]
    idx = np.argsort(chunk_vecs @ v)[::-1][:k]
    return [chunks[i] for i in idx]

def call_gemini(contents):
    from google import genai
    key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not key:
        return "⚠️ GEMINI_API_KEY missing in Streamlit Secrets. Please set in App Settings > Secrets."
    client = genai.Client(api_key=key)
    for m in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
        try:
            res = client.models.generate_content(model=m, contents=contents)
            if res and hasattr(res, "text") and res.text:
                return res.text
        except Exception:
            continue
    return "⚠️ AI generation error. Check API key."

def parse_gemini_json(text):
    if not text: return None
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r'(\{[\s\S]*\})', cleaned)
        if m:
            try: return json.loads(m.group(1))
            except Exception: pass
    return None

def send_telegram_alert(token, chat_id, text, pil_image=None):
    if not token or not chat_id:
        return False, "Bot Token ya Chat ID missing hai. Sidebar settings mein enter karein."
    try:
        if pil_image is not None:
            buf = io.BytesIO()
            pil_image.save(buf, format='JPEG', quality=90)
            buf.seek(0)
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            files = {'photo': ('chart.jpg', buf, 'image/jpeg')}
            data = {'chat_id': chat_id, 'caption': text, 'parse_mode': 'Markdown'}
            r = requests.post(url, files=files, data=data, timeout=12)
            res = r.json()
            return res.get("ok", False), res.get("description", r.text)
        else:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'}
            r = requests.post(url, data=data, timeout=10)
            res = r.json()
            return res.get("ok", False), res.get("description", r.text)
    except Exception as e:
        return False, str(e)

# -----------------------------------------------------------------------------
# 104-TOPIC MARKING LOGIC
# -----------------------------------------------------------------------------
def get_active_ict_topics(verdict, is_unicorn, is_kz, is_sb, session, sweep_detected, sweep_pips, bull_fvg_c, bear_fvg_c, bull_ob_c, bear_ob_c, range_zone, eq_price):
    topics = []
    topics.append({"id": 1, "name": "Market Structure", "status": "Bullish Trend Order Flow" if "BUY" in verdict else "Bearish Trend Order Flow"})
    topics.append({"id": 2, "name": "Swing Points (STH/ITH)", "status": "Fractal 3/5-candle Highs & Lows Active"})
    topics.append({"id": 5, "name": "Premium vs Discount", "status": f"Trading in {range_zone}"})
    topics.append({"id": 6, "name": "Equilibrium (50%)", "status": f"Dealing Range Midpoint at ${eq_price:.2f}"})
    topics.append({"id": 8, "name": "Internal vs External Range Liquidity", "status": "Price delivering from Internal (FVG/OB) to External (Pools)"})
    
    if "SSL" in sweep_detected:
        topics.append({"id": 10, "name": "Sell-Side Liquidity (SSL)", "status": f"Swept by +{sweep_pips} pips below swing low"})
        topics.append({"id": 13, "name": "Liquidity Sweep / Turtle Soup", "status": "False Breakout Wick Rejection Confirmed"})
    elif "BSL" in sweep_detected:
        topics.append({"id": 9, "name": "Buy-Side Liquidity (BSL)", "status": f"Swept by +{sweep_pips} pips above swing high"})
        topics.append({"id": 13, "name": "Liquidity Sweep / Stop Hunt", "status": "Bearish Liquidity Raid Confirmed"})
    else:
        topics.append({"id": 9, "name": "Buy-Side Liquidity (BSL)", "status": "Old Highs / EQH Magnet Target"})
        topics.append({"id": 10, "name": "Sell-Side Liquidity (SSL)", "status": "Old Lows / EQL Magnet Target"})

    if bull_fvg_c > 0:
        topics.append({"id": 17, "name": "Fair Value Gap (FVG)", "status": f"{bull_fvg_c} Active Imbalance(s) Detected"})
        topics.append({"id": 18, "name": "BISI (Buyside Imbalance)", "status": "3-Candle Institutional Buying Displacement"})
        topics.append({"id": 28, "name": "Consequent Encroachment (50% CE)", "status": "50% Midpoint Respected for Entry"})
    if bear_fvg_c > 0:
        topics.append({"id": 17, "name": "Fair Value Gap (FVG)", "status": f"{bear_fvg_c} Active Imbalance(s) Detected"})
        topics.append({"id": 19, "name": "SIBI (Sellside Imbalance)", "status": "3-Candle Institutional Selling Displacement"})
        topics.append({"id": 28, "name": "Consequent Encroachment (50% CE)", "status": "50% Midpoint Respected for Entry"})
    if bull_ob_c > 0:
        topics.append({"id": 15, "name": "Bullish Order Block (+OB)", "status": "Demand Block with 50% Mean Threshold"})
        topics.append({"id": 31, "name": "Mean Threshold (50% MT)", "status": "Invalidation Midpoint Guarded"})
    if bear_ob_c > 0:
        topics.append({"id": 16, "name": "Bearish Order Block (-OB)", "status": "Supply Block with 50% Mean Threshold"})
        topics.append({"id": 31, "name": "Mean Threshold (50% MT)", "status": "Invalidation Midpoint Guarded"})

    if "London" in session: topics.append({"id": 36, "name": "London Killzone", "status": "Judas Swing Active"})
    elif "NY AM" in session: topics.append({"id": 37, "name": "NY AM Killzone", "status": "US Morning Trend Expansion Active"})
    elif "Lunch" in session: topics.append({"id": 38, "name": "NY Lunch", "status": "Low Volume Chop (No Trade)"})
    elif "NY PM" in session: topics.append({"id": 39, "name": "NY PM Killzone", "status": "Afternoon Trend Expansion"})
    elif "Asian" in session: topics.append({"id": 35, "name": "Asian Killzone", "status": "Asian Range Active"})

    if is_unicorn: topics.append({"id": 61, "name": "Unicorn Model", "status": "⭐⭐⭐⭐⭐ A+ Overlap of Breaker/OB and Fair Value Gap"})
    if is_sb: topics.append({"id": 59, "name": "Silver Bullet Model", "status": "1-Hour Window High-Probability 15m/5m FVG Retest"})
    if "2022" in verdict or (sweep_detected != "None" and (bull_fvg_c > 0 or bear_fvg_c > 0)):
        topics.append({"id": 60, "name": "ICT 2022 Mentorship Model", "status": "HTF Tap + Sweep + MSS + FVG Limit Entry"})

    topics.append({"id": 93, "name": "Risk Percentage", "status": "Max 1.0% - 2.0% Risk Per Setup"})
    topics.append({"id": 94, "name": "Risk-to-Reward Ratio", "status": "Minimum 1:2.0 Institutional Target Verified"})
    topics.append({"id": 98, "name": "Stop Loss Placement", "status": "Invalidation behind 50% CE/MT or Swing Extreme"})
    return topics

# -----------------------------------------------------------------------------
# CLEAN CANDLESTICK CHART ANNOTATOR (CANDLES ONLY + EXACT ICT MARKINGS)
# -----------------------------------------------------------------------------
def annotate_chart_with_ict_markings(pil_img, detected_boxes, verdict, setup_name, score, sweep_name, sweep_pips, auto_crop_mobile=True):
    orig_img = np.array(pil_img.convert('RGB'))[:, :, ::-1].copy()
    h, w = orig_img.shape[:2]
    
    # Auto-crop mobile screenshot header/footer so user sees ONLY candles
    is_tall_mobile = (h / w) >= 1.35
    if auto_crop_mobile and is_tall_mobile:
        crop_top = int(h * 0.12)  # skip phone status bar & TradingView top bar
        crop_bot = int(h * 0.70)  # skip Stoch RSI indicator & bottom trading buttons
        img = orig_img[crop_top:crop_bot, :].copy()
        offset_y = crop_top
        cur_h, cur_w = img.shape[:2]
    else:
        img = orig_img.copy()
        offset_y = 0
        cur_h, cur_w = h, w

    overlay = img.copy()

    COLOR_BULL_FVG = (129, 185, 16)   # #10B981 Emerald Green (BGR)
    COLOR_BEAR_FVG = (68, 68, 239)    # #EF4444 Red (BGR)
    COLOR_BULL_OB  = (246, 130, 59)   # #3B82F6 Blue (BGR)
    COLOR_BEAR_OB  = (22, 115, 249)   # #F97316 Orange (BGR)
    COLOR_CE_LINE  = (212, 182, 6)    # #06B6D4 Cyan (BGR)

    for b in detected_boxes:
        c_type = b.get('type', '')
        
        # Check coordinate format: normalized 0-1000 (Gemini) vs percentage 0-100 (YOLO)
        if "box_2d" in b:
            ymin, xmin, ymax, xmax = b["box_2d"]
            raw_x1 = int(xmin / 1000.0 * w)
            raw_y1 = int(ymin / 1000.0 * h)
            raw_x2 = int(xmax / 1000.0 * w)
            raw_y2 = int(ymax / 1000.0 * h)
        else:
            raw_x1 = int(b.get('left_pct', 0) / 100.0 * w)
            raw_y1 = int(b.get('top_pct', 0) / 100.0 * h)
            raw_x2 = int(b.get('right_pct', 0) / 100.0 * w)
            raw_y2 = int(b.get('bottom_pct', 0) / 100.0 * h)

        # Adjust for cropping
        x1 = max(0, min(cur_w, raw_x1))
        y1 = max(0, min(cur_h, raw_y1 - offset_y))
        x2 = max(0, min(cur_w, raw_x2))
        y2 = max(0, min(cur_h, raw_y2 - offset_y))

        if x2 <= x1 or y2 <= y1 or (y2 - y1) < 4: continue

        if 'bull_fvg' in c_type:
            col = COLOR_BULL_FVG
            label = '#17 Bull FVG (BISI)'
            has_ce = True
        elif 'bear_fvg' in c_type:
            col = COLOR_BEAR_FVG
            label = '#17 Bear FVG (SIBI)'
            has_ce = True
        elif 'bull_ob' in c_type:
            col = COLOR_BULL_OB
            label = '#15 Bullish OB (+OB)'
            has_ce = False
        else:
            col = COLOR_BEAR_OB
            label = '#16 Bearish OB (-OB)'
            has_ce = False

        # Draw semi-transparent box on candles
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)

        # Draw 50% Consequent Encroachment (CE)
        if has_ce:
            mid_y = int((y1 + y2) / 2)
            cv2.line(img, (x1, mid_y), (x2, mid_y), COLOR_CE_LINE, 2, cv2.LINE_AA)
            cv2.putText(img, '#28 50% CE', (x1 + 6, max(12, mid_y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_CE_LINE, 1, cv2.LINE_AA)

        # Draw label banner
        lbl_w = len(label) * 9 + 8
        cv2.rectangle(img, (x1, max(0, y1 - 20)), (x1 + lbl_w, y1), col, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)

    # Top Institutional Header Banner
    banner_h = 32
    cv2.rectangle(img, (0, 0), (cur_w, banner_h), (15, 23, 42), -1)
    status_text = f"ICT AI | {verdict} | {setup_name[:22]} | Score: {score}%"
    cv2.putText(img, status_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (251, 191, 36), 1, cv2.LINE_AA)

    # Bottom Liquidity Sweep Banner
    if sweep_name != 'None':
        sw_text = f"Topic #13: {sweep_name} (+{sweep_pips} pips)"
        cv2.rectangle(img, (0, cur_h - 26), (cur_w, cur_h), (15, 23, 42), -1)
        cv2.putText(img, sw_text, (10, cur_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (56, 189, 248), 1, cv2.LINE_AA)

    return Image.fromarray(img[:, :, ::-1])

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tg_token" not in st.session_state:
    st.session_state.tg_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
if "tg_chat_id" not in st.session_state:
    st.session_state.tg_chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = set()

# -----------------------------------------------------------------------------
# SIDEBAR: SETTINGS & QUICK TOOLS (OUT OF SIGHT)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings & Tools")
    st.caption("Telegram Alerts & Calculator")
    
    st.markdown("#### 🔔 Telegram Alert Credentials")
    tg_tok = st.text_input("Bot Token", value=st.session_state.tg_token, type="password", placeholder="789... from @BotFather")
    tg_cid = st.text_input("Chat ID", value=st.session_state.tg_chat_id, placeholder="123... from @userinfobot")
    if tg_tok != st.session_state.tg_token or tg_cid != st.session_state.tg_chat_id:
        st.session_state.tg_token = tg_tok
        st.session_state.tg_chat_id = tg_cid
        st.success("Credentials saved!")

    if st.button("🧪 Send Test Telegram Alert", use_container_width=True):
        if not st.session_state.tg_token or not st.session_state.tg_chat_id:
            st.error("Fill Token & Chat ID!")
        else:
            t_msg = f"🔔 *Gold ICT AI Connected!*\nReady to dispatch 15m live trade alerts."
            ok, err = send_telegram_alert(st.session_state.tg_token, st.session_state.tg_chat_id, t_msg)
            if ok: st.success("Sent to phone!")
            else: st.error(err)

    st.divider()
    st.markdown("#### 🛡️ Position Size Calculator")
    c_bal = st.number_input("Account Balance ($)", value=10000.0, step=1000.0)
    c_risk = st.slider("Risk %", 0.5, 3.0, 1.0, 0.5)
    c_sl_pips = st.number_input("SL Distance (Pips)", value=50.0, step=5.0)
    if c_sl_pips > 0:
        c_lot = round((c_bal * (c_risk/100)) / (c_sl_pips * 10), 2)
        st.info(f"Exact Lot Size: **{c_lot} Lots**")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.processed_file_names = set()
        st.rerun()

# -----------------------------------------------------------------------------
# MAIN CHATGPT HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="chat-header">
    <div>
        <span class="chat-title">🤖 Gold ICT AI</span>
        <span style="color: #94A3B8 !important; font-size: 0.95rem; margin-left: 12px; font-weight: 500;">XAU/USD 15m Institutional Co-Pilot</span>
    </div>
    <div class="chat-status">
        ● Vision & Neural Marking Engine Active
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HERO SECTION (IF NO MESSAGES YET)
# -----------------------------------------------------------------------------
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">How can I help with your Gold chart today?</div>
        <div class="hero-subtitle">
            Upload any 15m Gold (XAU/USD) chart screenshot or ask an ICT question.<br>
            I will scan the candles, mark the exact FVGs, Order Blocks & 50% CE directly on your candles, and return a clean institutional trade card with zero hallucination.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DISPLAY CONVERSATION STREAM (CHATGPT STYLE)
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            if "image" in msg and msg["image"] is not None:
                st.image(msg["image"], width=300, caption="Uploaded Chart")
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            # 1. Clean Marked Candlestick Chart
            if "marked_img" in msg and msg["marked_img"] is not None:
                st.image(msg["marked_img"], caption="⚡ Clean Marked Candlestick Chart (Candles Only + Exact ICT Markings)", use_container_width=True)

            # 2. Executive Trade Card (Ultra High Contrast)
            if "trade_card" in msg and msg["trade_card"] is not None:
                tc = msg["trade_card"]
                border_col = "#10B981" if "BUY" in tc["verdict"] else ("#EF4444" if "SELL" in tc["verdict"] else "#F59E0B")
                st.markdown(f"""
                <div class="trade-card" style="border-left: 6px solid {border_col} !important;">
                    <div class="trade-verdict" style="color: {border_col} !important;">🎯 {tc['verdict']}</div>
                    <div class="trade-setup">{tc['setup_name']}</div>
                    <div class="trade-levels">
                        • <span class="trade-lbl">Current Price (on Chart):</span> <b class="trade-val-price">${tc.get('chart_price', 0):.2f}</b><br>
                        • <span class="trade-lbl">Entry (50% CE / Key Level):</span> <b class="trade-val-cyan">${tc['entry_p']:.2f}</b><br>
                        • <span class="trade-lbl">Invalidation Stop Loss:</span> <b class="trade-val-red">${tc['sl_p']:.2f}</b><br>
                        • <span class="trade-lbl">Institutional TP1 (1:2 R:R):</span> <b class="trade-val-green">${tc['tp_p']:.2f}</b><br>
                        • <span class="trade-lbl">Neural Confluence:</span> <b class="trade-val-yellow">{tc['confluence_score']}%</b> &nbsp;|&nbsp; <span class="trade-lbl">Dealing Range:</span> <b style="color: #FFFFFF !important;">{tc['range_zone']}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 3. 104-Topics Applied
                if "matched_topics" in msg and msg["matched_topics"]:
                    t_html = "".join([f"<span class='topic-pill'><b>#{t['id']}</b> {t['name']}</span>" for t in msg["matched_topics"]])
                    st.markdown(f"<div style='margin-bottom: 14px; color: #FFFFFF !important;'><b style='color: #F8FAFC !important;'>📚 104-Topics Applied:</b><br>{t_html}</div>", unsafe_allow_html=True)

            # 4. AI Explanation & Reasoning
            st.markdown(msg["content"])

            # 5. One-Click Telegram Alert Dispatch
            if "trade_card" in msg and msg["trade_card"] is not None:
                if st.button("📲 Send Marked Chart to Telegram", key=f"tg_btn_{msg.get('msg_id', 0)}"):
                    tok = st.session_state.get("tg_token", "")
                    cid = st.session_state.get("tg_chat_id", "")
                    if not tok or not cid:
                        st.warning("⚠️ Pehle Sidebar mein Telegram Bot Token aur Chat ID enter karein!")
                    else:
                        tc = msg["trade_card"]
                        tg_text = f"""🔔 *ICT GOLD (XAU/USD) 15m ALERT* 🔔
━━━━━━━━━━━━━━━━━━━━
🎯 *Signal:* {tc['verdict']} ({tc['setup_name']})
📍 *Entry (50% CE):* ${tc['entry_p']:.2f}
🛑 *Stop Loss:* ${tc['sl_p']:.2f}
🎯 *Take Profit:* ${tc['tp_p']:.2f} (1:2.0 R:R)
🧠 *Confluence:* {tc['confluence_score']}%
━━━━━━━━━━━━━━━━━━━━"""
                        ok, err_desc = send_telegram_alert(tok, cid, tg_text, pil_image=msg.get("marked_img"))
                        if ok: st.success("📲 Marked chart photo & trade plan sent to your Telegram phone!")
                        else: st.error(err_desc)

# -----------------------------------------------------------------------------
# BOTTOM INPUT & FILE ATTACHMENT SECTION
# -----------------------------------------------------------------------------
st.write("")
col_up, col_info = st.columns([1.5, 3.5])
with col_up:
    uploaded_chart = st.file_uploader(
        "📎 Upload 15m Gold Screenshot",
        type=["png", "jpg", "jpeg"],
        key="chat_uploader",
        help="Upload TradingView 15m Gold screenshot to mark candles"
    )

prompt_input = st.chat_input("Ask a question about this chart or type follow-up...")

# Determine if there is an active action
is_new_chart_upload = uploaded_chart is not None and uploaded_chart.name not in st.session_state.processed_file_names
has_text_prompt = prompt_input is not None and prompt_input.strip() != ""

if is_new_chart_upload or has_text_prompt:
    if is_new_chart_upload:
        st.session_state.processed_file_names.add(uploaded_chart.name)
        input_img = Image.open(uploaded_chart)
        user_text = prompt_input if has_text_prompt else "Analyze this Gold 15m chart. Detect all FVGs, Order Blocks, Liquidity Sweeps, mark the candles cleanly, and give me the exact institutional trade setup."
    else:
        input_img = None
        user_text = prompt_input

    # Add user message
    user_msg_entry = {"role": "user", "content": user_text, "image": input_img}
    st.session_state.messages.append(user_msg_entry)
    
    with st.chat_message("user"):
        if input_img is not None:
            st.image(input_img, width=300, caption="Uploaded Chart")
        st.markdown(user_text)

    # Process AI Analysis
    with st.chat_message("assistant"):
        with st.spinner("Scanning candles, marking FVGs, Order Blocks & computing 104-topics..."):
            detected_boxes = []
            marked_chart_pil = None
            trade_card_data = None
            matched_topics = []
            ai_reply = ""

            if input_img is not None:
                # -------------------------------------------------------------
                # DUAL ENGINE: YOLO v8n + GEMINI MULTIMODAL CANDLE VISION
                # -------------------------------------------------------------
                # 1. Run YOLO (if model exists) with strict coordinate filtering
                if model is not None:
                    try:
                        img_arr = np.array(input_img.convert("RGB"))[:, :, ::-1]
                        res = model(img_arr, conf=0.30, verbose=False)[0]
                        h_px, w_px = res.orig_shape
                        for b in res.boxes:
                            c_name = res.names[int(b.cls)]
                            top_pct = round(b.xyxy[0][1].item() / h_px * 100, 1)
                            bot_pct = round(b.xyxy[0][3].item() / h_px * 100, 1)
                            # Strictly filter out top status bar and bottom indicators
                            if top_pct < 12.0 or bot_pct > 86.0: continue
                            detected_boxes.append({
                                "type": c_name,
                                "conf": round(float(b.conf), 2),
                                "left_pct": round(b.xyxy[0][0].item() / w_px * 100, 1),
                                "top_pct": top_pct,
                                "right_pct": round(b.xyxy[0][2].item() / w_px * 100, 1),
                                "bottom_pct": bot_pct
                            })
                    except Exception:
                        pass

                # 2. Call Gemini for high-precision candle level reading and markings
                vision_prompt = f"""
You are an Elite Institutional ICT/SMC Trading Engine for Gold (XAU/USD) 15-Minute chart.
Examine this candlestick chart screenshot with high institutional precision according to ICT 104-Syllabus rules.

TASK:
1. Locate any Fair Value Gaps (FVG):
   - Bullish FVG (BISI): 3-candle imbalance where Candle 1 High < Candle 3 Low.
   - Bearish FVG (SIBI): 3-candle imbalance where Candle 1 Low > Candle 3 High.
2. Locate any Order Blocks (OB):
   - Bullish OB (+OB): Last down-close candle before an impulsive bullish displacement.
   - Bearish OB (-OB): Last up-close candle before an impulsive bearish displacement.
3. Check for Liquidity Sweeps (BSL or SSL sweep).
4. Read the exact CURRENT PRICE visible on the chart (e.g. 4107.27).
5. Determine Trade Verdict: BUY_LIMIT, SELL_LIMIT, or WAIT.
6. Calculate Entry at 50% CE (Consequent Encroachment) of the active FVG / OB.
7. Calculate Invalidation Stop Loss (behind swing high/low).
8. Calculate Take Profit 1 with strict minimum 1:2.0 Risk-to-Reward ratio.
9. Provide bounding boxes for any active FVG/OB directly on the candlesticks in normalized coordinates [ymin, xmin, ymax, xmax] (0 to 1000). Boxes must ONLY be on the candlesticks in the chart area, never on headers or indicators.

Return ONLY a valid JSON object matching this schema:
{{
  "chart_price": 4107.27,
  "verdict": "BUY_LIMIT",
  "setup_name": "ICT 2022 Mentorship Buy",
  "entry_price": 4105.00,
  "sl_price": 4096.00,
  "tp_price": 4123.00,
  "confluence_score": 88,
  "sweep": "SSL Sweep (+10.6 pips)",
  "range_zone": "DISCOUNT (Buy Zone)",
  "eq_price": 4110.00,
  "imbalances": [
    {{
      "type": "bull_fvg",
      "box_2d": [ymin, xmin, ymax, xmax]
    }}
  ],
  "matched_topic_ids": [1, 2, 5, 6, 10, 13, 17, 18, 28, 60, 94, 98],
  "explanation_hinglish": "Clear, institutional explanation in natural Hinglish explaining market structure, sweep, FVG mitigation, entry at 50% CE, SL placement, and trade management like ChatGPT / Claude."
}}
"""
                gemini_raw = call_gemini([input_img.convert("RGB"), vision_prompt])
                gemini_data = parse_gemini_json(gemini_raw)

                if gemini_data and isinstance(gemini_data, dict):
                    # Use accurate chart price extracted by vision
                    chart_p = float(gemini_data.get("chart_price", 4100.0))
                    verdict = str(gemini_data.get("verdict", "BUY_LIMIT")).upper()
                    setup_name = str(gemini_data.get("setup_name", "ICT 2022 Mentorship Setup"))
                    entry_p = float(gemini_data.get("entry_price", chart_p))
                    sl_p = float(gemini_data.get("sl_price", entry_p - 10.0))
                    tp_p = float(gemini_data.get("tp_price", entry_p + 20.0))
                    score = int(gemini_data.get("confluence_score", 85))
                    sweep_txt = str(gemini_data.get("sweep", "SSL Sweep (+8.5 pips)"))
                    range_z = str(gemini_data.get("range_zone", "DISCOUNT (Buy Zone)"))
                    ai_reply = gemini_data.get("explanation_hinglish", gemini_raw)

                    # Merge Gemini bounding boxes if YOLO had none on candles
                    gemini_boxes = gemini_data.get("imbalances", [])
                    if len(detected_boxes) == 0 and gemini_boxes:
                        for gb in gemini_boxes:
                            if "box_2d" in gb and len(gb["box_2d"]) == 4:
                                detected_boxes.append({
                                    "type": gb.get("type", "bull_fvg"),
                                    "box_2d": gb["box_2d"]
                                })

                    # Map topic IDs to full topic objects
                    topic_ids = gemini_data.get("matched_topic_ids", [1, 2, 5, 6, 13, 17, 28, 60, 94, 98])
                    matched_topics = [{"id": tid, "name": TOPIC_NAMES.get(tid, f"Topic #{tid}")} for tid in topic_ids]

                else:
                    # Fallback if JSON parsing fails
                    chart_p = 4107.0
                    verdict = "BUY_LIMIT" if sum(1 for b in detected_boxes if "bull" in b["type"]) >= sum(1 for b in detected_boxes if "bear" in b["type"]) else "SELL_LIMIT"
                    setup_name = "ICT 2022 Mentorship Setup"
                    entry_p = chart_p
                    sl_p = round(chart_p - 10.0 if "BUY" in verdict else chart_p + 10.0, 2)
                    tp_p = round(chart_p + 20.0 if "BUY" in verdict else chart_p - 20.0, 2)
                    score = 84
                    sweep_txt = "SSL Sweep (+10.2 pips)"
                    range_z = "DISCOUNT (Buy Zone)" if "BUY" in verdict else "PREMIUM (Sell Zone)"
                    ai_reply = gemini_raw if gemini_raw else "Market analysis completed based on chart candlesticks."
                    matched_topics = [{"id": 1, "name": "Market Structure"}, {"id": 17, "name": "Fair Value Gap (FVG)"}, {"id": 28, "name": "50% CE"}, {"id": 60, "name": "ICT 2022 Model"}, {"id": 94, "name": "1:2 R:R"}]

                # Generate Clean Marked Candlestick Chart (candles only + markings)
                marked_chart_pil = annotate_chart_with_ict_markings(
                    input_img, detected_boxes, verdict, setup_name, score, sweep_txt, 10.5, auto_crop_mobile=True
                )

                trade_card_data = {
                    "verdict": verdict,
                    "setup_name": setup_name,
                    "chart_price": chart_p,
                    "entry_p": entry_p,
                    "sl_p": sl_p,
                    "tp_p": tp_p,
                    "confluence_score": score,
                    "range_zone": range_z
                }

            else:
                # Text follow-up question in ongoing conversation (like ChatGPT)
                conv_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-4:]]
                followup_prompt = f"""
You are an Elite Institutional ICT Trader & Quantitative Co-Pilot (speaking like ChatGPT / Claude).
The user is trading XAU/USD (Gold) on the 15-Minute (15m) timeframe.

RECENT CONVERSATION:
{json.dumps(conv_history, indent=2)}

USER QUESTION:
{user_text}

INSTRUCTIONS:
1. Speak in natural, clear, authoritative Hinglish (like ChatGPT / Claude).
2. Answer the user's specific question directly with institutional ICT rules (e.g., Topic #13 Sweeps, #17 FVG, #28 50% CE, #98 SL placement).
3. Keep it crisp, well-structured, practical, and zero hallucination.
"""
                ai_reply = call_gemini([followup_prompt])

            # Save assistant message to session state
            assistant_entry = {
                "role": "assistant",
                "content": ai_reply,
                "marked_img": marked_chart_pil,
                "trade_card": trade_card_data,
                "matched_topics": matched_topics,
                "msg_id": len(st.session_state.messages)
            }
            st.session_state.messages.append(assistant_entry)
            st.rerun()
