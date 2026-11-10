import streamlit as st
import os, json, zoneinfo, io, requests, cv2, re, base64
import pandas as pd
import numpy as np
from datetime import datetime, time
from PIL import Image
from deriv_client import (
    fetch_live_gold_candles,
    compute_deterministic_market_structure,
    render_native_cameron4x_chart
)
from knowledge_store import ICTKnowledgeStore

st.set_page_config(
    page_title="Gold ICT AGI - Institutional Co-Pilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

NY_TZ = zoneinfo.ZoneInfo("America/New_York")

# -----------------------------------------------------------------------------
# KNOWLEDGE STORE & TEMPORAL KILLZONE RETRIEVAL
# -----------------------------------------------------------------------------
@st.cache_resource
def get_knowledge_store():
    return ICTKnowledgeStore(knowledge_dir="knowledge")

knowledge_store = get_knowledge_store()
TOPIC_NAMES = knowledge_store.get_all_topics_summary()

# -----------------------------------------------------------------------------
# LIVE NY TIME & INTERBANK SESSION DETECTOR
# -----------------------------------------------------------------------------
def get_live_ny_session():
    now_ny = datetime.now(NY_TZ)
    t = now_ny.time()
    time_str = now_ny.strftime("%H:%M:%S EST")
    if time(2, 0) <= t < time(5, 0):
        return f"🟢 London Killzone ({time_str})", "#10B981"
    elif time(8, 30) <= t < time(11, 0):
        if time(10, 0) <= t < time(11, 0):
            return f"⚡ NY AM Killzone + Silver Bullet Active! ({time_str})", "#F59E0B"
        return f"🟢 NY AM Killzone - Peak Gold Volatility ({time_str})", "#10B981"
    elif time(12, 0) <= t < time(13, 0):
        return f"⚠️ NY Lunch - Low Volume Chop ({time_str})", "#EF4444"
    elif time(13, 30) <= t < time(16, 0):
        if time(14, 0) <= t < time(15, 0):
            return f"⚡ NY PM Killzone + Silver Bullet Active! ({time_str})", "#F59E0B"
        return f"🟢 NY PM Killzone ({time_str})", "#10B981"
    elif time(20, 0) <= t or t < time(0, 0):
        return f"🔵 Asian Range - Daily Accumulation ({time_str})", "#38BDF8"
    else:
        return f"⚪ Off-Hours - Low Liquidity ({time_str})", "#94A3B8"

# -----------------------------------------------------------------------------
# SLEEK HIGH-CONTRAST CHATGPT / CLAUDE DARK THEME STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0B1120 !important;
        color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #F8FAFC !important;
    }
    
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span, [data-testid="stMarkdownContainer"] div {
        color: #F8FAFC !important;
    }

    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 22px;
        background: #1E293B;
        border-radius: 12px;
        margin-bottom: 16px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .chat-title { font-size: 1.35rem; font-weight: 800; color: #38BDF8 !important; margin: 0; }
    .chat-badge {
        background: linear-gradient(135deg, #0284C7, #2563EB);
        color: #FFFFFF !important;
        font-size: 0.72rem;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 800;
        margin-left: 8px;
        vertical-align: middle;
        letter-spacing: 0.5px;
    }
    .chat-ticker {
        display: inline-block;
        background: #0F172A;
        border: 1px solid #F59E0B;
        color: #FBBF24 !important;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        margin-left: 10px;
    }
    .chat-status { font-size: 0.88rem; font-weight: 700; }

    .hero-container {
        text-align: center;
        padding: 24px 20px 14px 20px;
        max-width: 820px;
        margin: 0 auto;
    }
    .hero-title { font-size: 2.1rem; font-weight: 800; color: #FFFFFF !important; margin-bottom: 10px; }
    .hero-subtitle { font-size: 1.05rem; color: #E2E8F0 !important; margin-bottom: 20px; line-height: 1.7; }
    
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
    .trade-grade-badge {
        display: inline-block;
        background: #0284C7;
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 0.85rem;
        padding: 4px 10px;
        border-radius: 6px;
        margin-left: 10px;
    }
    
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

    button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7, #059669) !important;
        border: 1px solid #34D399 !important;
        font-size: 1.02rem !important;
        box-shadow: 0 4px 14px rgba(5, 150, 105, 0.4) !important;
    }

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
# DYNAMIC FRONTIER GEMINI REST ENGINE
# -----------------------------------------------------------------------------
def get_available_gemini_models(api_key):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        r = requests.get(url, timeout=7)
        if r.status_code == 200:
            data = r.json()
            models = []
            for m in data.get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    m_name = m.get("name", "").replace("models/", "")
                    models.append(m_name)
            return models, None
        else:
            err_data = r.json().get("error", {})
            return [], err_data.get("message", r.text)
    except Exception as e:
        return [], str(e)

def call_gemini(contents, preferred_model_type="AGI Deep Thinking"):
    key = st.session_state.get("gemini_key", "").strip() or st.secrets.get("GEMINI_API_KEY", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None, None, "⚠️ GEMINI_API_KEY missing. Please enter your API key in the Sidebar Settings ⚙️."
    
    available_models, disc_err = get_available_gemini_models(key)
    
    if available_models:
        def rank_model(m):
            m_l = m.lower()
            if "AGI" in preferred_model_type:
                if "thinking" in m_l: return 0
                if "2.5-flash" in m_l or "2.5" in m_l: return 1
                if "2.0-flash" in m_l: return 2
                if "1.5-pro" in m_l: return 3
                if "1.5-flash" in m_l: return 4
            elif "Speed" in preferred_model_type:
                if "2.5-flash" in m_l or "2.0-flash" in m_l: return 0
                if "1.5-flash" in m_l: return 1
                if "thinking" in m_l: return 2
            else:
                if "1.5-pro" in m_l: return 0
                if "2.0-flash" in m_l: return 1
                if "thinking" in m_l: return 2
            return 10
        candidates = sorted(available_models, key=rank_model)
    else:
        candidates = [
            "gemini-2.5-flash",
            "gemini-2.0-flash-thinking-exp-01-21",
            "gemini-2.0-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash"
        ]

    parts = []
    for item in contents:
        if isinstance(item, str):
            parts.append({"text": item})
        elif isinstance(item, Image.Image):
            buf = io.BytesIO()
            item.convert("RGB").save(buf, format="JPEG", quality=85)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64_str
                }
            })

    payload = {"contents": [{"parts": parts}]}
    headers = {"Content-Type": "application/json"}

    last_err = disc_err or ""
    for mod in candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={key}"
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=28)
            if r.status_code == 200:
                res_data = r.json()
                cands = res_data.get("candidates", [])
                if cands:
                    p = cands[0].get("content", {}).get("parts", [])
                    if p:
                        st.session_state.active_gemini_model = mod
                        return p[0].get("text", ""), mod, None
            else:
                err_info = r.json().get("error", {})
                last_err = err_info.get("message", r.text)
        except Exception as e:
            last_err = str(e)
            continue

    return None, None, f"⚠️ Gemini API Error: {last_err}. Check your key in Sidebar Settings ⚙️."

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
            files = {'photo': ('gold_cameron4x_chart.jpg', buf, 'image/jpeg')}
            data = {'chat_id': chat_id, 'caption': text, 'parse_mode': 'Markdown'}
            r = requests.post(url, files=files, data=data, timeout=14)
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
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = st.secrets.get("GEMINI_API_KEY", "")
if "agi_model_mode" not in st.session_state:
    st.session_state.agi_model_mode = "AGI Deep Thinking"
if "active_gemini_model" not in st.session_state:
    st.session_state.active_gemini_model = "Auto-Detect"
if "account_balance" not in st.session_state:
    st.session_state.account_balance = 10000.0
if "risk_percent" not in st.session_state:
    st.session_state.risk_percent = 1.0
if "tg_token" not in st.session_state:
    st.session_state.tg_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
if "tg_chat_id" not in st.session_state:
    st.session_state.tg_chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = set()
if "last_spot_price" not in st.session_state:
    st.session_state.last_spot_price = None
if "last_spot_source" not in st.session_state:
    st.session_state.last_spot_source = "Deriv Gateway"
if "show_ewt" not in st.session_state:
    st.session_state.show_ewt = True

# -----------------------------------------------------------------------------
# SIDEBAR: SETTINGS, RISK, & ELLIOTT WAVE CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 AGI Institutional Control Hub")
    
    st.markdown("#### 🚀 Frontier AGI Model Engine")
    selected_mode = st.radio(
        "Reasoning Architecture",
        ["🧠 AGI 4-Agent Swarm (Deep CoT + EWT)", "⚡ Ultra-Fast Vision (Instant Scan)", "🔬 Institutional Quant Pro (Full 108 Audit)"],
        index=0
    )
    st.session_state.agi_model_mode = selected_mode

    st.markdown("#### 🔑 Google Gemini API Key")
    g_key_val = st.text_input("Gemini API Key", value=st.session_state.gemini_key, type="password", placeholder="AIzaSy... from Google AI Studio")
    if g_key_val != st.session_state.gemini_key:
        st.session_state.gemini_key = g_key_val.strip()
        st.success("API Key saved!")

    if st.button("🧪 Test AGI Connection & Models", use_container_width=True):
        if not st.session_state.gemini_key:
            st.error("Pehle API Key enter karein!")
        else:
            with st.spinner("Discovering authorized Gemini models..."):
                mods, err = get_available_gemini_models(st.session_state.gemini_key)
                if mods:
                    test_txt, active_m, test_err = call_gemini(["Respond with: 'AGI Swarm Intelligence Online & Ready'"], preferred_model_type=st.session_state.agi_model_mode)
                    if test_txt:
                        st.success(f"✅ Connected! Active: **{active_m}** ({len(mods)} models available)")
                    else:
                        st.warning(f"Models found: {len(mods)}, but ping returned: {test_err}")
                else:
                    st.error(f"API Error: {err}")

    st.divider()
    st.markdown("#### 🌊 Elliott Wave (EWT) Settings")
    show_ewt_val = st.checkbox("🌊 Show Elliott Wave Cycle (1-5 / A-C)", value=st.session_state.show_ewt)
    st.session_state.show_ewt = show_ewt_val

    st.divider()
    st.markdown("#### 🛡️ Live Risk Management & Lot Size")
    c_bal = st.number_input("Account Balance ($)", value=st.session_state.account_balance, step=1000.0)
    c_risk = st.slider("Risk Per Trade (%)", 0.5, 3.0, st.session_state.risk_percent, 0.5)
    st.session_state.account_balance = c_bal
    st.session_state.risk_percent = c_risk
    st.caption(f"Risk Capital: **${(c_bal * c_risk / 100):.2f}** per trade setup")

    st.divider()
    st.markdown("#### 🔔 Telegram Alert Integration")
    tg_tok = st.text_input("Bot Token", value=st.session_state.tg_token, type="password", placeholder="789... from @BotFather")
    tg_cid = st.text_input("Chat ID", value=st.session_state.tg_chat_id, placeholder="123... from @userinfobot")
    if tg_tok != st.session_state.tg_token or tg_cid != st.session_state.tg_chat_id:
        st.session_state.tg_token = tg_tok.strip()
        st.session_state.tg_chat_id = tg_cid.strip()
        st.success("Credentials saved!")

    now_ny = datetime.now(NY_TZ)
    mins_to_close = 15 - (now_ny.minute % 15)
    secs_to_close = 60 - now_ny.second
    next_close_min = ((now_ny.minute // 15 + 1) * 15) % 60
    st.info(f"⏳ Next 15m Candle Closes in: **{mins_to_close - 1}m {secs_to_close}s** (at :{next_close_min:02d}:00 EST)")

    if st.button("📲 Trigger Live 15m Telegram Alert", use_container_width=True):
        if not st.session_state.tg_token or not st.session_state.tg_chat_id:
            st.error("Fill Token & Chat ID!")
        else:
            with st.spinner("Fetching Deriv 15m spot candles & generating alert..."):
                df_a, p_a, src_a, err_a = fetch_live_gold_candles(granularity=900, count=40)
                if df_a is not None and not df_a.empty:
                    s_label, _ = get_live_ny_session()
                    m_struct = compute_deterministic_market_structure(df_a, account_balance=st.session_state.account_balance, risk_percent=st.session_state.risk_percent)
                    img_a = render_native_cameron4x_chart(
                        df_a,
                        math_structure=m_struct,
                        session_txt=s_label,
                        show_ewt=bool(st.session_state.get("show_ewt", True))
                    )
                    a_text = f"""🔔 *DERIV LIVE 15m GOLD (frxXAUUSD) ALERT* 🔔
━━━━━━━━━━━━━━━━━━━━
🎯 *Signal:* {m_struct['verdict']} ({m_struct['action']})
📍 *Spot Price:* ${p_a:.2f}
🌊 *Wave Phase:* {m_struct.get('elliott_wave', {}).get('cycle_phase', 'INCOMPLETE_CYCLE')}
🏛️ *Feed Source:* {src_a}
⏰ *NY Session:* {s_label}
━━━━━━━━━━━━━━━━━━━━"""
                    ok, err_d = send_telegram_alert(st.session_state.tg_token, st.session_state.tg_chat_id, a_text, pil_image=img_a)
                    if ok: st.success("✅ Real-time 15m Deriv chart & alert sent to Telegram!")
                    else: st.error(err_d)
                else:
                    st.error(f"Live feed error: {err_a}")

    st.divider()
    st.markdown("#### 📚 Search 108 ICT & EWT Topics")
    search_q = st.text_input("Search Topics", placeholder="e.g. Elliott Wave, FVG, MSS, Killzone")
    if search_q:
        matched = knowledge_store.retrieve(search_q, top_k=6)
        for t in matched:
            st.markdown(f"• **#{t['id']}** {t['name']}")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.processed_file_names = set()
        st.rerun()

# -----------------------------------------------------------------------------
# MAIN CHATGPT HEADER WITH LIVE SPOT TICKER & INTERBANK CLOCK
# -----------------------------------------------------------------------------
session_label, session_color = get_live_ny_session()
spot_display = f"${st.session_state.last_spot_price:.2f}" if st.session_state.last_spot_price else "Live Connecting..."

st.markdown(f"""
<div class="chat-header">
    <div>
        <span class="chat-title">🧠 Gold ICT AGI</span>
        <span class="chat-badge">SWARM AGI + EWT</span>
        <span class="chat-ticker">🟡 frxXAUUSD: <b>{spot_display}</b></span>
    </div>
    <div class="chat-status" style="color: {session_color} !important;">
        ● {session_label}
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HERO SECTION (IF NO MESSAGES YET)
# -----------------------------------------------------------------------------
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Deterministic Institutional Gold 15m Co-Pilot</div>
        <div class="hero-subtitle">
            All market structure levels (Sweeps, MSS, 3-Candle FVGs, 50% CE, and <b>Elliott Wave Cycles</b>) are mathematically calculated in deterministic Python before passing to the <b>4-Agent AGI Swarm</b>.<br>
            Click <b>Scan Live 15m Gold</b> below to run the live Deriv WebSocket audit with Unanimous Consensus Governance.
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
                st.image(msg["image"], width=280, caption="Uploaded Chart")
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            # 1. Pure Cameron4x Marked Candlestick Chart
            if "marked_img" in msg and msg["marked_img"] is not None:
                st.image(msg["marked_img"], caption="⚡ Cameron4x Candlestick Chart (MSS + Sweep + FVG + 50% CE + Elliott Wave Badges)", use_container_width=True)

            # 2. Executive Trade Card (Ultra High Contrast)
            if "trade_card" in msg and msg["trade_card"] is not None:
                tc = msg["trade_card"]
                is_exec = tc.get("action") == "EXECUTE_LIMIT"
                border_col = "#10B981" if (is_exec and "BUY" in tc["verdict"]) else ("#EF4444" if (is_exec and "SELL" in tc["verdict"]) else "#F59E0B")
                grade = tc.get("setup_grade", "A+ Institutional (High Probability)")
                rec_lot = tc.get("recommended_lot", 0.0)
                risk_amt = tc.get("risk_amount", 0.0)
                action_badge = "🟢 UNANIMOUS CONSENSUS APPROVED (EXECUTE LIMIT)" if is_exec else "⏸️ UNANIMOUS CONSENSUS: WAIT (Criteria Pending)"

                ewt_dict = tc.get("elliott_wave", {})
                ewt_ph = ewt_dict.get("cycle_phase", "INCOMPLETE_CYCLE")
                ewt_w3 = f"{ewt_dict.get('fib_w3_ratio', 0.0):.2f}x"
                ewt_inv = f"${ewt_dict.get('invalidation_price', 0.0):.2f}"

                st.markdown(f"""
                <div class="trade-card" style="border-left: 6px solid {border_col} !important;">
                    <div style="font-size: 0.85rem; font-weight: 800; margin-bottom: 6px;">{action_badge}</div>
                    <div class="trade-verdict" style="color: {border_col} !important;">
                        🎯 {tc['verdict']} <span class="trade-grade-badge">⭐ {grade}</span>
                    </div>
                    <div class="trade-setup">{tc['setup_name']}</div>
                    <div class="trade-levels">
                        • <span class="trade-lbl">Spot Price (frxXAUUSD):</span> <b class="trade-val-price">${tc.get('chart_price', 0):.2f}</b><br>
                        • <span class="trade-lbl">Elliott Wave Cycle:</span> <b style="color: #38BDF8 !important;">🌊 {ewt_ph}</b> &nbsp;|&nbsp; <span class="trade-lbl">Fib W3 Ext:</span> <b class="trade-val-yellow">{ewt_w3}</b> &nbsp;|&nbsp; <span class="trade-lbl">EWT Invalidation:</span> <b class="trade-val-red">{ewt_inv}</b><br>
                        • <span class="trade-lbl">Entry (50% CE / Key Level):</span> <b class="trade-val-cyan">${tc['entry_p']:.2f}</b><br>
                        • <span class="trade-lbl">Invalidation Stop Loss:</span> <b class="trade-val-red">${tc['sl_p']:.2f}</b><br>
                        • <span class="trade-lbl">Institutional TP1 (1:2.0 R:R):</span> <b class="trade-val-green">${tc['tp_p']:.2f}</b> &nbsp;|&nbsp; <span class="trade-lbl">Runner TP2:</span> <b style="color: #38BDF8 !important;">${tc.get('tp2_runner', tc['tp_p']):.2f}</b><br>
                        • <span class="trade-lbl">Calculated Risk:Reward:</span> <b class="trade-val-yellow">{tc.get('rr_ratio', 2.0):.2f} R:R</b> &nbsp;|&nbsp; <span class="trade-lbl">Recommended Lot:</span> <b style="color: #FBBF24 !important;">{rec_lot} Lots</b> (${risk_amt:.2f})<br>
                        • <span class="trade-lbl">Dealing Range:</span> <b style="color: #FFFFFF !important;">{tc['range_zone']}</b> &nbsp;|&nbsp; <span class="trade-lbl">Confluence Score:</span> <b class="trade-val-yellow">{tc['confluence_score']}%</b><br>
                        • <span class="trade-lbl">🪤 Trapped Liquidity:</span> <b style="color: #F87171 !important;">{tc.get('trapped_liquidity', 'Retail breakout participants trapped')}</b><br>
                        • <span class="trade-lbl">🛑 Invalidation Rule:</span> <b style="color: #FBBF24 !important;">{tc.get('invalidation_thesis', 'Candle close beyond SL level')}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if not is_exec and tc.get("rejection_reasons"):
                    reasons_html = "".join([f"<li>{r}</li>" for r in tc["rejection_reasons"]])
                    st.warning(f"**Missing Institutional Criteria for Execution:**\n<ul>{reasons_html}</ul>", icon="⚠️")

                if "matched_topics" in msg and msg["matched_topics"]:
                    t_html = "".join([f"<span class='topic-pill'><b>#{t['id']}</b> {t['name']}</span>" for t in msg["matched_topics"]])
                    st.markdown(f"<div style='margin-bottom: 14px; color: #FFFFFF !important;'><b style='color: #F8FAFC !important;'>📚 108-Topics Applied:</b><br>{t_html}</div>", unsafe_allow_html=True)

            # 4. Multi-Agent Swarm Debate Expander (4 Agents)
            if "agent_swarm" in msg and msg["agent_swarm"]:
                aswarm = msg["agent_swarm"]
                with st.expander("🤖 AGI 4-Agent Swarm Debate & Unanimous Consensus Protocol"):
                    st.markdown(f"**🕵️ Agent 1 (Liquidity Detective):** {aswarm.get('liquidity_detective', 'Analyzed sweep dynamics.')}")
                    st.markdown(f"**📐 Agent 2 (ICT Structure Auditor):** {aswarm.get('structure_auditor', 'Verified MSS and displacement energy.')}")
                    st.markdown(f"**🌊 Agent 3 (Elliott Wave Specialist):** {aswarm.get('elliott_wave_specialist', 'Audited 5-wave motive sequence and cardinal rules.')}")
                    st.markdown(f"**⚖️ Agent 4 (Chief Risk Officer):** {aswarm.get('risk_officer', 'Validated 1:2.0 R:R and risk sizing.')}")

            # 5. AI Explanation & Reasoning
            st.markdown(msg["content"])

            # 6. One-Click Telegram Alert Dispatch
            if "trade_card" in msg and msg["trade_card"] is not None:
                if st.button("📲 Send AGI Blueprint to Telegram", key=f"tg_btn_{msg.get('msg_id', 0)}"):
                    tok = st.session_state.get("tg_token", "")
                    cid = st.session_state.get("tg_chat_id", "")
                    if not tok or not cid:
                        st.warning("⚠️ Pehle Sidebar mein Telegram Bot Token aur Chat ID enter karein!")
                    else:
                        tc = msg["trade_card"]
                        ewt_c = tc.get("elliott_wave", {})
                        tg_text = f"""🔔 *GOLD (XAU/USD) 15m AGI BLUEPRINT* 🔔
━━━━━━━━━━━━━━━━━━━━
🎯 *Signal:* {tc['verdict']} ({tc.get('setup_grade', 'A+')})
📋 *Action:* {tc.get('action', 'WAIT')}
🌊 *Wave Cycle:* {ewt_c.get('cycle_phase', 'INCOMPLETE_CYCLE')}
📍 *Entry (50% CE):* ${tc['entry_p']:.2f}
🛑 *Stop Loss:* ${tc['sl_p']:.2f}
🎯 *TP1 (1:2 R:R):* ${tc['tp_p']:.2f}
🎯 *TP2 (Runner):* ${tc.get('tp2_runner', tc['tp_p']):.2f}
📊 *Recommended Lot:* {tc.get('recommended_lot', 0.2)} Lots
🧠 *Confluence:* {tc['confluence_score']}%
🪤 *Trap:* {tc.get('trapped_liquidity', 'Retail trapped')}
━━━━━━━━━━━━━━━━━━━━"""
                        ok, err_desc = send_telegram_alert(tok, cid, tg_text, pil_image=msg.get("marked_img"))
                        if ok: st.success("📲 Marked chart photo & AGI blueprint sent to your Telegram phone!")
                        else: st.error(err_desc)

# -----------------------------------------------------------------------------
# QUICK ACTION BAR (DUAL-MODE: DERIV WEBSOCKET SCAN & SCREENSHOT UPLOAD)
# -----------------------------------------------------------------------------
st.write("")
col_scan, col_ref, col_up = st.columns([2.5, 1.2, 2.3])
with col_scan:
    scan_live_clicked = st.button(
        "⚡ Scan Live 15m Gold (Deriv Spot)",
        use_container_width=True,
        type="primary",
        help="Pull live 15m candlesticks from Deriv WebSocket & execute deterministic math engine"
    )
with col_ref:
    if st.button("🔄 Refresh", use_container_width=True, help="Update live prices and session clock"):
        df_quick, p_quick, s_quick, _ = fetch_live_gold_candles(granularity=900, count=10)
        if df_quick is not None and not df_quick.empty:
            st.session_state.last_spot_price = p_quick
            st.session_state.last_spot_source = s_quick
        st.rerun()
with col_up:
    uploaded_chart = st.file_uploader(
        "📎 Upload Screenshot (Optional)",
        type=["png", "jpg", "jpeg"],
        key="chat_uploader",
        help="Upload TradingView screenshot to analyze and mark candles"
    )

prompt_input = st.chat_input("Ask AGI anything about Gold 15m or type follow-up...")

is_new_chart_upload = uploaded_chart is not None and uploaded_chart.name not in st.session_state.processed_file_names
has_text_prompt = prompt_input is not None and prompt_input.strip() != ""
trigger_live_scan = scan_live_clicked

if trigger_live_scan or is_new_chart_upload or has_text_prompt:
    input_img = None
    user_text = ""
    is_live_deriv_mode = False

    if trigger_live_scan:
        is_live_deriv_mode = True
        user_text = "⚡ Scan Live 15m Gold (Deriv Spot frxXAUUSD) with Deterministic Algorithmic Math, Elliott Wave Cycle Validation, and 4-Agent Unanimous Consensus."
    elif is_new_chart_upload:
        st.session_state.processed_file_names.add(uploaded_chart.name)
        input_img = Image.open(uploaded_chart)
        user_text = prompt_input if has_text_prompt else "Audit this Gold 15m chart against deterministic ICT market structure and Elliott Wave cycle, then render the Cameron4x candlestick chart."
    else:
        user_text = prompt_input

    user_msg_entry = {"role": "user", "content": user_text, "image": input_img}
    st.session_state.messages.append(user_msg_entry)
    
    with st.chat_message("user"):
        if input_img is not None:
            st.image(input_img, width=280, caption="Uploaded Chart")
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Executing deterministic vector calculations & 4-Agent Unanimous Consensus Protocol..."):
            marked_chart_pil = None
            trade_card_data = None
            matched_topics = []
            agent_swarm_data = {}
            ai_reply = ""

            is_kz_pass, active_kz, kz_code, kz_desc = knowledge_store.get_killzone_window(datetime.now(NY_TZ))
            df_live, spot_price, feed_source, feed_err = fetch_live_gold_candles(granularity=900, count=45)
            
            if df_live is not None and not df_live.empty:
                st.session_state.last_spot_price = spot_price
                st.session_state.last_spot_source = feed_source

                # 1. RUN PURE DETERMINISTIC VECTOR CALCULATIONS & EWT ANALYSIS
                math_struct = compute_deterministic_market_structure(
                    df_live,
                    account_balance=st.session_state.account_balance,
                    risk_percent=st.session_state.risk_percent
                )

                if not is_kz_pass:
                    math_struct["rejection_reasons"].append(f"Temporal Filter: Outside active liquidity window ({kz_code}: {kz_desc})")
                    math_struct["action"] = "WAIT"

                ewt_res = math_struct.get("elliott_wave", {})

                # 2. MULTI-AGENT SWARM GOVERNANCE PROMPT (4 AGENTS)
                swarm_governance_prompt = f"""
You are the Supreme Institutional Governance Protocol for Gold (XAU/USD / frxXAUUSD) 15m trading.
The algorithmic math engine has computed the deterministic market structure and Elliott Wave cycle:

DETERMINISTIC MARKET STRUCTURE:
{json.dumps(math_struct, indent=2)}

KILLZONE EVALUATION:
- Status: {kz_code}
- Active Window: {active_kz}
- Details: {kz_desc}
- Pass Flag: {is_kz_pass}

ELLIOTT WAVE CYCLE:
- Status: {ewt_res.get('status')}
- Cycle Phase: {ewt_res.get('cycle_phase')}
- Wave 3 Extension: {ewt_res.get('fib_w3_ratio')}x
- Cardinal Rules: {ewt_res.get('cardinal_rules')}
- Invalidation Price: {ewt_res.get('invalidation_price')}

UNANIMOUS CONSENSUS RULES (4 AGENTS):
1. Agent 1 (Liquidity Detective): Must confirm sweep depth and reject internal retail inducements.
2. Agent 2 (Structure Auditor): Must confirm candle body MSS and valid unmitigated FVG at 50% CE.
3. Agent 3 (Elliott Wave Specialist): Validates cycle phase against ICT trade direction. Hard veto if CARDINAL_RULE_VIOLATION or buying/selling Wave 5 terminal exhaustion wicks.
4. Agent 4 (Chief Risk Officer): Must verify Risk-to-Reward ratio >= 2.0 and Killzone pass ({kz_code}).
5. If ANY agent rejects or if Pass Flag is false, action MUST be "WAIT".

Return ONLY a valid JSON object matching this schema:
{{
  "setup_detected": {str(math_struct['has_setup']).lower()},
  "bias": "{math_struct['bias']}",
  "sweep_level": {math_struct['sweep_level']},
  "mss_level": {math_struct['mss_level']},
  "fvg_top": {math_struct['fvg_top']},
  "fvg_bottom": {math_struct['fvg_bottom']},
  "entry_ce": {math_struct['entry_ce']},
  "stop_loss": {math_struct['stop_loss']},
  "take_profit_1": {math_struct['take_profit_1']},
  "risk_reward_ratio": {math_struct['risk_reward_ratio']},
  "action": "{math_struct['action']}",
  "rejection_reason": "{'; '.join(math_struct['rejection_reasons']) if math_struct['rejection_reasons'] else None}",
  "agent_votes": {{
    "liquidity_detective": "{"PASS" if math_struct['sweep_type'] != 'NONE' else "REJECT"}",
    "structure_auditor": "{"PASS" if math_struct['mss_confirmed'] else "REJECT"}",
    "elliott_wave_specialist": "{"PASS" if ewt_res.get('status') != 'INVALID' and not ('TERMINAL_SWEEP' in ewt_res.get('cycle_phase', '')) else "REJECT"}",
    "risk_officer": "{"PASS" if (math_struct['risk_reward_ratio'] >= 2.0 and is_kz_pass) else "REJECT"}"
  }},
  "agent_swarm": {{
    "liquidity_detective": "Detailed evaluation of sweep depth, external BSL/SSL pools, and retail traps.",
    "structure_auditor": "Audit of displacement candle body ratio (>=0.70) and unmitigated FVG at 50% CE.",
    "elliott_wave_specialist": "Validation of 5-wave motive sequence, 3 Cardinal Rules, and cycle phase.",
    "risk_officer": "Verification of 1:2.0 RR minimum and Killzone window pass status."
  }},
  "explanation_hinglish": "Clear, professional institutional reasoning in Hinglish summarizing the 4-agent consensus."
}}
"""
                contents = [swarm_governance_prompt]
                if input_img is not None:
                    contents = [input_img.convert("RGB"), swarm_governance_prompt]

                gemini_raw, used_model, gemini_err = call_gemini(contents, preferred_model_type=st.session_state.agi_model_mode)
                
                final_action = math_struct["action"]
                final_rejection = "; ".join(math_struct["rejection_reasons"]) if math_struct["rejection_reasons"] else None
                
                if gemini_raw:
                    g_data = parse_gemini_json(gemini_raw)
                    if g_data and isinstance(g_data, dict):
                        agent_votes = g_data.get("agent_votes", {})
                        v1 = agent_votes.get("liquidity_detective", "PASS") == "PASS"
                        v2 = agent_votes.get("structure_auditor", "PASS") == "PASS"
                        v3 = agent_votes.get("elliott_wave_specialist", "PASS") == "PASS"
                        v4 = agent_votes.get("risk_officer", "PASS") == "PASS"

                        if not (v1 and v2 and v3 and v4 and is_kz_pass and math_struct["has_setup"]):
                            final_action = "WAIT"
                            final_rejection = g_data.get("rejection_reason") or final_rejection or "Unanimous Consensus not achieved"
                        else:
                            final_action = g_data.get("action", math_struct["action"])

                        agent_swarm_data = g_data.get("agent_swarm", {})
                        ai_reply = g_data.get("explanation_hinglish", gemini_raw)
                    else:
                        ai_reply = gemini_raw
                else:
                    agent_swarm_data = {
                        "liquidity_detective": f"Sweep Status: {math_struct['sweep_type']} at ${math_struct['sweep_level']:.2f}.",
                        "structure_auditor": f"MSS Status: {'Confirmed' if math_struct['mss_confirmed'] else 'Pending'} at ${math_struct['mss_level']:.2f}.",
                        "elliott_wave_specialist": f"Wave Cycle: {ewt_res.get('cycle_phase')} (W3 Ext: {ewt_res.get('fib_w3_ratio')}x).",
                        "risk_officer": f"R:R Ratio: {math_struct['risk_reward_ratio']} | Killzone Gate: {kz_code} ({'PASS' if is_kz_pass else 'HOLD'})."
                    }
                    ai_reply = f"""### 🧠 Deterministic Institutional Consensus Report:
- **Feed:** `{feed_source}` | **Current Spot:** **${spot_price:.2f}**
- **Action Verdict:** `{final_action}` ({math_struct['verdict']})
- **Elliott Wave Cycle:** `{ewt_res.get('cycle_phase')}` (W3 Ext: **{ewt_res.get('fib_w3_ratio')}x**)
- **Liquidity Sweep:** {math_struct['sweep_type']} at **${math_struct['sweep_level']:.2f}**
- **Market Structure Shift:** {'Confirmed' if math_struct['mss_confirmed'] else 'Pending'} at **${math_struct['mss_level']:.2f}**
- **50% CE Entry:** **${math_struct['entry_ce']:.2f}** | **SL:** **${math_struct['stop_loss']:.2f}** | **TP1:** **${math_struct['take_profit_1']:.2f}** (1:2.0 R:R)
- **Killzone Evaluation:** {kz_desc}
"""

                # 3. RENDER PURE CAMERON4X CANDLESTICK CHART (BOUND STRICTLY TO COMPUTED FLOATS)
                try:
                    marked_chart_pil = render_native_cameron4x_chart(
                        df_live,
                        math_structure=math_struct,
                        session_txt=session_label,
                        show_ewt=bool(st.session_state.get("show_ewt", True))
                    )
                except Exception as chart_err:
                    marked_chart_pil = None

                matched_topics = [
                    {"id": 13, "name": "Liquidity Sweep / Turtle Soup"},
                    {"id": 4, "name": "Market Structure Shift (MSS)"},
                    {"id": 17, "name": "Fair Value Gap (FVG)"},
                    {"id": 28, "name": "Consequent Encroachment (50% CE)"},
                    {"id": 94, "name": "Risk-to-Reward Ratio (1:2.0 Minimum)"},
                    {"id": 105, "name": "5-Wave Motive Impulse Sequence"},
                    {"id": 108, "name": "ICT & Elliott Wave Fusion"}
                ]

                trade_card_data = {
                    "verdict": math_struct["verdict"],
                    "action": final_action,
                    "setup_grade": math_struct["setup_grade"],
                    "setup_name": math_struct["setup_name"],
                    "chart_price": spot_price,
                    "entry_p": math_struct["entry_ce"],
                    "sl_p": math_struct["stop_loss"],
                    "tp_p": math_struct["take_profit_1"],
                    "tp2_runner": math_struct["take_profit_2"],
                    "rr_ratio": math_struct["risk_reward_ratio"],
                    "confluence_score": math_struct["confluence_score"],
                    "range_zone": math_struct["range_zone"],
                    "trapped_liquidity": math_struct["trapped_liquidity"],
                    "invalidation_thesis": math_struct["invalidation_thesis"],
                    "recommended_lot": math_struct["recommended_lot"] if final_action == "EXECUTE_LIMIT" else 0.0,
                    "risk_amount": math_struct["risk_amount"] if final_action == "EXECUTE_LIMIT" else 0.0,
                    "rejection_reasons": math_struct["rejection_reasons"] if final_action == "WAIT" else [],
                    "elliott_wave": ewt_res
                }
            else:
                ai_reply = f"⚠️ Unable to fetch live Gold feed: {feed_err}. Please check your connection."

            assistant_entry = {
                "role": "assistant",
                "content": ai_reply,
                "marked_img": marked_chart_pil,
                "trade_card": trade_card_data,
                "matched_topics": matched_topics,
                "agent_swarm": agent_swarm_data,
                "msg_id": len(st.session_state.messages)
            }
            st.session_state.messages.append(assistant_entry)
            st.rerun()
