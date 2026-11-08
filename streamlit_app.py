import streamlit as st
import os, json
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="ICT Institutional 104-Topic AI Scanner",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0px; }
    .subtitle { font-size: 1.05rem; color: #78909C; margin-bottom: 20px; }
    .metric-card { background: #1E293B; border-radius: 8px; padding: 14px; border: 1px solid #334155; }
    .badge-bull { background-color: #10B981; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .badge-bear { background-color: #EF4444; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏛️ ICT & SMC 104-Topic Institutional Chart Scanner</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Algorithmic Order Flow, PD Arrays, Killzones & Execution Models for XAU/USD (Gold)</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Scanner Settings")
    focus_model = st.selectbox(
        "🎯 Target Entry Model",
        [
            "Auto-Detect All Models (Full 104-Topic Audit)",
            "Model #60: ICT 2022 Mentorship Model",
            "Model #59: Silver Bullet (10-11 AM KZ + FVG)",
            "Model #61: Unicorn Model (Breaker + FVG)",
            "Model #62: Turtle Soup (Liquidity Sweep)",
            "Model #66: Optimal Trade Entry (OTE 62%-79%)"
        ]
    )
    timeframe = st.selectbox("🕒 Current Chart Timeframe", ["15-Minute (15m - Default)", "5-Minute (5m)", "1-Hour (1h - HTF Bias)"])
    session_select = st.selectbox(
        "🌍 Active Session / Killzone",
        [
            "Auto / Unknown",
            "London Killzone (02:00 - 05:00 NY)",
            "NY AM Killzone (08:30 - 11:00 NY)",
            "NY Lunch (No Trade Zone)",
            "NY PM Killzone (13:30 - 16:00 NY)",
            "Asian Range (Consolidation)"
        ]
    )
    st.divider()
    st.header("💰 Risk Calculator (Topics 93-95)")
    acc_balance = st.number_input("Account Balance ($)", min_value=100.0, value=5000.0, step=500.0)
    risk_pct = st.slider("Risk Per Trade (%)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
    st.caption(f"💵 Dollar Risk Amount: **${(acc_balance * risk_pct / 100):.2f}**")
    st.divider()
    st.info("📚 **Knowledge Base**: 104 ICT Topics Active (Foundations, PD Arrays, Killzones, Execution, Risk).")

# Cache models & embeddings
@st.cache_resource
def load_models():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        return None, None, [], None
    model = YOLO(model_path)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    raw = open("notes.txt", encoding="utf-8").read() if os.path.exists("notes.txt") else ""
    chunks = [c.strip() for c in raw.split("\n\n") if len(c.strip()) > 20]
    chunk_vecs = embedder.encode(chunks, normalize_embeddings=True) if chunks else []
    return model, embedder, chunks, chunk_vecs

model, embedder, chunks, chunk_vecs = load_models()

if model is None:
    st.error("⚠️ 'best.pt' model file nahi mili! Kripya check karein.")

def retrieve(q, k=4):
    if embedder is None or len(chunks) == 0:
        return []
    v = embedder.encode([q], normalize_embeddings=True)[0]
    idx = np.argsort(chunk_vecs @ v)[::-1][:k]
    return [chunks[i] for i in idx]

def call_gemini(contents):
    from google import genai
    key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    client = genai.Client(api_key=key)
    models_to_try = ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]
    for m in models_to_try:
        try:
            return client.models.generate_content(model=m, contents=contents).text
        except Exception:
            continue
    return "⚠️ AI response generation failed. Please verify API key status."

uploaded_file = st.file_uploader("📥 Gold (XAU/USD) Chart Screenshot Upload Karo (TradingView / MT5)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns([1.2, 1])
    
    img = Image.open(uploaded_file)
    arr = np.array(img.convert("RGB"))[:, :, ::-1]
    
    # YOLO Detection
    r = model(arr, conf=0.25, verbose=False)[0]
    h, w = r.orig_shape
    zones = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        zones.append({
            "type": r.names[int(b.cls)],
            "conf": round(float(b.conf), 2),
            "left_pct": round(x1 / w * 100),
            "right_pct": round(x2 / w * 100),
            "top_pct": round(y1 / h * 100),
            "bottom_pct": round(y2 / h * 100)
        })
    
    annotated = r.plot()[:, :, ::-1]
    
    with col1:
        st.image(annotated, caption="⚡ Computer Vision Scan: Bullish FVGs (Green) & Bearish FVGs (Red)", use_container_width=True)
    
    with col2:
        bull_count = sum(1 for z in zones if z["type"] == "bull_fvg")
        bear_count = sum(1 for z in zones if z["type"] == "bear_fvg")
        
        st.markdown("### 📊 Detection Overview")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total Imbalances", len(zones))
        mcol2.metric("Bullish FVG (BISI)", bull_count)
        mcol3.metric("Bearish FVG (SIBI)", bear_count)
        
        if zones:
            latest_zone = max(zones, key=lambda z: z["right_pct"])
            pos = "Discount Zone (Lower Half)" if latest_zone["top_pct"] > 50 else "Premium Zone (Upper Half)"
            st.success(f"🎯 **Most Recent Zone**: {latest_zone['type'].upper()} ({pos}, Conf: {latest_zone['conf']*100:.0f}%)")
        else:
            st.warning("⚠️ No prominent FVG detected with confidence > 25%. Scanning visual structure...")

        user_query = st.text_input(
            "💬 Custom Question / Specific Focus:",
            value=f"Check for {focus_model} and give full 104-topic institutional ICT audit."
        )
        analyze_btn = st.button("🚀 Run Institutional ICT Audit", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner("⏳ Analyzing chart across 104 ICT topics (Structure, PD Arrays, Killzones, Models, Execution)..."):
            # Retrieve relevant notes chunks
            retrieved_rules = retrieve(f"{focus_model} {user_query} liquidity sweep FVG order block killzone risk")
            rules_context = "\n".join(["- " + c.replace("\n", " ") for c in retrieved_rules])
            
            prompt = f"""
Tum ek Senior Institutional ICT / Smart Money Concepts (SMC) Trader aur Analyst ho.
User ne apne 104-Topic Syllabus ke mutabiq XAU/USD (Gold) chart upload kiya hai.

--- DETAILS ---
Timeframe: {timeframe}
Session Focus: {session_select}
Target Model: {focus_model}
Account Risk: {risk_pct}% on ${acc_balance} (${(acc_balance * risk_pct / 100):.2f})
Detected YOLO Zones: {json.dumps(zones) if zones else 'No large FVG boxes'}
Relevant Syllabus Rules:
{rules_context}

--- INSTRUCTIONS ---
Chart image aur YOLO zones ko analyze karke Hinglish mein structured, high-probability institutional report do.
Use this exact format with headings:

### 1. 🏛️ Market Structure & Liquidity (Topics 1-14)
- **Current Bias:** (Bullish / Bearish / Consolidating)
- **Swing Points & Liquidity:** (BSL ya SSL sweep hua kya? Equal Highs/Lows [EQH/EQL] kahan hain?)
- **BOS vs MSS/CHoCH:** (Break of structure ya trend shift displacement dikh raha hai?)
- **Dealing Range Position:** (Price Premium mein hai ya Discount mein?)

### 2. 📦 PD Arrays & Imbalances (Topics 15-34)
- **Fair Value Gaps (BISI/SIBI):** (Fresh unmitigated FVG kahan hai? Consequent Encroachment 50% CE level kahan hai?)
- **Order Block / Breaker Block:** (Kya koi strong OB ya Breaker Block form hua hai?)
- **Mitigation Status:** (Zone fresh hai ya already tapped ho chuka hai?)

### 3. ⏱️ Time & Killzone Alignment (Topics 35-58)
- **Session Context:** (London / NY AM / Judas Swing manipulation ya true distribution?)
- **Power of 3 (AMD):** (Price Accumulation, Manipulation ya Distribution kis phase mein hai?)

### 4. ⚡ Active ICT Entry Model (Topics 59-74)
- **Detected Setup:** (ICT 2022 Model / Silver Bullet / Unicorn / Turtle Soup bana hai ya invalid hai?)
- **Setup Quality:** (A+ Setup / B Setup / Wait for Confirmation / No Trade)

### 5. 🛡️ Execution & Trade Plan (Topics 93-104)
- **Suggested Entry Zone:** (Approx price area ya zone reference)
- **Invalidation Level (Stop Loss):** (Exact swing extreme / OB mean threshold reference)
- **Draw on Liquidity (Take Profit):** (Magnetic liquidity target / opposite pool)
- **Risk-Reward Ratio:** (Min 1:2.0)
- **Position Sizing:** (1% risk rule adherence)
"""
            response_text = call_gemini([img.convert("RGB"), prompt])
            
            st.divider()
            st.markdown("## 📋 Institutional ICT Audit Report")
            st.markdown(response_text)
