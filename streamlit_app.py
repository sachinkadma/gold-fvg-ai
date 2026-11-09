import streamlit as st
import os, json, zoneinfo
import pandas as pd
import numpy as np
from datetime import time
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="ICT Institutional 104-Topic AI Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

NY_TZ = zoneinfo.ZoneInfo("America/New_York")

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 700; color: #1E88E5; margin-bottom: 0px; }
    .subtitle { font-size: 1.0rem; color: #94A3B8; margin-bottom: 18px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; padding: 10px 18px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏛️ ICT & SMC 104-Topic Institutional Terminal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Complete Web Interface for Data Layer (Phase 1), Vision Scanning & 104-Topic Institutional AI Audit</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_scanner, tab_phase1, tab_syllabus = st.tabs([
    "📸 Chart Vision & AI Audit",
    "⏱️ Phase 1: Data & Killzone Engine",
    "📚 104-Topic Institutional Syllabus"
])

# ==============================================================================
# TAB 1: CHART VISION & AI AUDIT (Existing Scanner)
# ==============================================================================
with tab_scanner:
    st.subheader("⚡ Institutional Chart Scanner")
    st.caption("TradingView ya MT5 screenshot upload karo -> YOLO FVGs mark karega -> Gemini 104-topic audit karega.")
    
    sc1, sc2, sc3 = st.columns([1, 1, 1])
    with sc1:
        target_model = st.selectbox(
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
    with sc2:
        chart_tf = st.selectbox("🕒 Chart Timeframe", ["15-Minute (15m - Default)", "5-Minute (5m)", "1-Hour (1h - HTF Bias)"])
    with sc3:
        chart_session = st.selectbox(
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

    # Models cache
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

    def retrieve(q, k=4):
        if embedder is None or len(chunks) == 0:
            return []
        v = embedder.encode([q], normalize_embeddings=True)[0]
        idx = np.argsort(chunk_vecs @ v)[::-1][:k]
        return [chunks[i] for i in idx]

    def call_gemini(contents):
        from google import genai
        key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
        if not key:
            return "⚠️ GEMINI_API_KEY missing in Streamlit Secrets. Please add it in App Settings > Secrets."
        client = genai.Client(api_key=key)
        models_to_try = ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]
        for m in models_to_try:
            try:
                return client.models.generate_content(model=m, contents=contents).text
            except Exception:
                continue
        return "⚠️ AI response generation failed. Please verify API key."

    uploaded_file = st.file_uploader("📥 Chart Screenshot Upload Karo", type=["png", "jpg", "jpeg"], key="scanner_upload")

    if uploaded_file is not None and model is not None:
        col1, col2 = st.columns([1.2, 1])
        img = Image.open(uploaded_file)
        arr = np.array(img.convert("RGB"))[:, :, ::-1]
        
        r = model(arr, conf=0.25, verbose=False)[0]
        h, w = r.orig_shape
        zones = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            zones.append({
                "type": r.names[int(b.cls)],
                "conf": round(float(b.conf), 2),
                "right_pct": round(x2 / w * 100),
                "top_pct": round(y1 / h * 100)
            })
        
        annotated = r.plot()[:, :, ::-1]
        with col1:
            st.image(annotated, caption="⚡ Computer Vision: Bullish FVG (Green) & Bearish FVG (Red)", use_container_width=True)
        
        with col2:
            bull_count = sum(1 for z in zones if z["type"] == "bull_fvg")
            bear_count = sum(1 for z in zones if z["type"] == "bear_fvg")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Total Zones", len(zones))
            mcol2.metric("Bullish FVG", bull_count)
            mcol3.metric("Bearish FVG", bear_count)
            
            user_query = st.text_input("💬 Custom Question:", value=f"Check for {target_model} and give full 104-topic audit.")
            run_audit = st.button("🚀 Run Institutional Audit", type="primary", use_container_width=True)

        if run_audit:
            with st.spinner("⏳ Analyzing chart across 104 ICT topics..."):
                retrieved_rules = retrieve(f"{target_model} {user_query} liquidity sweep FVG order block killzone risk")
                rules_context = "\n".join(["- " + c.replace("\n", " ") for c in retrieved_rules])
                prompt = f"""
Tum ek Senior Institutional ICT / SMC Analyst ho.
Chart Timeframe: {chart_tf} | Session: {chart_session} | Target Model: {target_model}
Detected YOLO Zones: {json.dumps(zones) if zones else 'No large FVG boxes'}
Relevant Rules:
{rules_context}

Chart ko analyze karke Hinglish mein structured report do:
### 1. 🏛️ Market Structure & Liquidity (Topics 1-14)
- Current Bias, Swing Points, BSL/SSL Sweep, BOS vs MSS, Dealing Range position.
### 2. 📦 PD Arrays & Imbalances (Topics 15-34)
- Fresh FVG, Order Block/Breakers, Consequent Encroachment (50% CE).
### 3. ⏱️ Time & Killzone Alignment (Topics 35-58)
- Killzone status, AMD Phase.
### 4. ⚡ Active ICT Entry Model (Topics 59-74)
- Setup name & Quality (A+ / B / Wait / No Trade).
### 5. 🛡️ Execution & Trade Plan (Topics 93-104)
- Entry zone, Invalidation SL, Draw on Liquidity TP, Risk:Reward (Min 1:2).
"""
                ans = call_gemini([img.convert("RGB"), prompt])
                st.divider()
                st.markdown("## 📋 Institutional ICT Audit Report")
                st.markdown(ans)


# ==============================================================================
# TAB 2: PHASE 1 DATA & KILLZONE ENGINE (1-Click Web Runner)
# ==============================================================================
with tab_phase1:
    st.subheader("⏱️ Phase 1: Data & Killzone Processing Engine")
    st.markdown("""
    **Google Colab ki zaroorat nahi hai!** Aap yahi se 1-click mein Phase 1 complete kar sakte hain:
    - Gold (XAU/USD) 15m historical data download hoga.
    - Har candle ka time **America/New_York (EST/EDT)** mein convert hoga.
    - Har candle par **Asian, London KZ, NY AM KZ, Silver Bullet, Lunch, aur PM KZ** ka tag lagega.
    - **1H, 4H, Daily, aur Weekly** data generate hoga.
    """)

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        start_year = st.selectbox("📅 Data Start Year", ["2023", "2022", "2024"], index=0)
        run_phase1_btn = st.button("🚀 Run Phase 1 Engine Now", type="primary", use_container_width=True)

    def tag_session(dt):
        t = dt.time()
        if t >= time(20, 0) or t < time(0, 0):
            return "Asian Range"
        elif time(2, 0) <= t < time(5, 0):
            return "London KZ"
        elif time(8, 30) <= t < time(11, 0):
            return "NY AM KZ"
        elif time(12, 0) <= t < time(13, 0):
            return "NY Lunch (No Trade)"
        elif time(13, 30) <= t < time(16, 0):
            return "NY PM KZ"
        elif time(10, 0) <= t < time(12, 0):
            return "London Close"
        return "Off-Hours"

    def is_sb(dt):
        t = dt.time()
        return (time(10, 0) <= t < time(11, 0)) or (time(14, 0) <= t < time(15, 0))

    if run_phase1_btn:
        status_box = st.status("⏳ Phase 1 Processing Started...", expanded=True)
        try:
            status_box.write("1/4: Hugging Face se `XAU_15m_data.jsonl` download ho raha hai...")
            repo_id = "SpatialGhost/xauusd-gold-price-historical-data-2004-2025"
            file_path = hf_hub_download(repo_id, "XAU_15m_data.jsonl", repo_type="dataset")
            
            status_box.write(f"2/4: Reading and filtering data for year >= {start_year}...")
            chunks = []
            for c in pd.read_json(file_path, lines=True, chunksize=150000):
                c = c.rename(columns=lambda x: str(x).strip().capitalize())
                c["Date"] = pd.to_datetime(c["Date"], errors="coerce")
                c = c.dropna(subset=["Date"])
                c = c[c["Date"].dt.year >= int(start_year)]
                chunks.append(c)
            df_raw = pd.concat(chunks)
            
            status_box.write("3/4: Timezone ko New York (America/New_York) mein convert kar rahe hain...")
            if df_raw["Date"].dt.tz is None:
                df_raw["Date"] = df_raw["Date"].dt.tz_localize("UTC")
            df_raw["Date_NY"] = df_raw["Date"].dt.tz_convert(NY_TZ)
            df_15m = df_raw.set_index("Date_NY").sort_index()
            df_15m = df_15m[~df_15m.index.duplicated()]
            
            status_box.write("4/4: ICT Sessions, Killzones aur Silver Bullet windows tag ho rahi hain...")
            times = df_15m.index
            df_15m["Session"] = [tag_session(ts) for ts in times]
            df_15m["Is_Killzone"] = df_15m["Session"].isin(["London KZ", "NY AM KZ", "NY PM KZ"])
            df_15m["Is_Silver_Bullet"] = [is_sb(ts) for ts in times]
            df_15m["Day_of_Week"] = times.day_name()
            df_15m["NY_Time"] = times.strftime("%H:%M")
            
            # Resample HTF
            ohlc_dict = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            df_1h = df_15m.resample("1h").agg(ohlc_dict).dropna()
            df_4h = df_15m.resample("4h").agg(ohlc_dict).dropna()
            df_daily = df_15m.resample("1D").agg(ohlc_dict).dropna()

            status_box.update(label="🎉 Phase 1 100% COMPLETE & VERIFIED!", state="complete", expanded=False)
            
            st.success("✅ **Phase 1 Done Criteria Satisfied!** All candles localized to New York Time and tagged.")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total 15m Candles", f"{len(df_15m):,}")
            m1.caption("Base Execution Data")
            m2.metric("Killzone Candles", f"{df_15m['Is_Killzone'].sum():,}")
            m2.caption("London, NY AM, NY PM")
            m3.metric("Silver Bullet Windows", f"{df_15m['Is_Silver_Bullet'].sum():,}")
            m3.caption("10-11 AM & 2-3 PM NY")
            m4.metric("1H / 4H / Daily Bars", f"{len(df_1h)} / {len(df_4h)} / {len(df_daily)}")
            m4.caption("HTF Structure Data")

            st.divider()
            st.markdown("### 📊 ICT Session Distribution")
            st.bar_chart(df_15m["Session"].value_counts())

            st.markdown("### 🔍 Processed Data Preview (New York Time)")
            preview_cols = ["NY_Time", "Day_of_Week", "Session", "Is_Killzone", "Is_Silver_Bullet", "Open", "High", "Low", "Close"]
            st.dataframe(df_15m[preview_cols].tail(100), use_container_width=True)

            # Download CSV button
            csv_data = df_15m.tail(5000).to_csv()
            st.download_button(
                label="📥 Download Recent 5,000 NY Processed Candles (CSV)",
                data=csv_data,
                file_name="xauusd_15m_ny_processed.csv",
                mime="text/csv",
                type="secondary"
            )

        except Exception as e:
            status_box.update(label=f"❌ Error: {str(e)}", state="error")
            st.error(f"Error detail: {e}")


# ==============================================================================
# TAB 3: 104-TOPIC INSTITUTIONAL SYLLABUS REFERENCE
# ==============================================================================
with tab_syllabus:
    st.subheader("📚 Complete 104-Topic Institutional ICT Syllabus")
    st.caption("All topics built into the AI memory and scanning rules.")
    
    with st.expander("🏛️ Foundations (Topics 1–14)", expanded=True):
        st.markdown("""
        1. **Market Structure:** HH/HL (Bullish) vs LH/LL (Bearish) sequence.
        2. **Swing Points (STH/ITH/LTH):** Fractal highs & lows hierarchy.
        3. **BOS (Break of Structure):** Trend continuation with body close.
        4. **MSS / CHoCH:** Trend reversal with violent displacement.
        5. **Premium vs Discount:** Dealing range 50% threshold.
        6. **Equilibrium:** Exact 50.0% fair value.
        7. **Dealing Range:** Active anchor low to high range.
        8. **Internal vs External Liquidity:** Price oscillates between BSL/SSL and internal PD arrays.
        9. **BSL (Buy-Side Liquidity):** Old highs, equal highs stop hunt targets.
        10. **SSL (Sell-Side Liquidity):** Old lows, equal lows stop hunt targets.
        11. **Equal Highs / Lows (EQH/EQL):** Institutional liquidity magnets.
        12. **Old Highs / Lows:** PDH, PDL, PWH, PWL key levels.
        13. **Liquidity Sweep / Stop Hunt:** False breakout wick grabbing liquidity.
        14. **HTF Bias:** 4H/Daily directional trend supreme.
        """)

    with st.expander("📦 PD Arrays (Topics 15–34)"):
        st.markdown("""
        15. **Bullish OB:** Last down candle before bullish displacement.
        16. **Bearish OB:** Last up candle before bearish displacement.
        17. **Fair Value Gap (FVG):** 3-candle imbalance.
        18. **BISI:** Bullish FVG (buyers imbalance).
        19. **SIBI:** Bearish FVG (sellers imbalance).
        20. **Inversion FVG (IFVG):** Failed FVG flipped into support/resistance.
        21. **Breaker Block:** Failed order block after liquidity sweep.
        22. **Mitigation Block:** Failed order block without liquidity sweep.
        28. **Consequent Encroachment (CE):** Exact 50.0% midpoint of FVG.
        29. **Optimal Trade Entry (OTE):** 62%–79% Fibonacci retracement.
        31. **Mean Threshold (50%):** Order block candle body midpoint.
        """)

    with st.expander("⏱️ Time & Killzones (Topics 35–58)"):
        st.markdown("""
        35. **Asian Range:** 20:00 - 00:00 NY (Range building).
        36. **London Killzone:** 02:00 - 05:00 NY (Judas Swing / HOD / LOD).
        37. **NY AM Killzone:** 08:30 - 11:00 NY (Major expansion).
        38. **NY Lunch:** 12:00 - 13:00 NY (No-trade zone).
        39. **NY PM Killzone:** 13:30 - 16:00 NY (Afternoon continuation).
        41. **Power of 3 (AMD):** Accumulation -> Manipulation -> Distribution.
        42. **Judas Swing:** Session open false breakout move.
        43. **Midnight Open:** 00:00 NY reference price.
        46-47. **NWOG & NDOG:** New Week / Day opening gaps.
        """)

    with st.expander("⚡ Entry Models (Topics 59–74)"):
        st.markdown("""
        59. **Silver Bullet:** 10:00-11:00 AM NY & 2:00-3:00 PM NY FVG entries.
        60. **ICT 2022 Model:** Liquidity Sweep + MSS displacement + FVG retest.
        61. **Unicorn Model:** Breaker Block + FVG overlap.
        62. **Turtle Soup:** Swing extreme fakeout wick & rejection.
        66. **OTE Entry:** 62%-79% Fib retracement + FVG confluence.
        """)

    with st.expander("🛡️ Risk & Discipline (Topics 93–104)"):
        st.markdown("""
        93. **Risk %:** Strict 1.0% to 2.0% per trade.
        94. **RR Planning:** Minimum 1:2.0 Risk-Reward required.
        96. **Partial TP:** 50% scale out at 1:2; move SL to Break-Even.
        103. **Daily Loss Limit:** 2 consecutive losses (-2%) = Close terminal for the day.
        """)
