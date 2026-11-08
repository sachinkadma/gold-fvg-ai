import streamlit as st
import os, json
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Gold FVG AI", page_icon="📈", layout="centered")

st.title("📈 Gold (XAU/USD) 15m FVG AI Assistant")
st.caption("Upload 15m candlestick chart to detect Bullish & Bearish FVGs and get Gemini AI analysis.")

# Cache model & embeddings
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
    st.error("⚠️ 'best.pt' model file nahi mili! Kripya 'best.pt' ko repo mein upload karein.")

def retrieve(q, k=3):
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
    return "AI response nahi mil paya. Kripya API key ya quota check karein."

uploaded_file = st.file_uploader("XAU/USD 15m Chart Screenshot Upload Karo", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    img = Image.open(uploaded_file)
    arr = np.array(img.convert("RGB"))[:, :, ::-1]
    
    # YOLO detect
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
    st.image(annotated, caption="Marked FVG Zones (Green: Bullish, Red: Bearish)", use_container_width=True)
    
    bull = sum(1 for z in zones if z["type"] == "bull_fvg")
    bear = sum(1 for z in zones if z["type"] == "bear_fvg")
    st.info(f"📊 **Detected Zones**: Bullish FVG: {bull} | Bearish FVG: {bear}")
    
    question = st.text_input("Apna Sawal Puchho:", value="Chart ka overall analysis do aur important zones batao.")
    
    if st.button("🚀 Analyze with Gemini AI", type="primary"):
        with st.spinner("AI analyze kar raha hai..."):
            rules = retrieve(question)
            rules_txt = "\n".join("- " + c.replace("\n", " ") for c in rules)
            prompt = (
                "Tum ek experienced Gold trader ho. Hinglish mein short aur clear analysis do.\n"
                f"User Trading Rules:\n{rules_txt}\n\n"
                f"Detected FVG Zones: {json.dumps(zones)}\n\n"
                f"Sawal: {question}"
            )
            reply = call_gemini([img.convert("RGB"), prompt])
            st.markdown("### 🧠 AI Analysis:")
            st.write(reply)
