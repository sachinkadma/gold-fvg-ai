# 🧠 Gold ICT AGI - Institutional 15m Co-Pilot

An autonomous, multi-agent AI quantitative trading terminal for **Gold (XAU/USD / frxXAUUSD)** on the **15-Minute (15m)** timeframe, powered by **Frontier Gen AI (Gemini 2.0 Flash Thinking)**, **Deriv Public WebSocket Gateway**, and a native **Cameron4x Institutional Candlestick Chart Engine**.

---

## 🌐 Live Deployments & Links
- **🚀 Live Web Terminal:** [gold-fvg-ai-ahwmxvqf93rfjaqfkhxxyr.streamlit.app](https://gold-fvg-ai-ahwmxvqf93rfjaqfkhxxyr.streamlit.app)
- **📦 GitHub Repository:** [github.com/sachinkadma/gold-fvg-ai](https://github.com/sachinkadma/gold-fvg-ai)

---

## ⚡ Key Highlights & Capabilities

### 1. 📡 Direct Deriv WebSocket Spot Feed (`frxXAUUSD`)
- Direct live candlestick streaming from **Deriv Public WebSocket Gateway** (`wss://ws.derivws.com/websockets/v3?app_id=1089`) using public test App ID `1089` (no broker account or API token required).
- Resilient secondary REST spot Gold failover (`GC=F`) for 100% uptime with zero downtime.
- **1-Click Live Scan:** Users tap `⚡ Scan Live 15m Gold (Deriv Spot)` to instantly fetch the live 15m candle array, analyze market structure, and generate institutional execution blueprints.

### 2. 🕯️ Native Cameron4x Candlestick Chart Engine
- Eliminates messy phone screenshots, status bars, and misaligned boxes.
- Renders high-resolution (940x540) dark-themed institutional candlestick charts on `#0B1120`:
  - **Candles:** Emerald Green (`#10B981`) and Coral Red (`#EF4444`) with precise wicks and bodies.
  - **⚡ #13 Liquidity Sweep:** Horizontal cyan ray marking BSL or SSL sweeps.
  - **⚡ #4 MSS (Market Structure Shift):** Crisp white dashed line marking structural breaks.
  - **🟩 #17 Bull FVG / 🟥 #17 Bear FVG:** Translucent shaded institutional imbalance zones.
  - **🔹 #28 50% Consequent Encroachment (50% CE):** Cyan dashed midline representing the optimal limit entry.
  - **🎯 #94 Institutional Target:** Mint green rail enforcing strict 1:2.0 Risk-to-Reward ratio.
  - **🟡 Spot Price Ray:** Real-time spot price dotted line with right-axis price badge.

### 3. 🤖 3-Agent Frontier AGI Swarm
- Powered by Google Gemini Frontier models (auto-discovering active keys for `gemini-2.0-flash-thinking`, `gemini-2.5-flash`, `gemini-2.0-flash`).
- **Agent 1 (Liquidity Detective):** Scans old highs/lows purges, BSL/SSL sweeps, and retail liquidity trap engineering.
- **Agent 2 (ICT Structure Auditor):** Confirms displacement candle energy, MSS confirmation, 3-candle FVG boundaries, and 50% CE.
- **Agent 3 (Chief Risk Officer):** Enforces minimum 1:2.0 Risk-to-Reward ratio, invalidation thesis, and dynamic lot sizing.

### 4. 🛡️ Dynamic Risk Management & Lot Sizer (Topic #93 & #98)
- Inputs: Account Balance ($) and Risk Per Trade (0.5% - 3.0%).
- Automatically calculates:
  $$\text{Risk Capital} = \text{Balance} \times \left(\frac{\text{Risk } \%}{100}\right)$$
  $$\text{Recommended Lots} = \frac{\text{Risk Capital}}{\text{SL Pips} \times 10}$$

### 5. 📲 Telegram Automation & 15m Candle Close Alerts
- Dispatches high-res Cameron4x chart images and formatted markdown blueprints to Telegram channels or private phones with a single click.
- Live New York session clock tracking Asian Range, London KZ, NY AM KZ, Silver Bullet, and NY PM KZ.
- Live countdown to the next 15m candle close (`:00`, `:15`, `:30`, `:45` EST) with immediate test alert trigger.

### 6. 📚 104-Topics ICT Syllabus RAG Library
- Full syllabus covering ICT Core Content and Mentorship models.
- Semantic vector embeddings via `all-MiniLM-L6-v2` searching `notes.txt` for exact strategy rules.

---

## 📂 File Architecture

| File | Description |
| :--- | :--- |
| `streamlit_app.py` | Main application: Streamlit UI, ChatGPT-style chat, 3-Agent Swarm, trade cards, Telegram integration |
| `deriv_client.py` | Deriv WebSocket client, live 15m candle fetching, failover engine, and native Cameron4x chart renderer |
| `notes.txt` | Institutional knowledge base of ICT rules, syllabus topics, and algorithmic confirmations |
| `requirements.txt` | Python dependencies (`streamlit`, `websocket-client`, `opencv-python-headless`, `sentence-transformers`, etc.) |
| `packages.txt` | System dependencies for Linux containers (e.g. `libgl1` for OpenCV) |
| `.streamlit/config.toml` | High-contrast dark theme configuration for 100% text visibility on mobile and desktop |

---

## 🚀 Local Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/sachinkadma/gold-fvg-ai.git
cd gold-fvg-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application
streamlit run streamlit_app.py
```

### Environment / Secrets (Optional)
In `.streamlit/secrets.toml` or Streamlit Cloud Settings:
```toml
GEMINI_API_KEY = "AIzaSy..."
TELEGRAM_BOT_TOKEN = "789...:ABC..."
TELEGRAM_CHAT_ID = "12345678"
```
*(All keys can also be entered directly in the web app sidebar at runtime).*
