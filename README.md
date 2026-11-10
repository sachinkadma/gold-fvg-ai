# 🧠 Gold ICT AGI - Institutional 15m Co-Pilot

An autonomous, multi-agent AI quantitative trading terminal for **Gold (XAU/USD / frxXAUUSD)** on the **15-Minute (15m)** timeframe, powered by **Frontier Gen AI (Gemini 2.0 Flash Thinking)**, **Deriv Public WebSocket Gateway**, **Deterministic Elliott Wave Engine**, and a native **Cameron4x Institutional Candlestick Visualizer**.

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

### 2. 🌊 Deterministic Elliott Wave Theory (EWT) Engine
- Pure mathematical vector calculation of the **3 Cardinal Rules of Elliott Wave**:
  - **Rule 1 (Wave 2 Retracement):** Wave 2 never breaks below Wave 0 origin ($P_2 > P_0$ in bull; $P_2 < P_0$ in bear).
  - **Rule 2 (Wave 3 Length):** Wave 3 is never the shortest wave among Waves 1, 3, and 5.
  - **Rule 3 (Wave 4 Territory Overlap):** Wave 4 never enters Wave 1 price territory ($P_4 > P_1$ in bull; $P_4 < P_1$ in bear).
- Computes Fibonacci Extension ratios ($W_3 / W_1$) and classifies cycle phases:
  - `IN_WAVE_3_EXPANSION`
  - `IN_WAVE_4_PULLBACK`
  - `IN_WAVE_5_TERMINAL_SWEEP`
  - `COMPLETED_IMPULSE_AWAITING_ABC`
  - `CARDINAL_RULE_VIOLATION` (Hard Veto Gate)

### 3. 🕯️ Native Cameron4x Candlestick Visualizer & Wave Badges
- High-resolution (940x540) dark-themed institutional candlestick charts on `#0B1120`:
  - **Candles:** Emerald Green (`#10B981`) and Coral Red (`#EF4444`) with precise wicks and bodies.
  - **🌊 Wave Badges & Rails:** Circular cyan badges with `(0)`, `(1)`, `(2)`, `(3)`, `(4)`, `(5)` labels connected by dashed trend rails.
  - **⚡ #13 Liquidity Sweep:** Horizontal cyan ray marking BSL or SSL sweeps.
  - **⚡ #4 MSS (Market Structure Shift):** Crisp white dashed line marking structural breaks.
  - **🟩 #17 Bull FVG / 🟥 #17 Bear FVG:** Translucent shaded institutional imbalance zones.
  - **🔹 #28 50% Consequent Encroachment (50% CE):** Cyan dashed midline representing the optimal limit entry.
  - **🎯 #94 Institutional Target:** Mint green rail enforcing strict 1:2.0 Risk-to-Reward ratio.
  - **🟡 Spot Price Ray:** Real-time spot price dotted line with right-axis price badge.

### 4. 🤖 4-Agent Frontier AGI Swarm (Unanimous Consensus Protocol)
- **Agent 1 (Liquidity Detective):** Scans old highs/lows purges, BSL/SSL sweeps, and retail liquidity trap engineering.
- **Agent 2 (ICT Structure Auditor):** Confirms displacement candle energy, MSS confirmation, 3-candle FVG boundaries, and 50% CE.
- **Agent 3 (Elliott Wave Specialist):** Validates 5-wave motive sequence, cardinal rules, and prohibits buying/selling Wave 5 terminal exhaustion wicks.
- **Agent 4 (Chief Risk Officer):** Enforces minimum 1:2.0 Risk-to-Reward ratio, invalidation thesis, and dynamic lot sizing.

### 5. 🛡️ Dynamic Risk Management & Lot Sizer (Topic #93 & #98)
- Inputs: Account Balance ($) and Risk Per Trade (0.5% - 3.0%).
- Dynamically computes lot size:
  $$\text{Risk Capital} = \text{Balance} \times \left(\frac{\text{Risk } \%}{100}\right)$$
  $$\text{Recommended Lots} = \frac{\text{Risk Capital}}{\text{SL Pips} \times 10}$$

### 6. 📲 Telegram Automation & 15m Candle Close Alerts
- Dispatches high-res Cameron4x chart images and formatted markdown blueprints to Telegram channels or private phones with a single click.
- Live New York session clock tracking Asian Range, London KZ, NY AM KZ, Silver Bullet, and NY PM KZ.
- Live countdown to the next 15m candle close (`:00`, `:15`, `:30`, `:45` EST) with immediate test alert trigger.

### 7. 📚 Structured Knowledge Store (108 Topics)
- Structured JSON schemas in `knowledge/` (`liquidity.json`, `structure.json`, `entry_models.json`, `risk.json`, `elliott_wave.json`).
- Temporal Killzone filtering via `knowledge_store.py`.

---

## 📂 File Architecture

| File | Description |
| :--- | :--- |
| `streamlit_app.py` | Main application: Streamlit UI, ChatGPT-style chat, 4-Agent Swarm, trade cards, Telegram integration |
| `deriv_client.py` | Deriv WebSocket client, deterministic math engine, Elliott Wave analyzer, and Cameron4x visualizer |
| `knowledge_store.py` | Structured knowledge store with temporal killzone metadata retrieval |
| `knowledge/*.json` | Structured JSON topic schemas covering liquidity, structure, entry models, risk, and Elliott Wave |
| `test_elliott_wave.py` | Automated unit tests for Elliott Wave Cardinal Rules |
| `test_deterministic_engine.py` | Automated unit tests for deterministic ICT calculations |
| `requirements.txt` | Python dependencies (`streamlit`, `websocket-client`, `opencv-python-headless`, etc.) |
| `.streamlit/config.toml` | High-contrast dark theme configuration for 100% text visibility on mobile and desktop |

---

## 🚀 Local Setup & Installation

```bash
# 1. Clone repository
git clone https://github.com/sachinkadma/gold-fvg-ai.git
cd gold-fvg-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run unit tests
python test_elliott_wave.py
python test_deterministic_engine.py

# 4. Launch web terminal
streamlit run streamlit_app.py
```
