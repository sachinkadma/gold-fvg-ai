# Institutional XAUUSD ICT Quantitative Strategy Engine

A rigorous, causal algorithmic quantitative trading architecture for Gold (XAUUSD) on the 15-Minute timeframe. Built with zero look-ahead bias, realistic institutional friction ($0.35 spread, $0.10 slippage, $7/lot commission), and verified byte-for-byte reproducibility.

---

## 🏛️ Repository Architecture

```text
.
├── core/                           # Deterministic core strategy & backtesting engine
│   ├── config.py                   # Strategy parameters, costs, and session windows
│   ├── data_loader.py              # Schema normalization & DST-aware timezone handling
│   ├── signals.py                  # Causal 5-bar swing detection, MSS, and FVG triggers
│   ├── backtester.py               # Bar-by-bar causal simulation with conservative fill/SL rules
│   ├── metrics.py                  # Institutional performance metrics, CI, and equity curve
│   └── strategy.py                 # Core strategy abstraction
│
├── research/                       # Research, diagnostics & experiment tracking
│   ├── experiments.csv             # Pre-registered trials log (E1, E2, E3, E4)
│   └── meta_labeling/              # Causal feature generation & ML meta-labeling pipeline
│       ├── pipeline_metalabeling.py # LightGBM causal meta-labeling classifier
│       ├── trades.csv              # Stored benchmark baseline trades (684 trades)
│       └── metrics.json            # Baseline JSON metrics report
│
├── app/                            # User Interface & Analytics Web Applications
│   ├── streamlit_app.py            # Streamlit live visual co-pilot terminal
│   └── backtest_view.py            # Read-only backtest analytics tab component
│
├── vision/                         # Computer Vision / YOLO Chart Annotators
│   ├── best_v2_4class.pt           # YOLOv8 4-class bounding box weights
│   └── README.md                   # Vision audit disclaimer (annotation vs edge)
│
├── legacy/                         # Non-causal visualization & broker connectors
│   ├── deriv_client.py             # Deriv WebSocket client (marked non-causal)
│   ├── mt5_client.py               # MT5 broker bridge (graceful Linux fallback)
│   └── knowledge_store.py          # Structured ICT knowledge store
│
├── tests/                          # Automated Pytest Suite (All 6 test files)
│   ├── test_causality.py           # Strict causality & truncation invariance tests
│   ├── test_no_lookahead.py        # Random timestamp look-ahead assertion tests
│   ├── test_deterministic_engine.py# Deterministic vector math verification
│   ├── test_elliott_wave.py        # EWT cardinal rule validation tests
│   ├── test_candidate_gen.py       # Causal candidate generator tests
│   └── test_feature_gen.py         # Multi-timeframe feature generator tests
│
├── docs/                           # Methodological documentation & audit logs
│   ├── DECISIONS.md                # Quantitative design decisions & architecture
│   └── RESULTS.md                  # Baseline, Killzone, SL bucket & experiment findings
│
├── data/                           # Institutional Historical Data (Intact)
│   ├── xauusd_15m_long.csv         # Dukascopy 2015-2026 15m bid-side historical data
│   └── candidates.parquet          # Generated 24h causal candidates dataset
│
├── verify.py                       # Root verification script (hash lock & test suite)
├── STATE.md                        # Master state tracker (< 50 lines)
├── requirements.txt                # Python dependencies
└── .env.example                    # Environment variable template
```

---

## ⚡ Quickstart Commands

### 1. Verify Everything (One-Click Hash Lock Check)
Runs the entire pytest test suite (7 tests across 6 files), re-runs the 7-year baseline simulation on TRAIN (2015–2021), and verifies that `trades.csv` matches the stored SHA256 byte-for-byte:
```bash
python verify.py
# Outputs: PASS
```

### 2. Run the Causal Backtester CLI
Execute the baseline strategy over the TRAIN split with full institutional execution friction:
```bash
python core/backtester.py --data data/xauusd_15m_long.csv --data-tz UTC --split train
```
*Note: The locked TEST set (2022+) requires the `--final` flag to prevent accidental snooping.*

### 3. Run the Meta-Labeling ML Pipeline
Extract 24-hour causal features at signal-bar close, train shallow LightGBM models on TRAIN (2015–2019), and evaluate on VALID (2020–2021) with zero leakage:
```bash
python research/meta_labeling/pipeline_metalabeling.py
```

### 4. Launch the Streamlit Analytics Terminal
Launch the visual trading co-pilot and backtest analytics dashboard:
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Core Performance Summary (TRAIN 2015–2021)
- **Baseline Edge**: -0.171 R/trade net expectancy (684 trades, $0.45 total round-turn friction).
- **Structural Stop Finding**: Micro-stops (< 5.0 pts) are consumed by friction; wide structural stops ($\ge 8.0$ pts) hold a verified positive edge (**+0.356 R on VALID**, 95% CI `[+0.106, +0.607]`).
- **Full Findings**: Read [`docs/RESULTS.md`](docs/RESULTS.md) and [`docs/DECISIONS.md`](docs/DECISIONS.md).
