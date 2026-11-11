# Quantitative Architecture & Design Decisions

## 1. Zero Look-Ahead Bias & Causal Signal Design
- **Deterministic 5-Bar Swings**: A swing high or low at candle index $i$ is only acknowledged and confirmed after candle $i + 5$ closes. No intermediate or unconfirmed pivots are accessed.
- **Strict Close Evaluation**: Setups are evaluated strictly at the close of candle $t$, with limit orders submitted to fill on subsequent candles $t+1 \dots t+8$.
- **Bid-Side Spread Mechanics**: Dukascopy historical data is BID-side. Buys enter at ask (`price + spread`) and exit SL at bid (`sl - slippage`). Sells enter at bid (`price`) and exit SL at ask (`sl + slippage + spread`).

## 2. Order Cancellation vs Session-Independent Persistence
- **Original Baseline (`backtester.py`)**: Required signals to occur within defined Killzone windows (London 02:00–05:00 NY, NY AM 07:00–10:00 NY). Crucially, pending limit orders were automatically canceled if the session ended before filling (`is_in_killzone(cur_time.time()) != p_kz`).
- **Meta-Labeling Pipeline (`candidates.parquet`)**: Allows setups across all 24 hours (`killzone_gate=False`) and permits limit orders to persist for up to 8 bars regardless of session boundaries.
- **Decision**: Preserved the original conservative cancellation behavior in `core/backtester.py` as the benchmark baseline, while logging 24-hour hour-by-hour features in `research/meta_labeling/` to expose session dynamics.

## 3. Pre-Registered Experiments Protocol
- **Anti-Overfitting Discipline**: The TEST split (2022–2026) is strictly locked and unrun. All diagnostic iterations are restricted to in-sample TRAIN (2015–2019) and out-of-sample exploration VALID (2020–2021).
- **Trial Logging**: Every pre-registered test (E1: ATR stop floor, E2: 16-bar time stop, E3/E4: structural stop distance buckets) is logged sequentially in `research/experiments.csv` with zero post-hoc parameter tweaking.

## 4. Separation of Legacy / Visual Layers from Core Engine
- Legacy visualization utilities, MT5 live connectors, and Deriv WebSockets are isolated in `legacy/` and explicitly marked non-causal.
- The Streamlit interface in `app/` is strictly decoupled as a read-only reporting and visual aid component, featuring persistent truth-in-advertising banners.
