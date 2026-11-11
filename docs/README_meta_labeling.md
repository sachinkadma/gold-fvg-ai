# Institutional Meta-Labeling Pipeline (XAUUSD 15m)

This pipeline implements a leak-free, causal meta-labeling system (advancing the base ICT mechanical strategy) using LightGBM on multi-year institutional Gold data.

---

## 1. Architecture Overview

```
Dukascopy 15m (UTC -> NY Time)
               │
               ▼
   [Causal Setup Generator]
   (Liquidity Sweep + MSS + 50% CE FVG across all hours)
               │
               ▼
   [Forward Multi-RR Simulator]
   (Conservative limit fill, same-candle SL-first, spread + slippage + commission)
               │
               ▼
   data/candidates.parquet (3,729 records with 14 features + multi-RR labels)
               │
               ▼
   [LightGBM Meta-Classifier]
   (TRAIN 2015-2019, TimeSeriesSplit CV threshold optimization, early stopping on VALID)
               │
               ▼
   Evaluation on VALID (2020-2021) | Shuffled-Label Zero-Leakage Audit
```

---

## 2. Dataset Schema (`data/candidates.parquet`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `candidate_id` | int | Unique sequential candidate index |
| `signal_time` | datetime | Exact close timestamp of the signal candle (America/New_York) |
| `year` | int | Calendar year |
| `direction` | int | Trade direction: 1 = BUY, 0 = SELL |
| `hour` | int | New York hour (0–23) |
| `day_of_week` | int | Day of week (0 = Monday, 6 = Sunday) |
| `killzone` | int | Session regime: 1 = London, 2 = NY AM, 0 = Other |
| `atr_14` | float | 14-period Average True Range on 15m bars |
| `stop_dist` | float | Absolute stop distance in price units (`abs(entry - sl)`) |
| `stop_dist_atr` | float | Normalized stop distance (`stop_dist / atr_14`) |
| `dist_to_swing_high_atr` | float | Distance from signal close to nearest confirmed swing high / ATR |
| `dist_to_swing_low_atr` | float | Distance from signal close to nearest confirmed swing low / ATR |
| `position_in_session_range`| float | Relative position in current day's range (0.0 = low, 1.0 = high) |
| `prev_day_range` | float | Total price range of the prior trading day |
| `trend_1h_sign` | float | Slope sign of 1H EMA(20) on closed 1H bars (+1 or -1) |
| `trend_daily_sign` | float | Slope sign of Daily EMA(20) on closed Daily bars (+1 or -1) |
| `volatility_rank` | float | Percentile rank of current ATR(14) over 500 rolling bars (0.0 to 1.0) |
| `filled` | bool | Whether the pending limit order filled within 8 bars |
| `net_r_15` | float | Realized net R outcome targeting 1:1.5 R:R after costs |
| `net_r_20` | float | Realized net R outcome targeting 1:2.0 R:R after costs (baseline target) |
| `net_r_30` | float | Realized net R outcome targeting 1:3.0 R:R after costs |
| `exit_reason_20` | str | Exit trigger: `TP`, `SL`, `SL (Conservative Same-Candle)`, `UNFILLED` |

---

## 3. Strict Execution & Causality Guarantees

1. **Bid-Side Accounting**:
   - Buys enter at ask (`entry + spread`) and exit at bid.
   - Sells enter at bid (`entry`) and exit at ask (`exit + spread`).
   - Round-turn commission (\$7.00/standard lot) and stop-loss slippage (\$0.10/oz) applied to every execution.
2. **Conservative Execution**:
   - Limit orders require price to trade *through* the limit price by at least \$0.01 (mere touches never fill).
   - If price breaches stop-loss before or during fill, invalidation cancels the order.
   - If price touches both TP and SL on the same bar, SL is counted first.
3. **Anti-Leakage Safeguards**:
   - **Threshold Selection**: Picked on TRAIN (2015–2019) via 5-fold TimeSeriesSplit cross-validation. The validation set is never peeked at.
   - **Shuffled-Label Test**: Retraining on permuted labels collapses VALID expectancy to 0.000 R, proving no predictive leakage exists.
   - **TEST Split**: 2022–2026 data remains strictly locked and untouched.
