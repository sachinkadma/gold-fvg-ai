# Quantitative Strategy Engine State

## Architecture
- **Market**: XAUUSD 15M (Dukascopy historical data, UTC -> America/New_York DST-aware).
- **Core Signal**: Causal ICT setup (Sweep of confirmed 5-bar swing -> Displacement body ratio >= 0.50 -> Unmitigated FVG >= 0.35 pts).
- **Execution**: BID-side limit order (trades through entry by 0.01 pt within 8 bars; Buy asks = entry + 0.35, Sell bids = entry). Conservative same-candle SL-first resolution. Costs: 0.35 pt spread, 0.10 pt slippage, $7.0/lot comm.
- **Partitions**: TRAIN (2015–2019), VALID (2020–2021), TEST (2022–2026, locked & unrun).

## Current Performance Metrics
| Engine / Model / Trial | Partition | Trades | Win Rate | Net Expectancy (R) | 95% Bootstrap CI | Profit Factor | Max DD (R) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Killzone)** | TRAIN (2015-2021) | 684 | 31.0% | -0.171 R | [-0.276, -0.067] | 0.78 | 122.51 R |
| **All Candidates (Ungated)** | VALID (2020-2021) | 343 | 39.7% | +0.115 R | [-0.039, +0.269] | 1.18 | 21.41 R |
| **Trial E1 (Stop >= 1.0 ATR)** | VALID (2020-2021) | 342 | 39.8% | +0.119 R | [-0.033, +0.272] | 1.20 | 21.38 R |
| **Trial E2 (E1 + 16-Bar Time Stop)** | VALID (2020-2021) | 342 | 47.1% | +0.003 R | [-0.096, +0.104] | 1.01 | 17.07 R |
| **Trial E3 (SL >= 5.0 pts, Buckets B+C)** | VALID (2020-2021) | 257 | 41.2% | +0.178 R | [-0.000, +0.365] | 1.29 | 18.89 R |
| **Trial E4 (SL >= 8.0 pts, Bucket C)** | VALID (2020-2021) | 133 | 46.6% | **+0.356 R** | **[+0.106, +0.607]** | **1.64** | **16.11 R** |
| **LightGBM Meta-Model** | VALID (2020-2021) | 86 | 40.7% | +0.058 R | [-0.170, +0.285] | 1.15 | 12.88 R |

*Buy-and-Hold Gold Benchmark (2015–2021): +54.56% ($1,183.41 -> $1,829.06).*  
*Shuffled-label validation: 0 trades taken, 0.000 R expectancy (verified 0 feature leakage).*

## Current Status & Next Steps
- **Status**: Reorganization complete on branch `reorg`. `verify.py` passes hash lock. SL-bucket diagnostic logged in `research/experiments.csv`. Streamlit honesty pass and documentation complete.
- **Next Step**: Await user directive. TEST split (2022+) remains strictly locked and unrun.
