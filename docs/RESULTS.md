# Quantitative Research & Diagnostic Results

## 1. Verified Baseline Benchmark (TRAIN 2015–2021)
Evaluated on Dukascopy 15m Gold data (UTC -> America/New_York DST-aware) with full realistic costs ($0.35 spread, $0.10 slippage, $7.0/lot commission):

| Metric | Value |
| :--- | :--- |
| **Total Trades** | 684 |
| **Win Rate** | 31.0% (212 Wins / 472 Losses) |
| **Gross Expectancy** | -0.070 R |
| **Net Expectancy (after costs)** | **-0.171 R** |
| **Cost Drag** | 0.101 R per trade (~59% of loss) |
| **95% Bootstrap CI** | [-0.276 R, -0.067 R] |
| **Profit Factor** | 0.78 |
| **Max Drawdown (R)** | 122.51 R (72.52%) |
| **Buy-and-Hold Gold Return** | +54.56% ($1,183.41 -> $1,829.06) |

---

## 2. Killzone & Session Timing Anomaly
Comparison between trades inside defined Killzones vs outside on VALID (2020–2021):
- **Inside Killzones** (138 trades): 37.0% WR | +0.036 R expectancy | +4.94 R net | PF 1.05
- **Outside Killzones** (205 trades): 41.5% WR | **+0.169 R expectancy** | **+34.63 R net** | **PF 1.27**

### Hour-by-Hour Key Discoveries
- **Strongest Performance**: 14:00 UTC (+0.241 R, +19.79 R net), 16:00 UTC (+0.347 R, +12.50 R net), 09:00 UTC (+0.208 R, +9.56 R net). Late NY morning and transition hours provide clean momentum fills without whip.
- **Worst Performance**: 01:00 UTC (-0.566 R, -28.30 R net, 18.0% WR) and 08:00 UTC (-0.196 R, -8.43 R net). Thin Asian hours and the London opening chop produce heavy whipsaws.

---

## 3. Stop Distance Distribution & Structural Edge

### Percentile Distribution
| Percentile | Baseline `trades.csv` | Candidates `candidates.parquet` | Multiple of ATR(14) |
| :--- | :---: | :---: | :---: |
| **Min** | 3.00 pts | 3.00 pts | ~0.95x |
| **25%** | 3.95 pts | 4.15 pts | 2.50x - 2.75x |
| **Median (50%)** | **4.98 pts** | **5.46 pts** | **3.23x - 3.54x** |
| **75%** | 6.58 pts | 7.37 pts | 4.16x - 4.63x |
| **Max** | 22.41 pts | 40.93 pts | 9.90x - 25.83x |

*Note: Since 15m ATR on Gold averages $1.50–$1.85, a $3.00 hard floor stop is already 1.9x to 3.5x ATR. Pre-registered E1 (`SL >= 1.0 * ATR`) only filtered 1 trade out of 343.*

### Stop Bucket Performance Breakdown
| Bucket | Filter | TRAIN (2015–2019) Net Exp | VALID (2020–2021) Net Exp | VALID 95% CI | VALID Profit Factor |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Bucket A** | `3.0 <= SL < 5.0 pts` | -0.144 R | -0.073 R | [-0.358, +0.243] | 0.90 |
| **Bucket B** | `5.0 <= SL < 8.0 pts` | -0.063 R | -0.012 R | [-0.255, +0.232] | 0.98 |
| **Bucket C** | **`SL >= 8.0 pts`** | **+0.092 R** | **+0.356 R** | **[+0.106, +0.607]** | **1.64** |

**Conclusion**: Micro-stops (< 5.0 pts) are consumed by execution drag ($0.45 total round-turn friction). Wide structural stops ($\ge 8.0$ pts) hold a statistically verified positive edge (+0.356 R net expectancy, 95% CI strictly positive).

---

## 4. Pre-Registered Experiment Summary (`research/experiments.csv`)

| Trial | Experiment | Filter Rule | VALID Trades | VALID WR% | VALID Exp (R) | VALID Net R | Profit Factor |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | E1_Stop_Dist_ATR_Ge_1 | $\text{SL} \ge 1.0 \times \text{ATR}(14)$ | 342 / 1,119 | 39.8% | +0.119 R | +40.72 R | 1.20 |
| **2** | E2_E1_Plus_16Bar_TimeStop | E1 + 16-bar time stop | 342 / 1,119 | 47.1% | +0.003 R | +0.98 R | 1.01 |
| **3** | E3_Stop_Dist_Ge_5pts | $\text{SL} \ge 5.0\text{ pts}$ (Buckets B+C) | 257 / 1,119 | 41.2% | +0.178 R | +45.84 R | 1.29 |
| **4** | E4_Stop_Dist_Ge_8pts | $\text{SL} \ge 8.0\text{ pts}$ (Bucket C) | 133 / 1,119 | 46.6% | **+0.356 R** | **+47.32 R** | **1.64** |
