# Pre-Registered Out-of-Sample Evaluation Protocol (TEST Split: 2022–2026)

> **STATUS: LOCKED & UNRUN.**  
> Neither candidate generation nor strategy simulation has been executed on data beyond 2021-12-31. This document locks all hypotheses, execution rules, and decision thresholds prior to touching out-of-sample data.

---

## 1. Primary Model Specification (Decision Hypothesis)

The primary model tests whether institutional orderflow displacement paired with wide structural invalidation levels and session liquidity timing exhibits statistically significant positive expectancy out-of-sample.

### Core Rules
1. **Candidate Signal (Causal ICT Setup)**:
   - Swing high/low sweep of causal 5-bar swing (`SWING_LEN = 5`).
   - Displacement candle with body-to-range ratio $\ge 0.50$.
   - Fair Value Gap (FVG) creation $\ge 0.35$ pts.
   - Signal bar close confirmed prior to order placement (zero lookahead).
2. **Session Gate (Killzone Only in `America/New_York` Local Time)**:
   - Evaluated strictly in `America/New_York` local clock time, with dynamic daylight saving transitions handled via Python `zoneinfo`:
     - **London Killzone**: `02:00 <= t < 05:00` America/New_York.
     - **New York AM Killzone**: `07:00 <= t < 10:00` America/New_York.
   - Setups with signal-bar close outside these New York local hours are strictly rejected.
3. **Price-Scaled Stop Floor ($P\%$)**:
   $$\text{Stop Distance} \ge P\% \times \text{Close}_{\text{signal}}$$
   Where:
   $$P = \frac{8.0}{\text{Median}(\text{Close}_{\text{candidates 2015--2021}})} \times 100 = \frac{8.0}{1316.4990} \times 100 = 0.6077\%$$
   - Eliminates fragile micro-stops without curve-fitting to fixed dollar units across inflation regimes.
4. **Execution & Risk Parameters**:
   - **Target Risk-Reward**: Fixed $\text{RR} = 2.0$ (no trailing stops, no dynamic exits).
   - **Limit Order Expiry**: 8 bars (120 minutes) on 15m timeframe.
   - **Execution Friction (Standard)**:
     - Spread: 0.35 pts ($35.00/lot). Buy fills at Ask ($\text{Bid} + 0.35$), Sell fills at Bid.
     - Slippage: 0.10 pts ($10.00/lot) on both entry and exit.
     - Commission: $7.00 per round lot ($0.07 pts equivalent).
   - **Intra-Bar Collision Policy**: Conservative SL-first priority (same-candle high/low touching both TP and SL is scored strictly as -1.0 R loss).

---

## 2. In-Sample Benchmark Reference (2015–2021)

| Split | Trades ($N$) | Win Rate | Net Expectancy (R) | 95% Monthly Block CI | 99.5% Monthly Block CI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN (2015–2019)** | 36 | 52.8% | +0.534 R | [-0.020 R, +0.954 R] | [-0.304 R, +1.087 R] |
| **VALID (2020–2021)** | 26 | 42.3% | +0.233 R | [-0.435 R, +1.003 R] | [-0.632 R, +1.338 R] |
| **COMBINED (2015–2021)**| 62 | 48.4% | +0.408 R | [+0.007 R, +0.824 R] | [-0.215 R, +0.985 R] |

*Robustness checks on Combined: Leave-2020-Out = +0.385 R ($N=44$); Leave-2016-Out = +0.321 R ($N=44$).*

---

## 3. Decision Rule & Verification Criteria

Upon unlocking the TEST partition (2022-01-01 to latest), the outcome will be judged according to the following mutually exclusive categories:

### A. PASS (Hypothesis Accepted)
Must meet all three conditions:
1. $N \ge 60$ filled trades on TEST.
2. $\text{Net Expectancy} > 0.000\text{ R}$.
3. Monthly-block 95% bootstrap CI lower bound $> 0.000\text{ R}$ (2,000 resamples).

### B. FAIL (Hypothesis Rejected)
- $\text{Net Expectancy} \le 0.000\text{ R}$ (with $N \ge 30$).

### C. INCONCLUSIVE (Forward Paper-Trading Required)
- Everything else:
  - $\text{Net Expectancy} > 0.000\text{ R}$ but 95% CI includes zero ($\text{CI}_{\text{lower}} \le 0.000\text{ R}$), OR
  - $N < 60$ executed trades.

### Strict Governance & Capital Protection
- **No Live Money**: No capital allocation under any outcome.
- **Action on INCONCLUSIVE**: Forward paper-trading of the unchanged, frozen rule.
- **One Run Only**: Exactly one evaluation run permitted on the TEST partition.
- **Zero Re-tuning**: No post-test adjustments to parameters, thresholds, or execution rules after observing TEST results.

---

## 4. Secondary Diagnostic Track (Informational Only)

- **Fixed Floor Hypothesis (`SL >= 8.0 pts`)**:
  - Killzone-only fixed 8.0-point floor will be evaluated concurrently.
  - Results will be reported for comparison to determine whether fixed dollar stops degrade under $2,000–$2,800 Gold spot prices.
  - **Notice**: The fixed 8.0-point model is strictly informational and does **not** govern the PASS / FAIL / INCONCLUSIVE gate. The secondary benchmark (fixed stop >= 8.0 price units) is informational only. It must never change, soften, or override the verdict from the primary rule, in any outcome.

---

## 5. Execution Command

When authorized by the user, the single out-of-sample evaluation must be executed via:

```bash
python research/meta_labeling/run_prereg_test.py --split test --p-scaled 0.6077 --final
```
