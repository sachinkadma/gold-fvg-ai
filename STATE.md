# Quantitative Strategy Engine State

## Architecture
- **Market**: XAUUSD 15M (Dukascopy historical data + live MetaTrader 5 broker feed, Cyprus EU DST aware).
- **Core Signal**: Causal ICT setup (Sweep of confirmed 5-bar swing -> Displacement body ratio >= 0.50 -> Unmitigated FVG >= 0.35 pts). Strategy code is completely frozen.
- **Execution**: BID-side limit order (trades through entry by 0.01 pt within 8 bars; Buy asks = entry + 0.35, Sell bids = entry). Conservative same-candle SL-first resolution. Costs: 0.35 pt spread, 0.10 pt slippage, $7.0/lot comm.
- **Partitions**: TRAIN (2015–2019), VALID (2020–2021), TEST (2022–2026, burned).
- **Forward Stage**: PREREG Section 5 forward paper-trading stage ($N\_forward = 20$ executed trades OR 30 months).

## Performance Reference
| Engine / Model / Trial | Partition | Trades | Win Rate | Net Expectancy (R) | 95% Bootstrap CI | Profit Factor | Max DD (R) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN Baseline** | 2015–2019 | 36 | 52.8% | +0.534 R | [-0.020, +0.954] | 1.88 | - |
| **VALID Baseline** | 2020–2021 | 26 | 42.3% | +0.233 R | [-0.435, +1.003] | 1.34 | - |
| **TEST Split (Burned)** | 2022–2026 | 49 | 51.0% | +0.510 R | [+0.147, +0.829] | 1.76 | - |

*Note: TEST split evaluated on 2026-11-11. Verdict: INCONCLUSIVE ($N=49 < 60$ required). Forward paper-trading protocol triggered per PREREG Section 5.*

## Forward Paper-Trading Infrastructure
- **Modules**:
  - `forward/logger.py`: Immutable, append-only cryptographic hash chain logging (`GENESIS`, `SIGNAL`, `SPREAD_OBS`, `SKIPPED`, `FILL`, `EXPIRE`, `CLOSE`, `GAP`). Remote anchoring to `forward-anchors` branch.
  - `forward/execution_tracker.py`: Real-time MT5 tick/bar spread tracker (`SPREAD_OBS`). Realized spread observation vs 0.35 pts assumed.
  - `forward/monitor.py`: Read-only PREREG Section 5 gate evaluator ($N=12$ latched early-fail, $N=20$ PASS_FWD, 30m limit) with monthly bootstrap CI and Telegram alerting.
  - `forward/run_forward.py`: Live/dry-run polling daemon with state recovery, downtime gap detection, and deduplication.
  - `forward/PROTOCOL_NOTES.md`: Exact audit of TEST log timestamp, forward start date (`2026-11-12`), and 30-month calendar boundary (`2029-05-12`).
- **Release Tags**: `forward-v1` .. `forward-v7` (all pushed to remote).

## Operational Commands
```powershell
# 1. Run Complete Verification Suite
python verify.py > out/verify.txt 2>&1; $LASTEXITCODE

# 2. 3-Day Dry-Run Replay from Live MT5
python forward/run_forward.py --forward-start-date "2026-11-12T00:00:00Z" --dry-run-days 3 --log-path forward/dry_run.jsonl

# 3. Live Forward Paper Runner
python forward/run_forward.py --forward-start-date "2026-11-12T00:00:00Z" --log-path forward/signals.jsonl

# 4. Forward Gate Monitor
python forward/monitor.py --log forward/signals.jsonl --status forward/forward_status.json
```
