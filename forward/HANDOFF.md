# Forward Paper-Trading System Handoff

## 1. Frozen Constraints
- **Frozen Strategy Code**: `core/candidate_generator.py`, `core/backtester.py`, `core/signals.py`, `core/config.py`.
- **Frozen Parameters**: $RR = 2.0$, $P\_SCALED = 0.6077$, killzones, entry/SL/TP rules, 8-bar expiry, SL-first tie.
- **Burned Split**: TEST (2022-2026) is burned. `run_prereg_test.py` and `test_runs.log` one-shot locks are permanent.
- **Tuning/Filtering**: No new filters, no parameter tuning. Long/short is strictly observation-only ("NOT FOR DECISIONS").

## 2. Current State
- **Branch / Commit**: `reorg` (Phase D complete).
- **Tags**: `forward-v1`..`forward-v7`, `prereg-v1`..`prereg-v4`. Tags pushed to `origin`.
- **Pytest**: 57 passed in test suite (Exit Code: 0).
- **Verify.py**: PASS (Exit Code: 0) — SHA256 integrity verified against baseline `trades.csv`.

## 3. Architecture & File Inventory
- `forward/__init__.py`: Package init.
- `forward/logger.py`: Immutable append-only event hash chain (`GENESIS`, `SIGNAL`, `SPREAD_OBS`, `SKIPPED`, `FILL`, `EXPIRE`, `CLOSE`, `GAP`).
  - Key functions: `ForwardSignalLogger`, `verify_chain`, `verify_against_anchor`, `convert_mt5_time_to_utc`, `push_external_anchor_to_git`.
  - Storage: Append-only JSONL (`forward/signals.jsonl`).
  - Companion Files: `[log]_buffer.json` (buffer snapshot), `[log]_anchor_queue.json` (un-pushed retry queue).
  - External Anchoring: Head hash committed via git plumbing to dedicated orphan branch `forward-anchors` on remote `origin`.
- `forward/execution_tracker.py`: Realized spread observation logger and reporting engine (`SPREAD_OBS`).
  - Key functions: `ExecutionTracker`, `generate_spread_report`.
  - `source` field: strictly `"TICK"` or `"BAR"`. Units: `price_points_usd`.
  - Metrics: Realized spread vs assumed 0.35 pts (mean, median, p95, worst), split by London / NY AM. Report ignores GAP records.
  - Contract: Strictly report-only; frozen cost model and net R are unaffected; no slippage (not measurable in paper mode).
- `forward/monitor.py`: Read-only forward monitor and PREREG Section 5 gate evaluator.
  - Key functions: `ForwardMonitor`, `block_bootstrap_ci`, `send_telegram_alert`, `alert_new_signal`, `alert_trade_closed`.
  - Gates: Deterministic latch at $N=12$ (exp $\le 0 \implies \text{FAIL\_FWD}$; never re-evaluated at $N=13..19$); End at 20th trade OR 30 months; PASS_FWD if $N=20$, $\text{exp} > 0$ and $\text{ci\_lower} > 0$; FAIL_FWD if $N \ge 12$ and $\text{exp} \le 0$; otherwise `INCONCLUSIVE-FINAL`. Any trade beyond 20th is ignored.
  - Directional breakdown: labeled `[OBSERVATION ONLY - NOT FOR DECISIONS]`.
  - Chain integrity: halts with `CHAIN_INVALID` if `verify_chain()` fails.
- `forward/PROTOCOL_NOTES.md`: Protocol audit notes covering TEST log date (`2026-11-11`), start date (`2026-11-12`), 10.7h gap, 30m end boundary (`2029-05-12`), and gap sample exclusion.
- `forward/run_forward.py`: Live/dry-run runner orchestrating MT5 closed 15m polling, deduplication, downtime recovery, and restart state reconstruction.
- `tests/test_forward_logger.py`: 25 forward logger test cases (tamper detection, truncation, restart recovery, parity, warm-up suppression).
- `tests/test_execution_tracker.py`: 5 execution tracker test cases (live tick vs bar spread, chain validity, GAP filtering, session metrics, SPREAD_OBS neutrality on N/WR/EXP).
- `tests/test_forward_monitor.py`: 15 monitor test cases (bootstrap equality, synthetic smoke test, CHAIN_INVALID tamper halt, Telegram safe failure, and 11 gate boundary scenarios).
- `tests/test_candidate_gen.py`: Includes golden-file test on fixed TRAIN slice (500 bars) with SHA256 integrity hash verification.

## 4. Verified Decisions
- **Broker Timezone**: MetaQuotes-Demo runs on Cyprus EU DST (`server_tz="EU_EET"`, UTC+2 standard). Empirically verified against Dukascopy UTC: EU offset MAE 0.13–0.34 pts (92–100% matches); US offset (NY-7h) failed with MAE 6–18 pts (1–7% matches).
- **Warm-Up Buffer**: 135 bars (`MIN_BUFFER_WARMUP = 135`) safe minimum. Signal emission strictly suppressed during warmup.
- **Parity Protocol**: Trade outcomes strictly verified against `research.meta_labeling.pipeline_metalabeling:simulate_candidate_forward` with 100% exact `net_r` match.
- **Startup Date Contract**: `FORWARD_START_DATE` is mandatory at startup (e.g. `2026-11-12T00:00:00Z`). Logger refuses start if missing or stale (>24h past). Written to `GENESIS`.
- **Branch Protection**: Remote `forward-anchors` branch must have force-push disabled (`--force` prohibited).
- **Gate Evaluation**: Evaluated ONLY at $N=12$ and at end ($N=20$ or 30 months). First 20 trades only. Once terminal, never changes.

## 5. Open Items (Next Steps)
- None. Forward paper-trading infrastructure (Phases 1, A, B, C, D) is complete and verified. Awaiting deployment authorization.

## 6. Exact Operational Commands

```powershell
# Run full test suite (redirected output)
pytest > out_pytest.txt 2>&1; $LASTEXITCODE

# Run integrity verification
python verify.py > out_verify.txt 2>&1; $LASTEXITCODE

# 3-Day Dry-Run Replay from Live MT5 (no live orders)
python forward/run_forward.py --forward-start-date "2026-11-12T00:00:00Z" --dry-run-days 3 --log-path forward/dry_run.jsonl

# Start Live Forward Runner
python forward/run_forward.py --forward-start-date "2026-11-12T00:00:00Z" --log-path forward/signals.jsonl

# Run Forward Monitor & Gate Evaluator
python forward/monitor.py --log forward/signals.jsonl --status forward/forward_status.json

# Verify Hash Chain Locally
python -c "from forward.logger import verify_chain; print(verify_chain('forward/signals.jsonl'))"

# Verify Hash Chain Against External Anchors (Detects Truncation/Tampering)
python -c "from forward.logger import verify_against_anchor; print(verify_against_anchor('forward/signals.jsonl', 'forward/anchors.json'))"
```
