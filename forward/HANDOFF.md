# Forward Paper-Trading System Handoff

## 1. Frozen Constraints
- **Frozen Strategy Code**: `core/candidate_generator.py`, `core/backtester.py`, `core/signals.py`, `core/config.py`.
- **Frozen Parameters**: $RR = 2.0$, $P\_SCALED = 0.6077$, killzones, entry/SL/TP rules, 8-bar expiry, SL-first tie.
- **Burned Split**: TEST (2022-2026) is burned. `run_prereg_test.py` and `test_runs.log` one-shot locks are permanent.
- **Tuning/Filtering**: No new filters, no parameter tuning. Long/short is strictly observation-only ("NOT FOR DECISIONS").

## 2. Current State
- **Branch / Commit**: `reorg` (Pre-rehearsal cleanup complete).
- **Tags**: `forward-v1`..`forward-v10`, `prereg-v1`..`prereg-v4`. Tags pushed to `origin`.
- **Pytest**: 64 passed in test suite (Exit Code: 0).
- **Verify.py**: PASS (Exit Code: 0) — SHA256 integrity verified against baseline `trades.csv`.

## 3. Architecture & File Inventory
- `forward/__init__.py`: Package init.
- `forward/logger.py`: Immutable append-only event hash chain. Supports `rehearsal=True` isolation (strictly prohibits writing to `signals.jsonl` or pushing to `forward-anchors`).
  - Key functions: `ForwardSignalLogger`, `verify_chain`, `verify_against_anchor`, `convert_mt5_time_to_utc`, `push_external_anchor_to_git`.
- `forward/execution_tracker.py`: Realized spread observation; `source` = `"TICK"` or `"BAR"`. Units: `price_points_usd`.
- `forward/monitor.py`: Read-only PREREG Section 5 gate evaluator. 30-month limit uses `forward_end_date_30m` from GENESIS (calendar-exact). Removes 913.125-day approximation.
- `forward/PROTOCOL_NOTES.md`: Protocol audit — TEST log `2026-11-11T13:18:05Z`, start date `2026-11-12`, gap 10.7h, 30m end `2029-05-12T00:00:00Z`.
- `forward/run_forward.py`: Live/dry-run/rehearsal runner (MT5, deduplication, state recovery, gap detection). `--rehearsal` flag isolates log and anchor branch.
- `forward/launch/run_forever.ps1`: Windows supervisor — infinite restart loop, 10s backoff, Telegram alert on startup refusal, hard-stop after 3 consecutive refusals.
- `forward/launch/heartbeat.py`: Separate read-only process — daily alive Telegram, STALE_FEED alert (>45min during open market hours only), PC time-jump detection.
- `forward/launch/rehearsal_check.py`: Read-only audit tool verifying chain validity, external anchors, event counts, filter status, gap breakdown, duplicate timestamps, and missing candles.
- `forward/launch/launch_checklist.md`: Pre-launch steps (clone, env vars, MT5, remote, 48–72h rehearsal, protocol notes update before first run, commit).
- `forward/launch/README_WINDOWS.md`: Full Windows setup — fresh clone, env vars, Task Scheduler, power plan, rehearsal drills, weekend routine.

## 4. Verified Decisions
- **Broker Timezone**: EU_EET (Cyprus DST). Dukascopy alignment verified; NY-7h rule rejected.
- **Warm-Up Buffer**: 135 bars. Signal emission suppressed during warmup.
- **Parity Protocol**: 100% exact `net_r` match vs `simulate_candidate_forward`.
- **Gate Evaluation**: Latched at $N=12$ only; end at $N=20$ or exact calendar 30 months. 21st+ trade ignored. Terminal once final verdict reached.
- **30-Month End**: Calendar-exact (`pd.DateOffset(months=30)`) stored in GENESIS as `forward_end_date_30m`. `2026-11-12` → `2029-05-12T00:00:00Z`.
- **Stale Feed Rule**: >45 min without new closed candle during open market hours only. Weekend (Fri 17:00 to Sun 18:00 NY) and daily break (Mon-Thu 17:00–18:00 NY) suppress alert.

## 5. Open Items
- **Rehearsal Redo**: Rehearsal must be redone on tag `forward-v12` to verify that external anchors commit with clean filenames (no Windows CRLF trailing `\r`).
- **Clean Anchor Branch**: The real `forward-anchors` branch must be clean (or recreated) before production launch.
- **Do not start the real forward run** until authorized.

## 6. Exact Operational Commands

```powershell
# Full verification
python verify.py > out/verify.txt 2>&1; Add-Content out/verify.txt "EXIT=$LASTEXITCODE"

# 48-72h rehearsal (isolated mode: rehearsal.jsonl and forward-anchors-rehearsal branch)
python forward/run_forward.py --forward-start-date "2026-11-12T00:00:00Z" --rehearsal

# Audit rehearsal run
python forward/launch/rehearsal_check.py --log forward/rehearsal.jsonl --anchor-ledger forward/rehearsal_anchors.json

# Production (after checklist complete, protocol notes committed)
.\forward\launch\run_forever.ps1   # supervisor (restarts automatically)
python forward/launch/heartbeat.py  # separate window

# Monitor gate status
python forward/monitor.py --log forward/signals.jsonl --status forward/forward_status.json

# Chain integrity
python -c "from forward.logger import verify_chain; print(verify_chain('forward/signals.jsonl'))"
python -c "from forward.logger import verify_against_anchor; print(verify_against_anchor('forward/signals.jsonl', 'forward/anchors.json'))"
```
