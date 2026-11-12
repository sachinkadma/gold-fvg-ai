# Forward Paper-Trading System Handoff

## 1. Frozen Constraints
- **Frozen Strategy Code**: `core/candidate_generator.py`, `core/backtester.py`, `core/signals.py`, `core/config.py`.
- **Frozen Parameters**: $RR = 2.0$, $P\_SCALED = 0.6077$, killzones, entry/SL/TP rules, 8-bar expiry, SL-first tie.
- **Burned Split**: TEST (2022-2026) is burned. `run_prereg_test.py` and `test_runs.log` one-shot locks are permanent.
- **Tuning/Filtering**: No new filters, no parameter tuning. Long/short is strictly observation-only ("NOT FOR DECISIONS").

## 2. Current State
- **Branch / Commit**: `reorg` @ `9286d8b` (clean working tree).
- **Tags**: `forward-v1` (`85ed7f3`), `forward-v2` (`9286d8b`), `prereg-v1`..`prereg-v4`. Tags pushed to `origin`.
- **Pytest**: 32 passed in 232.69s (Exit Code: 0).
- **Verify.py**: PASS (Exit Code: 0) — SHA256 integrity verified against baseline `trades.csv`.

## 3. Architecture & File Inventory
- `forward/__init__.py`: Package init.
- `forward/logger.py`: Immutable append-only event hash chain (`GENESIS`, `SIGNAL`, `SKIPPED`, `FILL`, `EXPIRE`, `CLOSE`, `GAP`).
  - Key functions: `ForwardSignalLogger`, `verify_chain`, `verify_against_anchor`, `convert_mt5_time_to_utc`, `push_external_anchor_to_git`.
  - Storage: Append-only JSONL (`forward/signals.jsonl`).
  - Companion Files: `[log]_buffer.json` (buffer snapshot), `[log]_anchor_queue.json` (un-pushed retry queue).
  - External Anchoring: Head hash committed via git plumbing to dedicated orphan branch `forward-anchors` on remote `origin`.
- `forward/run_forward.py`: Live/dry-run runner orchestrating MT5 closed 15m polling, deduplication, downtime recovery, and restart state reconstruction.
- `tests/test_forward_logger.py`: 24 forward test cases (tamper detection, truncation, restart recovery, parity, warm-up suppression).

## 4. Verified Decisions
- **Broker Timezone**: MetaQuotes-Demo runs on Cyprus EU DST (`server_tz="EU_EET"`, UTC+2 standard). Empirically verified against Dukascopy UTC: EU offset MAE 0.13–0.34 pts (92–100% matches); US offset (NY-7h) failed with MAE 6–18 pts (1–7% matches).
- **Warm-Up Buffer**: 135 bars (`MIN_BUFFER_WARMUP = 135`) safe minimum. Signal emission strictly suppressed during warmup.
- **Parity Protocol**: Trade outcomes strictly verified against `research.meta_labeling.pipeline_metalabeling:simulate_candidate_forward` with 100% exact `net_r` match.
- **Startup Date Contract**: `FORWARD_START_DATE` is mandatory at startup (e.g. `2026-11-12T00:00:00Z`). Logger refuses start if missing or stale (>24h past). Written to `GENESIS`.
- **Branch Protection**: Remote `forward-anchors` branch must have force-push disabled (`--force` prohibited).
- **Unattended Push Auth**: For headless/daemon forward runs, authenticate via GitHub Deploy Key (read/write SSH key in `~/.ssh/id_ed25519`) or fine-grained Personal Access Token (PAT with Repository Contents: Read/Write) configured in git remote URL.

## 5. Open Items (Next Steps)
- **A1. Genesis**: auto-read git_tag/commit (no hardcoded forward-v1); add server_tz=EU_EET, MT5 server name, symbol, feed name; run_forward default EU_EET; test that 2026-03-10 00:00 server and 2026-10-27 00:00 server both map to 22:00 UTC of the previous day.
- **A2. Dry-run** (3 days live MT5, no orders): SIGNAL count by filter_status; dedupe repeated setups if found; N = PASSED_PRIMARY + FILLED + CLOSED only; label weekend market-close GAP separately from downtime GAP.
- **A3. History preload**: bars before forward_start_date only fill the buffer; no SIGNAL/FILL/CLOSE for them; test.
- **A4. Recovery**: rebuild buffer from MT5 history and compare with _buffer.json (chain is source of truth); tests for restart with PENDING order and crash after append before anchor push.
- **A5. Anchor**: show `git ls-remote origin forward-anchors`; test that unreachable remote refuses startup; document deploy key/token for unattended push. Do not change remote settings.
- **A6. Read-only**: explain how simulate_candidate_forward handled end-of-data unresolved trades in the TEST run (no changes to frozen files).
- **A7. Final foreground pytest + verify.py on the tagged commit** (redirect to file, last 15 lines + exit code).
- **B (Phase 2, after A is committed)**: execution_tracker.py. SPREAD_OBS event type in the chain, realized spread vs assumed 0.35 pts ONLY (mean, median, p95, worst, split London/NY AM). No slippage (not measurable in paper mode). Never alter cost model or net R.
- **C (Phase 3)**: monitor.py. First quote PREREG_TEST.md Section 5 verbatim and implement only what it states; ask me about anything ambiguous (do not invent gates like INCONCLUSIVE_FINAL unless quoted). Early-fail checked after every closed trade once N>=12. Telegram alerts. Long/short block labeled NOT FOR DECISIONS.
- **D (Phase 4)**: gate boundary tests (N=11/12/19/20), golden-file test on a TRAIN slice, STATE.md update. Do not edit PREREG_TEST.md.

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

# Verify Hash Chain Locally
python -c "from forward.logger import verify_chain; print(verify_chain('forward/signals.jsonl'))"

# Verify Hash Chain Against External Anchors (Detects Truncation/Tampering)
python -c "from forward.logger import verify_against_anchor; print(verify_against_anchor('forward/signals.jsonl', 'forward/anchors.json'))"
```
