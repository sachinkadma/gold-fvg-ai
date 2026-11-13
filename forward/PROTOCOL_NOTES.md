# Forward Protocol Audit Notes (PREREG Section 5)

## 1. Pre-Registration Protocol Start Rule (Verbatim)
From `research/PREREG_TEST.md` Section 5:
> "Start date = the day after the TEST run is logged in test_runs.log. Fills and costs are simulated with the same cost model. Each trade is logged with timestamp, entry, SL, TP, outcome in R. Every signal must be logged, no skipping."
> "End condition: N_forward = 20 executed trades OR 30 months, whichever comes first."

## 2. Benchmark Audit & Date Accounting
- **TEST Run Log Entry**:
  - File: `test_runs.log`
  - Timestamp (UTC): `2026-11-11T13:18:05.508647+00:00`
  - Verdict: `INCONCLUSIVE` (N=49 < 60 required)
  - Commit Hash: `44102295bbdd45f7ade5096642eafdf4806453f2`
- **Actual Forward Start Date**:
  - `FORWARD_START_DATE = 2026-11-12T00:00:00Z` (`2026-11-12`)
- **Accounting Gap**:
  - Time elapsed between TEST run log and Forward Start:
    $$\Delta t = 10\text{h } 41\text{m } 55\text{s} \approx 0.446\text{ days (calendar day difference: +1 day)}$$
- **30-Month Duration & End Date**:
  - Computed from Forward Start Date (`2026-11-12T00:00:00Z`):
    - Strict Calendar 30-Month Boundary: `2029-05-12T00:00:00Z` (exact calendar-month computation stored in GENESIS).
  - End condition triggers on whichever comes first: 20 executed trades ($N=20$) OR 30 calendar months elapsed.

## 3. Sample Exclusion in the Intervening Gap
- All market bars, ticks, or candidate signals that occurred between `2026-11-11T13:18:05Z` (TEST completion) and `2026-11-12T00:00:00Z` (Forward Start) are strictly outside the forward sample.
- `forward/logger.py` enforces this boundary via the preload guard: bars before `FORWARD_START_DATE` are only used for the 135-bar rolling warm-up buffer; no `SIGNAL`, `FILL`, `SPREAD_OBS`, or `CLOSE` events may be emitted prior to `FORWARD_START_DATE`.

## 4. Infrastructure Audit: Canonical LF Frozen Hashes & Line-Ending Normalization (2026-11-13)
- **Discrepancy Discovered**:
  During pre-flight validation on clean checkouts with `core.autocrlf=false`, `forward/logger.py:_run_preflight_checks` failed on `core/signals.py` with hash mismatch (expected `786c1c3b...`, got `88f8f3bf...`).
- **Audit & Proof**:
  Investigation proved that all authoritative git blobs stored in the git repository at `forward-v12` use canonical LF line endings (`\n`). However, `FROZEN_FILE_HASHES` in `forward/logger.py` historically contained a hybrid mix:
  1. `core/signals.py`: Expected `786c1c3b82...` (exact hash when rendered with CRLF `\r\n`). Canonical git blob (LF) is `88f8f3bfb9...`.
  2. `core/backtester.py`: Expected `398d2f251f...` (exact hash when rendered with CRLF `\r\n`). Canonical git blob (LF) is `79fdc6df97...`.
  3. `research/meta_labeling/pipeline_metalabeling.py`: Expected `f7e7bcee6e...` (exact hash when rendered with CRLF `\r\n`). Canonical git blob (LF) is `f9f42f479c...`.
  4. `core/config.py`: Expected `af1afb8cdf...` (canonical LF hash).
  5. `research/PREREG_TEST.md`: Expected `6aca6d3c8f...` (canonical LF hash).
  No single global `core.autocrlf` setting could satisfy both LF and CRLF expected hashes simultaneously in a uniform checkout.
- **Resolution**:
  1. Updated `FROZEN_FILE_HASHES` in `forward/logger.py` to use canonical LF blob hashes across all 5 frozen files.
  2. Created `.gitattributes` enforcing explicit `text eol=lf` for exactly these 5 paths (`core/signals.py`, `core/config.py`, `core/backtester.py`, `research/PREREG_TEST.md`, `research/meta_labeling/pipeline_metalabeling.py`), guaranteeing that checkouts on both Windows and Linux checkout identical LF bytes regardless of host git configuration.
- **Protocol & Strategy Invariance**:
  - Exactly zero files under `core/` or `research/` were modified. Strategy rules, candidate generation, backtesting simulation, and parameters ($RR=2.0$, $P\_SCALED=0.6077$, killzones) remain 100% frozen and bit-for-bit identical to tag `forward-v12`.
  - Tag `forward-v12` (`3041484`) is preserved untouched. This infrastructure fix is tagged as `forward-v13`.
