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
    - Calendar 30-Month Boundary: `2029-05-12T00:00:00Z`
    - Standard Day Scale ($30 \times 30.4375\text{ days} = 913.125\text{ days}$): `2029-05-13T03:00:00Z`
  - End condition triggers on whichever comes first: 20 executed trades ($N=20$) OR 30 months elapsed.

## 3. Sample Exclusion in the Intervening Gap
- All market bars, ticks, or candidate signals that occurred between `2026-11-11T13:18:05Z` (TEST completion) and `2026-11-12T00:00:00Z` (Forward Start) are strictly outside the forward sample.
- `forward/logger.py` enforces this boundary via the preload guard: bars before `FORWARD_START_DATE` are only used for the 135-bar rolling warm-up buffer; no `SIGNAL`, `FILL`, `SPREAD_OBS`, or `CLOSE` events may be emitted prior to `FORWARD_START_DATE`.
