# Forward Paper-Trading Launch Checklist

Strict step-by-step pre-flight procedures to execute before initiating the forward paper-trading stage.

## 1. Fresh Clone at Final Tag
- Clone the repository into a clean, dedicated deployment folder:
  ```powershell
  git clone https://github.com/sachinkadma/gold-fvg-ai.git forward_runner
  cd forward_runner
  git checkout forward-v9
  ```
- Verify git status is completely clean:
  ```powershell
  git status --porcelain
  # Output must be empty
  ```

## 2. Environment Variables Setup
Ensure all required environment variables are set in the Windows environment or user profile:
- `MT5_LOGIN`: MetaTrader 5 account login (e.g. `115395873`)
- `MT5_PASSWORD`: MetaTrader 5 account password
- `MT5_SERVER`: MetaTrader 5 broker server (e.g. `MetaQuotes-Demo`)
- `MT5_PATH`: Path to terminal binary (e.g. `C:\Program Files\MetaTrader 5\terminal64.exe`)
- `TELEGRAM_BOT_TOKEN`: Telegram bot token for alerting
- `TELEGRAM_CHAT_ID`: Telegram channel/chat ID for alerts and external head hash anchoring
- `FORWARD_START_DATE`: Exact UTC ISO startup timestamp (e.g. `2026-11-12T00:00:00Z`).
  *Enforces the 1-day staleness rule: startup refuses if start date is >24 hours in the past.*

## 3. MetaTrader 5 Broker Connection
- Start MetaTrader 5 terminal and verify demo account is logged in.
- Ensure `XAUUSD` is selected and visible in Market Watch.
- Verify live tick streaming and bar chart loading for `XAUUSD` M15.

## 4. External Anchor Remote Verification
- Confirm write access to the orphan remote anchor branch `forward-anchors`:
  ```powershell
  git ls-remote origin forward-anchors
  # Must return the remote branch commit hash without error
  ```
- **Clean Branch Guarantee**: Real `forward-anchors` must be clean (or recreated) before launch. Ensure `git ls-tree -r origin/forward-anchors` contains clean filenames with no trailing `\r` or carriage return artifacts.

## 5. Mandatory 48–72h Rehearsal Drill & Verification
- Rehearsal must be redone on the latest tag (`forward-v12`) to verify the Windows CRLF git mktree fix.
- Run rehearsal mode to confirm 24x5 live stability without touching production logs or branches:
  ```powershell
  python forward/run_forward.py --forward-start-date "2026-11-12T00:00:00Z" --rehearsal
  ```
- Run the read-only rehearsal check:
  ```powershell
  python forward/launch/rehearsal_check.py --log forward/rehearsal.jsonl --anchor-ledger forward/rehearsal_anchors.json
  ```
  *Must return EXIT 0 (all checks clean: valid chain, valid anchors, no duplicate candles, no unexpected downtime).*
- Verify that `git ls-tree -r origin/forward-anchors-rehearsal` shows `rehearsal_anchors.json` without trailing `\r`.

## 6. Pre-Flight Protocol Audit Update (MANDATORY BEFORE FIRST RUN)
Before starting the live forward run, update `forward/PROTOCOL_NOTES.md` with:
1. The **actual forward start date and time** (`FORWARD_START_DATE`).
2. The **exact gap in hours and days** between the TEST run log (`2026-11-11T13:18:05Z`) and the actual start date.
3. The **exact 30 calendar-month end date** computed from the actual start date (e.g., `2026-11-12` -> `2029-05-12`).
4. Re-verify the gap sample exclusion rule: any candle data or signals occurring during the gap are strictly excluded from the forward sample.
5. **Commit this update to git BEFORE launching the live runner**:
   ```powershell
   git add forward/PROTOCOL_NOTES.md
   git commit -m "docs(forward): update launch dates and gap audit before forward run"
   git push origin <branch>
   ```

## 7. Launch Execution
- Start the supervisor process:
  ```powershell
  .\forward\launch\run_forever.ps1
  ```
- Start the daily heartbeat monitor:
  ```powershell
  python forward/launch/heartbeat.py
  ```
- Verify GENESIS record written as Index 0 in `forward/signals.jsonl`.
- Verify external anchor pushed to `forward-anchors` branch.
