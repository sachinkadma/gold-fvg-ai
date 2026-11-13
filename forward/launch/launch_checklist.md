# Forward Paper-Trading Launch Checklist

Strict step-by-step pre-flight procedures to execute before initiating the forward paper-trading stage.

## 1. Fresh Clone at Final Tag
- Clone the repository into a clean, dedicated deployment folder:
  ```powershell
  git clone https://github.com/sachinkadma/gold-fvg-ai.git forward_runner
  cd forward_runner
  git checkout forward-v14
  ```
- Verify git status is completely clean:
  ```powershell
  git status --porcelain
  # Output must be empty
  ```
- Install minimal runtime dependencies:
  ```powershell
  pip install -r requirements-forward.txt
  ```

## 2. Environment Variables Setup
Ensure broker credentials are configured.
*(Do NOT set a persistent `FORWARD_START_DATE` environment variable so stale timestamps cannot carry over across runs.)*

### PowerShell:
```powershell
$env:MT5_LOGIN = "<YOUR_MT5_LOGIN>"
$env:MT5_PASSWORD = "<YOUR_MT5_PASSWORD>"
$env:MT5_SERVER = "MetaQuotes-Demo"
$env:MT5_PATH = "C:\Program Files\MetaTrader 5\terminal64.exe"

# Optional Telegram alerting:
$env:TELEGRAM_BOT_TOKEN = "<YOUR_TELEGRAM_BOT_TOKEN>"
$env:TELEGRAM_CHAT_ID = "<YOUR_TELEGRAM_CHAT_ID>"
```

### Windows Command Prompt (cmd.exe):
```cmd
set MT5_LOGIN=<YOUR_MT5_LOGIN>
set MT5_PASSWORD=<YOUR_MT5_PASSWORD>
set MT5_SERVER=MetaQuotes-Demo
set MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe

rem Optional Telegram alerting:
set TELEGRAM_BOT_TOKEN=<YOUR_TELEGRAM_BOT_TOKEN>
set TELEGRAM_CHAT_ID=<YOUR_TELEGRAM_CHAT_ID>
```

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
- Rehearsal must be performed on the latest tag (`forward-v14`) to verify isolated logging and git anchor pushes.
- Launch rehearsal mode using the current UTC timestamp:
  - **PowerShell (Direct run)**:
    ```powershell
    $StartDate = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    python forward/run_forward.py --forward-start-date $StartDate --rehearsal
    ```
  - **PowerShell (Supervised run for kill & reboot drills)**:
    ```powershell
    $StartDate = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    powershell -ExecutionPolicy Bypass -File forward/launch/run_forever.ps1 -ForwardStartDate $StartDate -Rehearsal
    ```
  - **Command Prompt (cmd.exe)**:
    ```cmd
    python forward/run_forward.py --forward-start-date 2026-11-13T20:30:00Z --rehearsal
    ```
  *(Important: Any restart within the same 24h rehearsal must reuse the exact initial `$StartDate` timestamp.)*
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
- Start the supervisor process (passing the launch start date):
  ```powershell
  .\forward\launch\run_forever.ps1 -ForwardStartDate "YYYY-MM-DDTHH:MM:SSZ"
  ```
- Start the daily heartbeat monitor in a separate terminal:
  ```powershell
  python forward/launch/heartbeat.py
  ```
- Verify GENESIS record written as Index 0 in `forward/signals.jsonl`.
- Verify external anchor pushed to `forward-anchors` branch.
