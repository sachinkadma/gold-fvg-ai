# Windows Laptop Operational Guide (24x5 Forward Paper Trading)

Complete operational setup, automation, power management, disaster recovery, and rehearsal procedures for running the ICT/SMC Gold strategy forward paper-trading engine on a dedicated Windows laptop without a VPS.

---

## 1. Clean Clone & Isolation (Step A)
Run the forward engine in a dedicated, isolated deployment directory:
```powershell
# In PowerShell:
git clone https://github.com/sachinkadma/gold-fvg-ai.git C:\trading\forward_xauusd
cd C:\trading\forward_xauusd
git checkout forward-v12
pip install -r requirements.txt
# Note: data/ is gitignored; historical datasets are not required for forward runner operation.
git status --porcelain
# Confirm working tree is 100% clean
```

---

## 2. Environment Variables Configuration (Step B)
Set persistent user-level environment variables via PowerShell (start date is supplied via CLI):
```powershell
[System.Environment]::SetEnvironmentVariable("TELEGRAM_BOT_TOKEN", "your_telegram_bot_token_here", "User")
[System.Environment]::SetEnvironmentVariable("TELEGRAM_CHAT_ID", "your_telegram_chat_id_here", "User")
[System.Environment]::SetEnvironmentVariable("MT5_LOGIN", "your_mt5_login_here", "User")
[System.Environment]::SetEnvironmentVariable("MT5_PASSWORD", "your_password", "User")
[System.Environment]::SetEnvironmentVariable("MT5_SERVER", "MetaQuotes-Demo", "User")
[System.Environment]::SetEnvironmentVariable("MT5_PATH", "C:\Program Files\MetaTrader 5\terminal64.exe", "User")
```

---

## 3. Windows Task Scheduler Setup (Step C)
Configure two persistent tasks to launch automatically on user logon with restart-on-failure:

### Task 1: Forward Runner Supervisor (`ForwardRunner_Supervisor`)
- **Trigger**: At log on of any user.
- **Action**: Start a program.
  - Program/script: `powershell.exe`
  - Production Arguments: `-NoProfile -ExecutionPolicy Bypass -File "C:\trading\forward_xauusd\forward\launch\run_forever.ps1"`
  - Rehearsal Arguments (for reboot drills): `-NoProfile -ExecutionPolicy Bypass -Command "$s=(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'); while ($true) { python forward/run_forward.py --forward-start-date $s --rehearsal; Start-Sleep -Seconds 5 }"`
  - Start in: `C:\trading\forward_xauusd`
- **Settings**:
  - Check "If the task fails, restart every: 1 minute"
  - Attempt to restart up to: 999 times
  - Uncheck "Stop if the task runs longer than..."

### Task 2: Daily Heartbeat & Stale Feed Monitor (`ForwardRunner_Heartbeat`)
- **Trigger**: At log on of any user.
- **Action**: Start a program.
  - Program/script: `python.exe`
  - Arguments: `forward/launch/heartbeat.py`
  - Start in: `C:\trading\forward_xauusd`
- **Settings**:
  - Check "If the task fails, restart every: 1 minute"

---

## 4. Laptop Power Plan & Windows Update Settings (Step D)
To ensure uninterrupted 24x5 operation Monday morning through Friday evening:

1. **Power Plan (High Performance / Ultimate Performance)**:
   - Run in admin PowerShell:
     ```powershell
     # Disable Sleep on AC Power
     powercfg -change -standby-timeout-ac 0
     # Disable Hibernate on AC Power
     powercfg -change -hibernate-timeout-ac 0
     # Disable Display Turn-off or set to 15 min (display off is fine, sleep is prohibited)
     powercfg -change -monitor-timeout-ac 15
     ```
2. **Lid Close Action**:
   - Control Panel -> Power Options -> "Choose what closing the lid does":
     - Set "When I close the lid" (Plugged in): **Do nothing**.
3. **Windows Update Active Hours & Automatic Reboots**:
   - Settings -> Windows Update -> Advanced options:
     - Set Active Hours: Manually set 00:00 to 23:00 (maximum allowed range).
     - Disable "Get me up to date" automatic reboots.
     - Optional: Pause updates for 5 weeks during active forward phases, performing manual updates only during Saturday weekend maintenance.

---

## 5. Daily Backup Routine (Step E)
Configure a daily backup of the append-only hash chain and companion buffer state:
```powershell
# Daily Backup Script (e.g. to OneDrive or external drive D:\Backups\trading):
$SourceDir = "C:\trading\forward_xauusd\forward"
$BackupDir = "D:\Backups\forward_trading_$(Get-Date -Format 'yyyyMMdd')"
robocopy $SourceDir $BackupDir signals.jsonl signals_buffer.json signals_anchor_queue.json /R:3 /W:5
```

---

## 6. Mandatory 48–72h Live Rehearsal (Step F)
Before initiating the permanent forward trading log, conduct a 48–72 hour drill on an isolated rehearsal environment:

1. **Start Rehearsal (Isolated Mode)**:
   ```powershell
   $RehearsalStart = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
   python forward/run_forward.py --forward-start-date $RehearsalStart --rehearsal
   ```
   *Supervised Rehearsal Loop (for kill/reboot drills)*:
   ```powershell
   $RehearsalStart = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
   while ($true) { python forward/run_forward.py --forward-start-date $RehearsalStart --rehearsal; Start-Sleep -Seconds 5 }
   ```
   *Isolation Guarantees*:
   - Automatically writes to `forward/rehearsal.jsonl` (strictly prohibits writing to production `forward/signals.jsonl`).
   - Automatically uses anchor ledger `forward/rehearsal_anchors.json`.
   - Dedicated git anchor branch `forward-anchors-rehearsal` (strictly prohibits touching or pushing to `forward-anchors`).

2. **Drill Actions**:
   - **Kill Process Drill**: Terminate the python process abruptly via Task Manager. Verify `run_forever.ps1` restarts the engine within 10 seconds and recovers pending orders / open positions from `rehearsal.jsonl`.
   - **Reboot Drill**: Restart the laptop while orders are pending. Verify Task Scheduler resumes execution and reconstructs internal state cleanly.
   - **Network Offline Drill**: Disconnect Wi-Fi / Ethernet for 30 minutes. Verify downtime is detected and logged as a `DOWNTIME_GAP` upon reconnection.

3. **Rehearsal Audit & Verification**:
   ```powershell
   python forward/launch/rehearsal_check.py --log forward/rehearsal.jsonl --anchor-ledger forward/rehearsal_anchors.json
   ```
   *Audit Checks*:
   - Cryptographic hash chain validity (`verify_chain()`).
   - Anchor ledger consistency and truncation detection (`verify_against_anchor()`).
   - Counts of each event type (`GENESIS`, `SIGNAL`, `SPREAD_OBS`, `FILL`, `CLOSE`, `GAP`).
   - SIGNAL breakdown by `filter_status`.
   - GAP breakdown: scheduled weekend/daily breaks vs unexpected downtime gaps.
   - Candle timestamp uniqueness (zero duplicates).
   - Zero missing 15m candles outside scheduled market breaks.
   - Returns exit code 0 ONLY if all checks are completely clean.

---

## 7. Weekend Routine & Market Close GAPs (Step G)
- **Friday 17:00 NY**: Gold market closes.
  - The forward runner may remain running idling or can be shut down for maintenance.
  - Heartbeat automatically recognizes market close and suppresses `STALE_FEED` alerts.
- **Sunday 18:00 NY**: Gold market reopens.
  - The engine polls the first Sunday candle. If shut down over the weekend, upon startup it compares the first candle time against Friday's close and automatically inserts a `GAP` record with `gap_type: "WEEKEND_MARKET_CLOSE"`.
  - Weekend market close `GAP` records are normal, expected, and strictly ignored by trade metrics ($N$, win rate, expectancy, and net R).
