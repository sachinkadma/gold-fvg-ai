# Windows Operational Guide

This document mirrors [forward/launch/README_WINDOWS.md](file:///c:/Users/Harsh%20Computer/Music/New%20folder/forward/launch/README_WINDOWS.md). Please refer to the primary guide in `forward/launch/README_WINDOWS.md` for complete setup instructions.

### Quick Reference (forward-v14)
1. **Clone & Checkout**:
   ```powershell
   git clone https://github.com/sachinkadma/gold-fvg-ai.git C:\trading\forward_xauusd
   cd C:\trading\forward_xauusd
   git checkout forward-v14
   pip install -r requirements-forward.txt
   ```

2. **Environment Variables (cmd.exe vs PowerShell)**:
   - **cmd.exe**:
     ```cmd
     set MT5_LOGIN=<YOUR_MT5_LOGIN>
     set MT5_PASSWORD=<YOUR_MT5_PASSWORD>
     set MT5_SERVER=MetaQuotes-Demo
     set MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
     rem TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are optional
     ```
   - **PowerShell**:
     ```powershell
     $env:MT5_LOGIN = "<YOUR_MT5_LOGIN>"
     $env:MT5_PASSWORD = "<YOUR_MT5_PASSWORD>"
     $env:MT5_SERVER = "MetaQuotes-Demo"
     $env:MT5_PATH = "C:\Program Files\MetaTrader 5\terminal64.exe"
     # $env:TELEGRAM_BOT_TOKEN and $env:TELEGRAM_CHAT_ID are optional
     ```
   *(Do NOT set persistent FORWARD_START_DATE).*

3. **Rehearsal Launch**:
   - **PowerShell (Direct)**:
     ```powershell
     $StartDate = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
     python forward/run_forward.py --forward-start-date $StartDate --rehearsal
     ```
   - **PowerShell (Supervised for kill/reboot drills)**:
     ```powershell
     $StartDate = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
     powershell -ExecutionPolicy Bypass -File forward/launch/run_forever.ps1 -ForwardStartDate $StartDate -Rehearsal
     ```
   - Reuse `$StartDate` on any restart within the same 24h rehearsal.
