<#
.SYNOPSIS
    Supervisor script for running the forward paper-trading engine indefinitely on Windows.
.DESCRIPTION
    Runs forward/run_forward.py from repository root.
    Restarts automatically on transient exit/crash with a backoff delay.
    Writes supervisor events to forward/launch/supervisor.log (gitignored).
    Enforces strict pre-flight refusal handling: on startup refusal (dirty git tree,
    frozen file hash mismatch, unreachable git remote, or stale/invalid start date),
    sends an immediate Telegram alert and terminates retries after 3 attempts.
#>

param(
    [string]$LogPath = "forward/signals.jsonl",
    [string]$StartDate = $Env:FORWARD_START_DATE,
    [int]$MaxRefusalRetries = 3,
    [int]$BackoffSeconds = 10
)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $RepoRoot) { $RepoRoot = Get-Location }
Set-Location -Path $RepoRoot

$SupervisorLog = Join-Path $PSScriptRoot "supervisor.log"

function Write-SupervisorLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $logLine = "$timestamp [$Level] $Message"
    Write-Host $logLine
    Add-Content -Path $SupervisorLog -Value $logLine -ErrorAction SilentlyContinue
}

function Send-TelegramSupervisorAlert {
    param([string]$Message)
    $botToken = $Env:TELEGRAM_BOT_TOKEN
    $chatId = $Env:TELEGRAM_CHAT_ID
    if (-not $botToken -or -not $chatId) {
        Write-SupervisorLog "Telegram alert skipped: missing credentials" "WARN"
        return
    }
    try {
        $uri = "https://api.telegram.org/bot$botToken/sendMessage"
        $body = @{
            chat_id = $chatId
            text = $Message
            parse_mode = "Markdown"
        } | ConvertTo-Json
        $response = Invoke-RestMethod -Uri $uri -Method Post -ContentType "application/json" -Body $body -TimeoutSec 10
    } catch {
        Write-SupervisorLog "Failed to send Telegram alert: $_" "WARN"
    }
}

Write-SupervisorLog "Starting Forward Trading Supervisor in $RepoRoot" "INFO"

if (-not $StartDate) {
    $StartDate = "2026-11-12T00:00:00Z"
    Write-SupervisorLog "FORWARD_START_DATE not set in environment. Defaulting to $StartDate" "WARN"
}

$refusalCount = 0

while ($true) {
    Write-SupervisorLog "Launching forward/run_forward.py (StartDate: $StartDate, Log: $LogPath)..." "INFO"
    $startTime = Get-Date

    # Temporary stdout/stderr capture file
    $procOutput = Join-Path $PSScriptRoot "last_run_output.tmp"
    if (Test-Path $procOutput) { Remove-Item $procOutput -Force }

    $pythonExe = "python"
    $argsList = @(
        "forward/run_forward.py",
        "--forward-start-date", $StartDate,
        "--log-path", $LogPath
    )

    $process = Start-Process -FilePath $pythonExe -ArgumentList $argsList -NoNewWindow -Wait -PassThru -RedirectStandardOutput $procOutput -RedirectStandardError $procOutput
    $exitCode = $process.ExitCode
    $duration = (Get-Date) - $startTime

    $outputText = ""
    if (Test-Path $procOutput) {
        $outputText = Get-Content $procOutput -Raw -ErrorAction SilentlyContinue
        Write-Host $outputText
        Add-Content -Path $SupervisorLog -Value $outputText -ErrorAction SilentlyContinue
        Remove-Item $procOutput -Force -ErrorAction SilentlyContinue
    }

    Write-SupervisorLog "Process exited with code $exitCode after $($duration.TotalSeconds)s." "INFO"

    # Detect startup refusal (dirty git tree, frozen hash tampering, unreachable remote, bad start date)
    $isStartupRefusal = ($outputText -match "PRE-FLIGHT ERROR" -or 
                         $outputText -match "STARTUP ERROR" -or 
                         $outputText -match "FORWARD LOG INTEGRITY VIOLATION" -or
                         $outputText -match "Refusing to start")

    if ($exitCode -ne 0 -and $isStartupRefusal) {
        $refusalCount++
        Write-SupervisorLog "STARTUP REFUSAL detected (Attempt $refusalCount of $MaxRefusalRetries): $outputText" "ERROR"
        
        $alertMsg = "🚨 *FORWARD SUPERVISOR: STARTUP REFUSAL*`nAttempt: $refusalCount/$MaxRefusalRetries`nExit Code: $exitCode`nError: $outputText"
        Send-TelegramSupervisorAlert -Message $alertMsg

        if ($refusalCount -ge $MaxRefusalRetries) {
            Write-SupervisorLog "FATAL: Reached maximum startup refusal retries ($MaxRefusalRetries). Halting supervisor to prevent masking pre-flight failures." "FATAL"
            $fatalMsg = "🛑 *FORWARD SUPERVISOR TERMINATED*`nMax startup refusal attempts reached ($MaxRefusalRetries). Check git tree, frozen hashes, and remote connectivity."
            Send-TelegramSupervisorAlert -Message $fatalMsg
            exit 2
        }
    } else {
        # If the process ran for more than 5 minutes before exiting, reset refusal counter
        if ($duration.TotalMinutes -gt 5) {
            $refusalCount = 0
        }
    }

    Write-SupervisorLog "Waiting $BackoffSeconds seconds before restart..." "INFO"
    Start-Sleep -Seconds $BackoffSeconds
}
