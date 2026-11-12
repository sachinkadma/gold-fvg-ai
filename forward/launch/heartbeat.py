"""
Daily Heartbeat and Feed Stale Monitor for Forward Paper-Trading (PREREG Section 5).

Responsibilities:
1. Runs as a separate, strictly READ-ONLY monitoring process.
2. Sends daily "alive" Telegram status message with:
   - Last closed candle timestamp (UTC & NY)
   - Last chain index & head hash
   - Executed trades N so far
   - Gate status (from ForwardMonitor)
3. Detects STALE_FEED:
   - Alerts if no new closed candle for >45 min during active market hours.
   - Strictly suppresses alert during weekend market close (Fri 17:00 to Sun 18:00 NY)
     and daily maintenance break (Mon-Thu 17:00 to 18:00 NY).
4. Detects PC SLEEP / TIME JUMP:
   - Alerts if loop iteration interval indicates PC was suspended, asleep, or offline.
5. Telegram delivery fails safely: never crashes or raises exceptions.
6. Environment variables only: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple
import zoneinfo

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from forward.monitor import ForwardMonitor, send_telegram_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [heartbeat] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("heartbeat")

NY_TZ = zoneinfo.ZoneInfo("America/New_York")


def is_xauusd_market_open(dt_ny: datetime) -> bool:
    """
    Returns True if the XAUUSD market is actively trading.
    Standard Trading Schedule in America/New_York clock time:
      - Opens Sunday 18:00 NY.
      - Closes Friday 17:00 NY.
      - Daily maintenance break: Mon-Thu 17:00 - 18:00 NY.
      - Closed Saturday full day.
      - Closed Sunday 00:00 - 18:00 NY.
      - Closed Friday 17:00 - 24:00 NY.
    """
    weekday = dt_ny.weekday()  # Monday = 0, ..., Sunday = 6
    hour = dt_ny.hour

    # Saturday: completely closed
    if weekday == 5:
        return False

    # Sunday: closed before 18:00 NY
    if weekday == 6:
        return hour >= 18

    # Friday: closes at 17:00 NY
    if weekday == 4:
        return hour < 17

    # Monday - Thursday (0, 1, 2, 3):
    # Daily market break between 17:00 and 18:00 NY
    if hour == 17:
        return False

    return True


def get_latest_chain_state(log_path: str) -> Dict[str, Any]:
    """Reads the last record from the append-only JSONL log (read-only)."""
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return {
            "has_data": False,
            "last_index": None,
            "last_hash": None,
            "last_event": None,
            "last_time_utc": None
        }

    last_line = None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
    except Exception as e:
        logger.warning(f"Error reading forward log: {e}")
        return {"has_data": False, "error": str(e)}

    if not last_line:
        return {"has_data": False}

    try:
        rec = json.loads(last_line)
        return {
            "has_data": True,
            "last_index": rec.get("index"),
            "last_hash": rec.get("record_hash"),
            "last_event": rec.get("event"),
            "last_time_utc": rec.get("timestamp_utc"),
            "data": rec.get("data", {})
        }
    except Exception as e:
        logger.warning(f"Error parsing last line: {e}")
        return {"has_data": False, "error": str(e)}


class ForwardHeartbeatDaemon:
    def __init__(
        self,
        log_path: str = "forward/signals.jsonl",
        stale_threshold_seconds: int = 45 * 60,  # 45 minutes
        poll_interval_seconds: int = 60,
        daily_report_hour_utc: int = 22  # 22:00 UTC (17:00 NY market break)
    ):
        self.log_path = os.path.abspath(log_path)
        self.stale_threshold_seconds = stale_threshold_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.daily_report_hour_utc = daily_report_hour_utc
        self.monitor = ForwardMonitor(log_path=self.log_path)

        self.last_alive_date_str = None
        self.stale_alert_sent = False
        self.last_candle_time_seen = None

    def run_check_cycle(self, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Executes a single check cycle.
        Returns status dictionary for unit testing.
        """
        now = now_utc or datetime.now(timezone.utc)
        now_ny = now.astimezone(NY_TZ)
        market_open = is_xauusd_market_open(now_ny)

        chain_state = get_latest_chain_state(self.log_path)
        eval_status = self.monitor.evaluate(current_time=now) if chain_state.get("has_data") else {}

        cycle_results = {
            "market_open": market_open,
            "chain_state": chain_state,
            "verdict": eval_status.get("verdict", "NO_DATA"),
            "stale_alert_triggered": False,
            "daily_alive_triggered": False
        }

        # 1. Stale Feed Check (only during market hours)
        last_ts_str = chain_state.get("last_time_utc")
        if last_ts_str:
            try:
                last_dt = datetime.fromisoformat(last_ts_str)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                self.last_candle_time_seen = last_dt
            except Exception:
                pass

        if market_open and self.last_candle_time_seen:
            diff_sec = (now - self.last_candle_time_seen).total_seconds()
            if diff_sec > self.stale_threshold_seconds:
                if not self.stale_alert_sent:
                    msg = (
                        f"⚠️ *STALE FEED ALERT*: No new closed candle for {diff_sec/60:.1f} minutes!\n"
                        f"*Market Status*: Open (`{now_ny.strftime('%a %H:%M')} NY`)\n"
                        f"*Last Candle Time*: `{self.last_candle_time_seen.isoformat()}`\n"
                        f"*Threshold*: {self.stale_threshold_seconds/60:.0f} min\n"
                        f"Check MT5 connection and run_forever supervisor!"
                    )
                    logger.warning(msg)
                    send_telegram_alert(msg)
                    self.stale_alert_sent = True
                    cycle_results["stale_alert_triggered"] = True
            else:
                self.stale_alert_sent = False
        else:
            # During market close, reset stale alert flag
            self.stale_alert_sent = False

        # 2. Daily Alive Telegram Message
        today_date_str = now.strftime("%Y-%m-%d")
        if now.hour == self.daily_report_hour_utc and self.last_alive_date_str != today_date_str:
            m = eval_status.get("metrics", {})
            last_idx = chain_state.get("last_index", 0)
            last_h = (chain_state.get("last_hash") or "N/A")[:12]
            ci = m.get("ci_95", [0.0, 0.0])

            alive_msg = (
                f"💓 *FORWARD TRADING DAILY HEARTBEAT*\n"
                f"*Status*: Operational | Market Open: `{market_open}`\n"
                f"*Verdict*: `{eval_status.get('verdict')}`\n"
                f"*Trades (N)*: {m.get('n_trades', 0)}/20 (early-fail gate: 12)\n"
                f"*Net Expectancy*: `{m.get('expectancy_r', 0.0):+.3f} R`\n"
                f"*95% CI*: `[{ci[0]:+.3f}, {ci[1]:+.3f}]`\n"
                f"*Chain Index*: `{last_idx}` (Hash: `{last_h}...`)\n"
                f"*Last Candle*: `{last_ts_str or 'N/A'}`"
            )
            logger.info("Sending daily heartbeat Telegram alert...")
            send_telegram_alert(alive_msg)
            self.last_alive_date_str = today_date_str
            cycle_results["daily_alive_triggered"] = True

        return cycle_results

    def start_loop(self):
        """Starts the infinite heartbeat loop with time jump detection."""
        logger.info(f"Heartbeat daemon started on log: {self.log_path}")
        last_loop_time = time.monotonic()
        last_wall_time = datetime.now(timezone.utc)

        while True:
            try:
                now_wall = datetime.now(timezone.utc)
                now_mono = time.monotonic()

                # Detect time jump (PC was asleep, hibernated, or suspended)
                wall_elapsed = (now_wall - last_wall_time).total_seconds()
                mono_elapsed = now_mono - last_loop_time

                if wall_elapsed > (self.poll_interval_seconds + 120):
                    jump_msg = (
                        f"⚠️ *PC TIME JUMP / SLEEP DETECTED*\n"
                        f"System was asleep or offline for ~{wall_elapsed/60:.1f} minutes!\n"
                        f"Wall gap: {wall_elapsed:.0f}s (expected ~{self.poll_interval_seconds}s).\n"
                        f"Resuming forward monitoring."
                    )
                    logger.warning(jump_msg)
                    send_telegram_alert(jump_msg)

                last_loop_time = now_mono
                last_wall_time = now_wall

                self.run_check_cycle(now_wall)

            except Exception as e:
                logger.error(f"Error in heartbeat cycle (safe fallback): {e}")

            time.sleep(self.poll_interval_seconds)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Forward Trading Heartbeat & Stale Feed Monitor")
    parser.add_argument("--log", default="forward/signals.jsonl", help="Path to forward event log")
    args = parser.parse_args()

    daemon = ForwardHeartbeatDaemon(log_path=args.log)
    daemon.start_loop()
