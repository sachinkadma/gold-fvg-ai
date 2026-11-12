"""
Forward Monitor and Gate Evaluator for PREREG Section 5.

================================================================================
PREREG_TEST.md Section 5 (Verbatim):
--------------------------------------------------------------------------------
## 5. Forward Paper-Trading Stage (only if TEST verdict = INCONCLUSIVE)

- Rule is frozen: killzone + stop >= 0.6077% of signal close + RR 2.0 + standard costs. No parameter, filter or session changes during the forward stage.
- Start date = the day after the TEST run is logged in test_runs.log. Fills and costs are simulated with the same cost model. Each trade is logged with timestamp, entry, SL, TP, outcome in R. Every signal must be logged, no skipping.
- End condition: N_forward = 20 executed trades OR 30 months, whichever comes first.
- Forward verdict:
  - FAIL_FWD: N >= 12 AND net expectancy <= 0, evaluated only at N=12 and at end.
  - PASS_FWD: N >= 20 AND net expectancy > 0 AND monthly-block 95% CI lower bound > 0.
  - Anything else (including N < 20 at 30 months): strategy research is CLOSED as INCONCLUSIVE-FINAL. No further extension.
- Operational context: Expected N ≈ 17-20; PASS_FWD needs roughly +0.65R; INCONCLUSIVE-FINAL / FAIL_FWD most likely. PASS_FWD does not authorize live money.
================================================================================

Governance:
1. READ-ONLY on the hash chain. Never appends to or alters the forward log.
2. Verified hash chain: verify_chain() must pass before computing metrics;
   outputs CHAIN_INVALID and halts if integrity check fails.
3. N definition: trades with PASSED_PRIMARY + FILLED + CLOSED only.
   Ignores SIGNAL-only, SPREAD_OBS, and GAP records.
4. Uses net_r from CLOSE events.
5. Monthly-block 95% bootstrap CI copied unchanged from run_prereg_test.py.
6. Long/Short breakdown block is strictly labeled [OBSERVATION ONLY - NOT FOR DECISIONS].
7. Telegram alerts fail gracefully: network or config failure never crashes the monitor.
"""

import os
import sys
import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from forward.logger import verify_chain, FORWARD_START_DATE

logger = logging.getLogger("forward_monitor")

# PREREG Section 5 Constants
TARGET_N_FORWARD = 20
EARLY_FAIL_N = 12
MAX_DURATION_MONTHS = 30.0


def block_bootstrap_ci(r_vals, month_periods, n_resamples=2000, seed=42) -> Tuple[float, float]:
    """
    Computes 95% monthly block bootstrap confidence interval.
    Exact implementation copied unchanged from research/meta_labeling/run_prereg_test.py.
    """
    if len(r_vals) < 5 or len(month_periods) == 0:
        mean_val = float(np.mean(r_vals)) if len(r_vals) > 0 else 0.0
        return (mean_val, mean_val)

    df_tmp = pd.DataFrame({"r": r_vals, "month": month_periods})
    monthly_groups = [group["r"].values for _, group in df_tmp.groupby("month")]
    n_blocks = len(monthly_groups)

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_resamples):
        sampled_blocks = rng.choice(n_blocks, size=n_blocks, replace=True)
        sampled_trades = np.concatenate([monthly_groups[i] for i in sampled_blocks])
        boot_means.append(np.mean(sampled_trades))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return round(ci_lower, 3), round(ci_upper, 3)


def send_telegram_alert(message: str, token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    """
    Sends an alert message to Telegram.
    Reads credentials from TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars if not passed.
    Fails safely: logs and returns False on any error, never crashes.
    """
    bot_token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    target_chat = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not target_chat:
        logger.debug("Telegram alert skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": target_chat,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning(f"Telegram alert delivery failed (safe fallback): {e}")
        return False


def alert_new_signal(signal_data: Dict[str, Any], token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    """Sends Telegram alert for a new valid candidate signal."""
    sig_id = signal_data.get("signal_id", "UNKNOWN")
    direction = signal_data.get("direction", "UNKNOWN")
    entry = signal_data.get("entry", 0.0)
    sl = signal_data.get("sl", 0.0)
    tp = signal_data.get("tp", 0.0)
    kz = signal_data.get("killzone", "KZ")
    msg = (
        f"🎯 *NEW FORWARD SIGNAL*: `{sig_id}`\n"
        f"*Direction*: `{direction}`\n"
        f"*Entry*: `{entry:.3f}` | *SL*: `{sl:.3f}` | *TP*: `{tp:.3f}`\n"
        f"*Killzone*: `{kz}`"
    )
    return send_telegram_alert(msg, token=token, chat_id=chat_id)


def alert_trade_closed(close_data: Dict[str, Any], token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    """Sends Telegram alert for a completed paper forward trade."""
    sig_id = close_data.get("signal_id", "UNKNOWN")
    direction = close_data.get("direction", "UNKNOWN")
    net_r = close_data.get("net_r", 0.0)
    reason = close_data.get("exit_reason", "UNKNOWN")
    outcome_icon = "🟢" if net_r > 0 else "🔴"
    msg = (
        f"{outcome_icon} *TRADE CLOSED*: `{sig_id}`\n"
        f"*Direction*: `{direction}`\n"
        f"*Net R*: `{net_r:+.3f} R`\n"
        f"*Exit Reason*: `{reason}`"
    )
    return send_telegram_alert(msg, token=token, chat_id=chat_id)


class ForwardMonitor:
    """
    Read-only forward trading monitor and PREREG Section 5 gate evaluator.
    """

    def __init__(
        self,
        log_path: str,
        status_path: Optional[str] = None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None
    ):
        self.log_path = os.path.abspath(log_path)
        self.status_path = os.path.abspath(status_path) if status_path else None
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id

    def evaluate(self, current_time: Optional[Union[str, datetime]] = None) -> Dict[str, Any]:
        """
        Evaluates the forward trading chain against PREREG Section 5 gates.
        Returns a comprehensive status dictionary.
        """
        # 1. Integrity check: chain must be 100% valid
        if not os.path.exists(self.log_path):
            return {
                "verdict": "NO_DATA",
                "status_detail": f"Log file not found at {self.log_path}",
                "chain_valid": False,
                "n_trades": 0
            }

        valid, msg, count = verify_chain(self.log_path)
        if not valid:
            return {
                "verdict": "CHAIN_INVALID",
                "status_detail": f"Forward log chain integrity check failed: {msg}",
                "chain_valid": False,
                "event_count": count,
                "n_trades": 0
            }

        # 2. Parse records
        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not records:
            return {
                "verdict": "NO_DATA",
                "status_detail": "Forward log contains zero records.",
                "chain_valid": True,
                "n_trades": 0
            }

        # 3. Read GENESIS metadata
        genesis = records[0].get("data", {}) if records[0].get("event") == "GENESIS" else {}
        start_date_str = genesis.get("forward_start_date", FORWARD_START_DATE)

        try:
            if "T" in start_date_str:
                start_dt = datetime.fromisoformat(start_date_str)
            else:
                start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
        except Exception:
            start_dt = datetime(2026, 11, 12, tzinfo=timezone.utc)

        # 4. Extract trades: N = PASSED_PRIMARY + FILLED + CLOSED
        signals_map = {}
        for r in records:
            if r.get("event") == "SIGNAL":
                d = r.get("data", {})
                if d.get("passed_primary") is True:
                    sig_id = d.get("signal_id")
                    if sig_id:
                        signals_map[sig_id] = {
                            "signal_data": d,
                            "signal_time": r.get("timestamp_utc"),
                            "direction": d.get("direction")
                        }

        # Match CLOSE records to valid signals
        raw_closed_trades = []
        latest_event_time = start_dt

        for r in records:
            ev_ts_str = r.get("timestamp_utc")
            if ev_ts_str:
                try:
                    ev_ts = datetime.fromisoformat(ev_ts_str)
                    if ev_ts.tzinfo is None:
                        ev_ts = ev_ts.replace(tzinfo=timezone.utc)
                    if ev_ts > latest_event_time:
                        latest_event_time = ev_ts
                except Exception:
                    pass

            if r.get("event") == "CLOSE":
                d = r.get("data", {})
                sig_id = d.get("signal_id")
                # Must be a primary passed signal that was filled and closed
                if sig_id in signals_map:
                    sig_info = signals_map[sig_id]
                    close_time_str = d.get("exit_time_utc") or r.get("timestamp_utc")
                    direction = d.get("direction") or sig_info["direction"]
                    raw_closed_trades.append({
                        "signal_id": sig_id,
                        "direction": direction,
                        "net_r": float(d.get("net_r", 0.0)),
                        "exit_reason": d.get("exit_reason", "UNKNOWN"),
                        "entry_price": d.get("entry_price"),
                        "exit_price": d.get("exit_price"),
                        "close_time": close_time_str
                    })

        # Cap sample at exactly TARGET_N_FORWARD (20 trades). Any 21st+ trade is ignored.
        closed_trades = raw_closed_trades[:TARGET_N_FORWARD]
        total_closed_recorded = len(raw_closed_trades)

        # 5. Compute calendar-month elapsed time and exact 30-month end limit
        end_date_30m_str = genesis.get("forward_end_date_30m")
        if end_date_30m_str:
            try:
                end_dt_30m = datetime.fromisoformat(end_date_30m_str)
                if end_dt_30m.tzinfo is None:
                    end_dt_30m = end_dt_30m.replace(tzinfo=timezone.utc)
            except Exception:
                end_dt_30m = (pd.Timestamp(start_dt) + pd.DateOffset(months=30)).to_pydatetime()
        else:
            end_dt_30m = (pd.Timestamp(start_dt) + pd.DateOffset(months=30)).to_pydatetime()

        if end_dt_30m.tzinfo is None:
            end_dt_30m = end_dt_30m.replace(tzinfo=timezone.utc)

        if current_time is not None:
            if isinstance(current_time, str):
                eval_dt = datetime.fromisoformat(current_time)
            else:
                eval_dt = current_time
            if eval_dt.tzinfo is None:
                eval_dt = eval_dt.replace(tzinfo=timezone.utc)
        else:
            eval_dt = latest_event_time

        elapsed_days = max(0.0, (eval_dt - start_dt).total_seconds() / 86400.0)
        total_30m_days = max(1.0, (end_dt_30m - start_dt).total_seconds() / 86400.0)
        elapsed_months = min(30.0, (elapsed_days / total_30m_days) * 30.0)
        time_limit_reached = (eval_dt >= end_dt_30m)

        # 6. Compute statistical metrics on the active sample (capped at 20)
        n_trades = len(closed_trades)
        if n_trades > 0:
            r_vals = np.array([t["net_r"] for t in closed_trades], dtype=float)
            month_periods = [
                pd.to_datetime(t["close_time"]).tz_localize(None).to_period("M")
                for t in closed_trades
            ]
            wins = r_vals[r_vals > 0]
            losses = r_vals[r_vals <= 0]
            win_rate_pct = round((len(wins) / n_trades) * 100.0, 1)
            expectancy_r = round(float(np.mean(r_vals)), 3)
            total_r = round(float(np.sum(r_vals)), 2)

            gross_gains = float(np.sum(wins)) if len(wins) > 0 else 0.0
            gross_losses = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
            if gross_losses > 0:
                profit_factor = round(gross_gains / gross_losses, 2)
            else:
                profit_factor = round(gross_gains, 2) if gross_gains > 0 else 0.0

            ci_lower, ci_upper = block_bootstrap_ci(r_vals, month_periods, n_resamples=2000, seed=42)
        else:
            win_rate_pct = 0.0
            expectancy_r = 0.0
            total_r = 0.0
            profit_factor = 0.0
            ci_lower, ci_upper = 0.0, 0.0

        # Check early-fail gate at N=12 (latched using the first 12 closed trades in chain order)
        n12_evaluated = (n_trades >= EARLY_FAIL_N)
        if n12_evaluated:
            r_12 = np.array([t["net_r"] for t in closed_trades[:EARLY_FAIL_N]], dtype=float)
            exp_12 = round(float(np.mean(r_12)), 3)
            n12_failed = (exp_12 <= 0.0)
        else:
            exp_12 = None
            n12_failed = False

        # 7. Long/Short breakdown [OBSERVATION ONLY - NOT FOR DECISIONS]
        long_trades = [t for t in closed_trades if t["direction"] == "BUY"]
        short_trades = [t for t in closed_trades if t["direction"] == "SELL"]

        def calc_dir_stats(t_list):
            cnt = len(t_list)
            if cnt == 0:
                return {"n": 0, "win_rate_pct": 0.0, "expectancy_r": 0.0, "total_r": 0.0}
            vals = np.array([t["net_r"] for t in t_list], dtype=float)
            w = len(vals[vals > 0])
            return {
                "n": cnt,
                "win_rate_pct": round((w / cnt) * 100.0, 1),
                "expectancy_r": round(float(np.mean(vals)), 3),
                "total_r": round(float(np.sum(vals)), 2)
            }

        obs_direction_breakdown = {
            "label": "[OBSERVATION ONLY - NOT FOR DECISIONS]",
            "long": calc_dir_stats(long_trades),
            "short": calc_dir_stats(short_trades)
        }

        # 8. Evaluate Gates per PREREG Section 5 wording:
        # - Early-fail evaluated ONLY at N=12 and at end.
        # - End = 20th closed trade OR 30 months from genesis start date.
        # - Terminal once a final status is reached.
        verdict = "CONTINUE"
        verdict_detail = ""

        # Gate 1: Did early-fail latch at N=12?
        if n12_failed:
            verdict = "FAIL_FWD"
            verdict_detail = (
                f"FAIL_FWD: Early-fail latched at N={EARLY_FAIL_N} trades with "
                f"net expectancy={exp_12:+.3f}R <= 0. Terminal verdict."
            )
        # Gate 2: End condition reached (N=20 closed trades OR 30 months)
        elif n_trades >= TARGET_N_FORWARD or time_limit_reached:
            if n_trades >= TARGET_N_FORWARD and expectancy_r > 0.0 and ci_lower > 0.0:
                verdict = "PASS_FWD"
                verdict_detail = (
                    f"PASS_FWD at end: N={n_trades} >= {TARGET_N_FORWARD}, "
                    f"net expectancy={expectancy_r:+.3f}R > 0, and monthly-block 95% CI lower bound={ci_lower:+.3f}R > 0."
                )
            elif n_trades >= EARLY_FAIL_N and expectancy_r <= 0.0:
                verdict = "FAIL_FWD"
                verdict_detail = (
                    f"FAIL_FWD at end: N={n_trades} >= {EARLY_FAIL_N} with net expectancy={expectancy_r:+.3f}R <= 0."
                )
            else:
                verdict = "INCONCLUSIVE-FINAL"
                if n_trades >= TARGET_N_FORWARD:
                    verdict_detail = (
                        f"INCONCLUSIVE-FINAL at end: N={n_trades} reached with net expectancy={expectancy_r:+.3f}R > 0 "
                        f"but 95% CI lower bound={ci_lower:+.3f}R <= 0. Strategy research CLOSED as INCONCLUSIVE-FINAL."
                    )
                else:
                    verdict_detail = (
                        f"INCONCLUSIVE-FINAL at end: 30-month duration elapsed with N={n_trades} (< {TARGET_N_FORWARD} required). "
                        f"Strategy research CLOSED as INCONCLUSIVE-FINAL."
                    )
        # Gate 3: Ongoing evaluation (N < 20 and elapsed < 30 months)
        else:
            verdict = "CONTINUE"
            if n_trades >= EARLY_FAIL_N:
                verdict_detail = (
                    f"CONTINUE: Early-fail avoided at N={EARLY_FAIL_N} (exp_12={exp_12:+.3f}R > 0). "
                    f"N={n_trades}/{TARGET_N_FORWARD} trades executed. Elapsed: {elapsed_months:.1f}/{MAX_DURATION_MONTHS:.1f} months."
                )
            else:
                verdict_detail = (
                    f"CONTINUE: Forward stage active. N={n_trades}/{TARGET_N_FORWARD} trades executed. "
                    f"Elapsed: {elapsed_months:.1f}/{MAX_DURATION_MONTHS:.1f} months."
                )

        status_result = {
            "verdict": verdict,
            "verdict_detail": verdict_detail,
            "chain_valid": True,
            "forward_start_date": start_date_str,
            "forward_end_date_30m": end_dt_30m.isoformat(),
            "eval_time": eval_dt.isoformat(),
            "elapsed_months": round(elapsed_months, 2),
            "max_duration_months": MAX_DURATION_MONTHS,
            "time_limit_reached": time_limit_reached,
            "metrics": {
                "n_trades": n_trades,
                "target_n": TARGET_N_FORWARD,
                "early_fail_n": EARLY_FAIL_N,
                "win_rate_pct": win_rate_pct,
                "expectancy_r": expectancy_r,
                "total_r": total_r,
                "profit_factor": profit_factor,
                "ci_95": [ci_lower, ci_upper]
            },
            "observation_direction_breakdown": obs_direction_breakdown,
            "trades": closed_trades
        }

        if self.status_path:
            self.save_status(status_result, self.status_path)

        return status_result

    def save_status(self, status: Dict[str, Any], path: Optional[str] = None):
        """Saves evaluation status to JSON file."""
        out_path = path or self.status_path
        if not out_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, default=str)

    def format_console_summary(self, status: Dict[str, Any]) -> str:
        """Formats clean console summary of forward status."""
        lines = []
        lines.append("=" * 72)
        lines.append(f"PREREG SECTION 5 FORWARD MONITOR - STATUS: {status.get('verdict')}")
        lines.append("=" * 72)
        lines.append(f"  Detail             : {status.get('verdict_detail')}")
        lines.append(f"  Chain Valid        : {status.get('chain_valid')}")
        lines.append(f"  Forward Start Date : {status.get('forward_start_date')}")
        lines.append(f"  Elapsed Months     : {status.get('elapsed_months', 0.0):.1f} / {status.get('max_duration_months', 30.0):.1f}")

        m = status.get("metrics", {})
        lines.append(f"\n  Executed Trades (N): {m.get('n_trades', 0)} / {m.get('target_n', 20)} (early-fail gate: N={m.get('early_fail_n', 12)})")
        lines.append(f"  Win Rate           : {m.get('win_rate_pct', 0.0):.1f}%")
        lines.append(f"  Net Expectancy     : {m.get('expectancy_r', 0.0):+.3f} R")
        lines.append(f"  Total Net R        : {m.get('total_r', 0.0):+.2f} R")
        lines.append(f"  Profit Factor      : {m.get('profit_factor', 0.0):.2f}")
        ci = m.get("ci_95", [0.0, 0.0])
        lines.append(f"  95% Monthly CI     : [{ci[0]:+.3f} R, {ci[1]:+.3f} R]")

        # Directional breakdown (observation only)
        obs = status.get("observation_direction_breakdown", {})
        lines.append(f"\n  {obs.get('label', '[OBSERVATION ONLY - NOT FOR DECISIONS]')}:")
        lng = obs.get("long", {})
        lines.append(f"    LONG : N={lng.get('n', 0)}, WR={lng.get('win_rate_pct', 0.0):.1f}%, EXP={lng.get('expectancy_r', 0.0):+.3f}R, TOT={lng.get('total_r', 0.0):+.2f}R")
        sht = obs.get("short", {})
        lines.append(f"    SHORT: N={sht.get('n', 0)}, WR={sht.get('win_rate_pct', 0.0):.1f}%, EXP={sht.get('expectancy_r', 0.0):+.3f}R, TOT={sht.get('total_r', 0.0):+.2f}R")
        lines.append("=" * 72)
        return "\n".join(lines)

    def alert_on_status_change(self, old_verdict: Optional[str], current_status: Dict[str, Any]) -> bool:
        """Sends Telegram alert if gate status changed."""
        cur_verdict = current_status.get("verdict")
        if not cur_verdict or cur_verdict == old_verdict:
            return False

        m = current_status.get("metrics", {})
        ci = m.get("ci_95", [0.0, 0.0])
        msg = (
            f"🚨 *FORWARD GATE VERDICT CHANGE*\n"
            f"*Verdict*: `{cur_verdict}` (was `{old_verdict}`)\n"
            f"*Detail*: {current_status.get('verdict_detail')}\n"
            f"*Trades (N)*: {m.get('n_trades', 0)}/20\n"
            f"*Expectancy*: `{m.get('expectancy_r', 0.0):+.3f} R`\n"
            f"*95% CI*: `[{ci[0]:+.3f}, {ci[1]:+.3f}]`\n"
            f"*Elapsed*: {current_status.get('elapsed_months', 0.0):.1f}/30.0 months"
        )
        return send_telegram_alert(msg, token=self.telegram_token, chat_id=self.telegram_chat_id)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PREREG Section 5 Forward Monitor")
    parser.add_argument("--log", default="forward/forward_signals.jsonl", help="Path to forward event log")
    parser.add_argument("--status", default="forward/forward_status.json", help="Path to save status JSON")
    parser.add_argument("--eval-time", default=None, help="Custom evaluation timestamp (ISO format)")
    args = parser.parse_args()

    monitor = ForwardMonitor(log_path=args.log, status_path=args.status)
    status = monitor.evaluate(current_time=args.eval_time)
    print(monitor.format_console_summary(status))


if __name__ == "__main__":
    main()
