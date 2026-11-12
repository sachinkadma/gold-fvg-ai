"""
Forward Execution Tracker for Realized Spread Observations.
PREREG Section 5 / Phase B Execution Quality Monitoring.

Responsibilities:
1. Observe and record real MT5 bid/ask/spread at each signal and each paper fill.
   - For live ticks: uses tick.bid, tick.ask (spread = ask - bid).
   - For replay/historical bars: uses bar['spread'] with symbol point size (0.01 for XAUUSD).
   - State units: price points in USD (e.g. 0.35 pts).
2. Format immutable SPREAD_OBS event payloads for the hash chain.
3. Generate realized spread reports comparing realized spread vs assumed 0.35 pts
   (mean, median, p95, worst), split by London and NY AM killzones.
4. Report-only: strictly never alters the frozen cost model or net R.
   Slippage is not measurable in paper forward mode; realized spread observation only.
"""

import os
import sys
import json
from typing import Dict, List, Optional, Any, Union
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.signals import is_in_killzone

# FROZEN ASSUMPTION (PREREG SECTION 5 / core/backtester.py)
ASSUMED_SPREAD_PTS = 0.35
DEFAULT_POINT_SIZE = 0.01  # MetaQuotes XAUUSD standard point size (100 pts = $1.00)
SPREAD_UNITS = "price_points_usd"


class ExecutionTracker:
    """
    Tracks and analyzes execution quality (realized spread) for paper forward trading.
    """

    def __init__(
        self,
        assumed_spread_pts: float = ASSUMED_SPREAD_PTS,
        point_size: float = DEFAULT_POINT_SIZE,
        symbol: str = "XAUUSD"
    ):
        self.assumed_spread_pts = float(assumed_spread_pts)
        self.point_size = float(point_size)
        self.symbol = symbol

    def build_spread_observation(
        self,
        trigger_event: str,
        ref_id: str,
        timestamp_ny: str,
        session: Optional[str] = None,
        tick_bid: Optional[float] = None,
        tick_ask: Optional[float] = None,
        bar_spread_raw: Optional[float] = None,
        realized_spread: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Builds the canonical SPREAD_OBS event payload.

        Units: price_points_usd (e.g., 0.35 USD/oz for XAUUSD).
        Source hierarchy:
          1. Live tick bid/ask if both present and valid (ask > bid).
          2. Explicit realized_spread if provided.
          3. Bar spread field converted with symbol point size.
          4. Assumed spread fallback (0.35 pts) if no data provided.
        """
        bid = None
        ask = None
        source = "assumed_default"

        if tick_bid is not None and tick_ask is not None and tick_ask >= tick_bid:
            bid = round(float(tick_bid), 3)
            ask = round(float(tick_ask), 3)
            spread_pts = round(ask - bid, 4)
            source = "TICK"
        elif realized_spread is not None:
            spread_pts = round(float(realized_spread), 4)
            source = "BAR"
        elif bar_spread_raw is not None:
            raw = float(bar_spread_raw)
            # If bar spread is in broker points (e.g. 35 for 0.35 pts), scale by point_size
            if raw > 2.0:
                spread_pts = round(raw * self.point_size, 4)
            else:
                spread_pts = round(raw, 4)
            source = "BAR"
        else:
            spread_pts = round(self.assumed_spread_pts, 4)
            source = "BAR"

        diff_pts = round(spread_pts - self.assumed_spread_pts, 4)

        return {
            "trigger_event": trigger_event,
            "ref_id": ref_id,
            "timestamp_ny": timestamp_ny,
            "session": session or "Outside Killzone",
            "bid": bid,
            "ask": ask,
            "realized_spread_pts": spread_pts,
            "assumed_spread_pts": self.assumed_spread_pts,
            "spread_diff_pts": diff_pts,
            "point_size": self.point_size,
            "units": SPREAD_UNITS,
            "source": source
        }

    def generate_spread_report(
        self,
        log_source: Union[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Parses forward event log, extracts SPREAD_OBS events, and computes
        execution metrics (mean, median, p95, worst) split by London / NY AM.

        Strictly ignores all GAP records and non-spread records.
        """
        records: List[Dict[str, Any]] = []

        if isinstance(log_source, str):
            if not os.path.exists(log_source):
                return self._empty_report()
            with open(log_source, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        records.append(rec)
                    except json.JSONDecodeError:
                        continue
        else:
            records = log_source

        # Filter strictly for SPREAD_OBS events (ignores GAP, GENESIS, SIGNAL, FILL, etc.)
        spread_events = []
        for r in records:
            if r.get("event") == "SPREAD_OBS":
                data = r.get("data", {})
                spread_events.append(data)

        if not spread_events:
            return self._empty_report()

        # Split observations by session
        all_spreads = [e["realized_spread_pts"] for e in spread_events if "realized_spread_pts" in e]
        london_spreads = [
            e["realized_spread_pts"] for e in spread_events
            if e.get("session") == "London" and "realized_spread_pts" in e
        ]
        ny_am_spreads = [
            e["realized_spread_pts"] for e in spread_events
            if e.get("session") == "NY AM" and "realized_spread_pts" in e
        ]

        # Trigger breakdown
        signal_spreads = [
            e["realized_spread_pts"] for e in spread_events
            if e.get("trigger_event") == "SIGNAL" and "realized_spread_pts" in e
        ]
        fill_spreads = [
            e["realized_spread_pts"] for e in spread_events
            if e.get("trigger_event") == "FILL" and "realized_spread_pts" in e
        ]

        report = {
            "symbol": self.symbol,
            "assumed_spread_pts": self.assumed_spread_pts,
            "units": SPREAD_UNITS,
            "total_observations": len(spread_events),
            "by_trigger": {
                "signal_count": len(signal_spreads),
                "fill_count": len(fill_spreads)
            },
            "overall": self._calculate_metrics(all_spreads),
            "london": self._calculate_metrics(london_spreads),
            "ny_am": self._calculate_metrics(ny_am_spreads),
            "notes": (
                "Slippage is not measurable in paper forward mode; realized spread observation only. "
                "Cost model and net R are frozen and unaffected."
            )
        }

        return report

    def _calculate_metrics(self, values: List[float]) -> Dict[str, Any]:
        """Calculates count, mean, median, p95, worst and diff vs assumed."""
        if not values:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "p95": None,
                "worst": None,
                "diff_vs_assumed": {
                    "mean_diff": None,
                    "median_diff": None,
                    "p95_diff": None,
                    "worst_diff": None
                }
            }

        arr = np.array(values, dtype=float)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        p95_val = float(np.percentile(arr, 95))
        worst_val = float(np.max(arr))

        return {
            "count": len(values),
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "p95": round(p95_val, 4),
            "worst": round(worst_val, 4),
            "diff_vs_assumed": {
                "mean_diff": round(mean_val - self.assumed_spread_pts, 4),
                "median_diff": round(median_val - self.assumed_spread_pts, 4),
                "p95_diff": round(p95_val - self.assumed_spread_pts, 4),
                "worst_diff": round(worst_val - self.assumed_spread_pts, 4)
            }
        }

    def _empty_report(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "assumed_spread_pts": self.assumed_spread_pts,
            "units": SPREAD_UNITS,
            "total_observations": 0,
            "by_trigger": {"signal_count": 0, "fill_count": 0},
            "overall": self._calculate_metrics([]),
            "london": self._calculate_metrics([]),
            "ny_am": self._calculate_metrics([]),
            "notes": (
                "Slippage is not measurable in paper forward mode; realized spread observation only. "
                "Cost model and net R are frozen and unaffected."
            )
        }

    def format_report_text(self, report: Dict[str, Any]) -> str:
        """Formats the structured spread report as a clean human-readable table."""
        lines = []
        lines.append("=" * 72)
        lines.append(f"EXECUTION QUALITY REPORT: REALIZED SPREAD vs ASSUMED ({report.get('symbol', 'XAUUSD')})")
        lines.append(f"Assumed Spread: {report.get('assumed_spread_pts', 0.35):.2f} {report.get('units', 'pts')} | Total Observations: {report.get('total_observations', 0)}")
        lines.append("-" * 72)
        lines.append(f"{'Session':<12} | {'Count':<6} | {'Mean':<8} | {'Median':<8} | {'P95':<8} | {'Worst':<8} | {'Diff(Mean)':<10}")
        lines.append("-" * 72)

        for name, key in [("Overall", "overall"), ("London", "london"), ("NY AM", "ny_am")]:
            m = report.get(key, {})
            cnt = m.get("count", 0)
            if cnt > 0:
                mean_s = f"{m['mean']:.4f}"
                med_s = f"{m['median']:.4f}"
                p95_s = f"{m['p95']:.4f}"
                wrst_s = f"{m['worst']:.4f}"
                diff_s = f"{m['diff_vs_assumed']['mean_diff']:+.4f}"
            else:
                mean_s = med_s = p95_s = wrst_s = diff_s = "N/A"
            lines.append(f"{name:<12} | {cnt:<6} | {mean_s:<8} | {med_s:<8} | {p95_s:<8} | {wrst_s:<8} | {diff_s:<10}")

        lines.append("-" * 72)
        lines.append(f"Triggers: SIGNAL={report.get('by_trigger', {}).get('signal_count', 0)}, FILL={report.get('by_trigger', {}).get('fill_count', 0)}")
        lines.append(f"Note: {report.get('notes', '')}")
        lines.append("=" * 72)
        return "\n".join(lines)


def generate_spread_report(log_path: str) -> Dict[str, Any]:
    """Convenience functional interface to generate spread report from log file."""
    tracker = ExecutionTracker()
    return tracker.generate_spread_report(log_path)
