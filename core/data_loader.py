"""
Data Loader module for XAUUSD 15m historical data.
Handles schema normalization, DST-aware timezone conversion to America/New_York,
duplicate removal, sorting, gap detection, and OHLC data integrity validation.
"""

import os
import zoneinfo
import numpy as np
import pandas as pd
try:
    from core import config
except ImportError:
    import config


def load_and_clean_data(file_path=None, data_tz=None, target_tz=None, verbose=True):
    """
    Loads historical 15m data, normalises schema to lowercase OHLCV with datetime index,
    converts timezone to America/New_York with strict DST handling,
    drops duplicates, sorts, validates OHLC, and logs quality metrics.

    Returns:
        pd.DataFrame: Cleaned dataframe with index in target_tz (America/New_York)
                      and columns: ['open', 'high', 'low', 'close', 'volume'].
    """
    path = file_path or config.DATA_PATH
    src_tz = data_tz or config.DATA_TZ
    tgt_tz = target_tz or config.TARGET_TZ

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")

    # 1. Read CSV
    # Detect separator (comma or semicolon)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","

    df = pd.read_csv(path, sep=sep)

    # 2. Normalize column names (strip whitespace and convert to lowercase)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 3. Detect and construct datetime series (handles strings, ISO, and epoch milliseconds/seconds)
    def parse_flexible_datetime(series):
        try:
            numeric_series = pd.to_numeric(series, errors="coerce")
            if numeric_series.notna().sum() > len(series) * 0.8:
                valid_num = numeric_series.dropna()
                first_val = float(valid_num.iloc[0])
                if first_val > 1e11:  # Epoch milliseconds (e.g. Dukascopy timestamp in ms)
                    return pd.to_datetime(numeric_series, unit="ms", errors="coerce")
                elif first_val > 1e8:  # Epoch seconds
                    return pd.to_datetime(numeric_series, unit="s", errors="coerce")
        except Exception:
            pass
        return pd.to_datetime(series, errors="coerce")

    # Match common Dukascopy and institutional datetime column conventions
    time_col = None
    for candidate in ["gmt time", "gmt_time", "local time", "datetime", "timestamp", "time (utc)", "ts"]:
        if candidate in df.columns:
            time_col = candidate
            break

    if "date" in df.columns and "time" in df.columns:
        dt_series = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    elif time_col is not None:
        dt_series = parse_flexible_datetime(df[time_col])
    elif "date" in df.columns:
        dt_series = parse_flexible_datetime(df["date"])
    elif "time" in df.columns:
        dt_series = parse_flexible_datetime(df["time"])
    else:
        # Check if first column can be parsed as datetime or epoch
        dt_series = parse_flexible_datetime(df.iloc[:, 0])

    # Drop rows where datetime failed to parse
    valid_dt_mask = dt_series.notna()
    if not valid_dt_mask.all():
        dropped_dt = (~valid_dt_mask).sum()
        df = df[valid_dt_mask].copy()
        dt_series = dt_series[valid_dt_mask]
    else:
        dropped_dt = 0

    # 4. Standardize OHLCV columns
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ["open", "o"]:
            col_map[col] = "open"
        elif cl in ["high", "h"]:
            col_map[col] = "high"
        elif cl in ["low", "l"]:
            col_map[col] = "low"
        elif cl in ["close", "c"]:
            col_map[col] = "close"
        elif cl in ["volume", "vol", "tick_volume", "v"]:
            col_map[col] = "volume"

    df = df.rename(columns=col_map)
    required_cols = ["open", "high", "low", "close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}. Available: {list(df.columns)}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df.index = dt_series

    # 5. Timezone conversion with DST handling
    # If naive, localize to source timezone; if already tz-aware, convert
    if df.index.tz is None:
        try:
            df.index = df.index.tz_localize(src_tz, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            # Fallback for complex DST seams
            df.index = df.index.tz_localize(src_tz, ambiguous="NaT", nonexistent="shift_forward")
            df = df[df.index.notna()]

    # Convert to target timezone (America/New_York)
    df.index = df.index.tz_convert(tgt_tz)
    df.index.name = "datetime_ny"

    # 6. Drop duplicate timestamps and sort chronologically
    initial_rows = len(df)
    duplicates_count = df.index.duplicated(keep="last").sum()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # 7. Validate OHLC Integrity (high >= max(open, close), low <= min(open, close))
    body_max = df[["open", "close"]].max(axis=1)
    body_min = df[["open", "close"]].min(axis=1)
    invalid_high_mask = df["high"] < body_max
    invalid_low_mask = df["low"] > body_min

    invalid_high_count = int(invalid_high_mask.sum())
    invalid_low_count = int(invalid_low_mask.sum())

    if invalid_high_count > 0:
        # Enforce high >= max(open, close)
        df.loc[invalid_high_mask, "high"] = body_max[invalid_high_mask]
    if invalid_low_count > 0:
        # Enforce low <= min(open, close)
        df.loc[invalid_low_mask, "low"] = body_min[invalid_low_mask]

    # 8. Gap Analysis
    time_diffs = df.index.to_series().diff()
    normal_step = pd.Timedelta(minutes=15)
    gaps_mask = time_diffs > normal_step

    weekend_gaps = []
    unusual_gaps = []

    if gaps_mask.any():
        gap_indices = df.index[gaps_mask]
        for prev_idx, cur_idx in zip(df.index[df.index.get_indexer_for(gap_indices) - 1], gap_indices):
            gap_duration = cur_idx - prev_idx
            # Weekend gap: spans across Friday evening (>= 16:30 NY) to Sunday (>= 17:00 NY)
            is_weekend = (prev_idx.weekday() == 4 and cur_idx.weekday() in [6, 0]) or (gap_duration >= pd.Timedelta(hours=40))
            # Daily roll-over break: gold markets typically close daily around 17:00-18:00 NY for 1 to 2 hours
            is_daily_break = (gap_duration <= pd.Timedelta(hours=2, minutes=30)) and (prev_idx.date() == cur_idx.date() or (cur_idx.date() - prev_idx.date()).days == 1)
            
            if is_weekend:
                weekend_gaps.append((prev_idx, cur_idx, gap_duration))
            elif not is_daily_break:
                unusual_gaps.append((prev_idx, cur_idx, gap_duration))

    # 9. Report
    if verbose:
        print("=" * 65)
        print("DATA QUALITY & NORMALIZATION REPORT")
        print("=" * 65)
        print(f"* Source File:            {path}")
        print(f"* Total Rows Loaded:      {len(df):,}")
        print(f"* Timezone Converted:     {src_tz} -> {tgt_tz} (DST Handled)")
        print(f"* Date Range (NY):        {df.index[0]} to {df.index[-1]}")
        print(f"* Duplicate Rows Dropped: {duplicates_count}")
        print(f"* Dropped Invalid Dates:  {dropped_dt}")
        print(f"* OHLC Anomalies Fixed:   High < Max(O,C): {invalid_high_count} | Low > Min(O,C): {invalid_low_count}")
        print(f"* Expected Weekend Gaps:  {len(weekend_gaps)}")
        print(f"* Unusual Intraday Gaps:  {len(unusual_gaps)}")
        if unusual_gaps:
            print("  First 3 unusual gaps:")
            for p, c, d in unusual_gaps[:3]:
                print(f"    - From {p} to {c} ({d})")
        print("=" * 65)

    return df


# Alias for backward/forward compatibility
load_data = load_and_clean_data

if __name__ == "__main__":
    df = load_and_clean_data()
    print("Clean Data Sample:")
    print(df.head())
