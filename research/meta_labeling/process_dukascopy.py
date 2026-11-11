"""
Merge Dukascopy raw CSV chunks into data/xauusd_15m_long.csv,
clean, deduplicate, sort chronologically, and run sanity checks.
"""

import os
import glob
import pandas as pd
import numpy as np
import zoneinfo

def merge_and_validate(raw_dir="data/raw", output_path="data/xauusd_15m_long.csv"):
    files = sorted(glob.glob(os.path.join(raw_dir, "xauusd-m15-*.csv")))
    if not files:
        print(f"[ERROR] No raw CSV files found in {raw_dir}")
        return None

    print(f"Found {len(files)} chunk files to merge:")
    dfs = []
    for f in files:
        print(f"  - Reading {f}...")
        df_chunk = pd.read_csv(f)
        dfs.append(df_chunk)

    df_all = pd.concat(dfs, ignore_index=True)
    initial_count = len(df_all)
    print(f"\nTotal combined raw rows: {initial_count:,}")

    # Standardize column names
    df_all.columns = [c.strip().lower() for c in df_all.columns]

    # Handle timestamp: Dukascopy gives epoch ms
    if "timestamp" in df_all.columns:
        # Check if epoch ms
        if df_all["timestamp"].iloc[0] > 1e11:
            df_all["datetime_utc"] = pd.to_datetime(df_all["timestamp"], unit="ms", utc=True)
        else:
            df_all["datetime_utc"] = pd.to_datetime(df_all["timestamp"], unit="s", utc=True)
    elif "date" in df_all.columns and "time" in df_all.columns:
        df_all["datetime_utc"] = pd.to_datetime(df_all["date"].astype(str) + " " + df_all["time"].astype(str), utc=True)
    else:
        # Try parsing first column
        df_all["datetime_utc"] = pd.to_datetime(df_all.iloc[:, 0], utc=True)

    # Drop duplicates by datetime
    df_all = df_all.drop_duplicates(subset=["datetime_utc"])
    dedup_count = len(df_all)
    print(f"Rows after dropping duplicates: {dedup_count:,} (dropped {initial_count - dedup_count:,} duplicates)")

    # Sort chronologically
    df_all = df_all.sort_values(by="datetime_utc").reset_index(drop=True)

    # Filter OHLC columns
    req = ["open", "high", "low", "close"]
    for c in req:
        if c not in df_all.columns:
            raise ValueError(f"Missing OHLC column: {c}")

    if "volume" not in df_all.columns:
        df_all["volume"] = 0.0

    # Ensure correct types
    for c in req + ["volume"]:
        df_all[c] = df_all[c].astype(float)

    # Format timestamp column for output (standardized datetime string in UTC)
    # Output schema: timestamp, open, high, low, close, volume
    out_df = pd.DataFrame({
        "timestamp": df_all["datetime_utc"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "open": df_all["open"],
        "high": df_all["high"],
        "low": df_all["low"],
        "close": df_all["close"],
        "volume": df_all["volume"]
    })

    out_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Saved merged dataset to: {output_path} ({len(out_df):,} rows)")

    # -------------------------------------------------------------
    # SANITY & TIMEZONE REPORT
    # -------------------------------------------------------------
    first_dt = df_all["datetime_utc"].iloc[0]
    last_dt = df_all["datetime_utc"].iloc[-1]
    print("\n" + "=" * 65)
    print("DUKASCOPY MERGED DATASET REPORT")
    print("=" * 65)
    print(f"First Candle (UTC): {first_dt}")
    print(f"Last Candle (UTC):  {last_dt}")
    print(f"Total Candles:      {len(out_df):,}")

    # Gap check in UTC
    dt_series = df_all["datetime_utc"]
    diffs = dt_series.diff()
    normal_15m = pd.Timedelta(minutes=15)
    gaps = diffs[diffs > normal_15m]
    weekend_gaps = []
    unusual_gaps = []
    for idx, delta in gaps.items():
        prev_time = dt_series.loc[idx - 1]
        curr_time = dt_series.loc[idx]
        # Weekend is Friday evening to Sunday evening
        if prev_time.weekday() == 4 and curr_time.weekday() in (6, 0):
            weekend_gaps.append((prev_time, curr_time, delta))
        else:
            unusual_gaps.append((prev_time, curr_time, delta))

    print(f"Expected Weekend Gaps: {len(weekend_gaps)}")
    print(f"Unusual Intraday Gaps: {len(unusual_gaps)}")
    if unusual_gaps:
        print("  Sample unusual gaps (first 5):")
        for p, c, d in unusual_gaps[:5]:
            print(f"    - From {p} to {c} (duration: {d})")

    # -------------------------------------------------------------
    # 17:00-18:00 NY ZERO-CANDLE TEST (CONVERTING UTC -> AMERICA/NEW_YORK)
    # -------------------------------------------------------------
    ny_tz = zoneinfo.ZoneInfo("America/New_York")
    dt_ny = dt_series.dt.tz_convert(ny_tz)
    weekdays_mask = dt_ny.dt.weekday < 5
    ny_weekday_hours = dt_ny[weekdays_mask].dt.hour

    hour_dist = ny_weekday_hours.value_counts().sort_index()
    zero_candle_hours = [h for h in range(24) if h not in hour_dist or hour_dist[h] == 0]

    print("\n" + "-" * 65)
    print("17:00-18:00 NEW YORK ZERO-CANDLE SANITY TEST (SOURCE TZ = UTC)")
    print("-" * 65)
    print(f"Zero-candle hours across all weekday bars: {zero_candle_hours}")
    print(f"Hour 17 (17:00-17:59 NY) candle count:    {hour_dist.get(17, 0)}")
    print(f"Hour 18 (18:00-18:59 NY) candle count:    {hour_dist.get(18, 0)}")
    print(f"Hour 19 (19:00-19:59 NY) candle count:    {hour_dist.get(19, 0)}")
    print(f"Hour 02 (London Open NY) candle count:    {hour_dist.get(2, 0)}")
    print(f"Hour 07 (NY AM Open NY)  candle count:    {hour_dist.get(7, 0)}")
    print("=" * 65)

    return out_df

if __name__ == "__main__":
    merge_and_validate()
