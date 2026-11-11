"""
Test script for causal feature precomputation on 15m data.
"""

import pandas as pd
import numpy as np
import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from core.data_loader import load_and_clean_data
except ImportError:
    from data_loader import load_and_clean_data

def precompute_market_features(df):
    highs = df["high"]
    lows = df["low"]
    closes = df["close"]
    prev_closes = closes.shift(1)

    # 1. ATR(14)
    tr = pd.concat([
        highs - lows,
        (highs - prev_closes).abs(),
        (lows - prev_closes).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=1).mean()

    # 2. Volatility Rank (percentile over rolling 500 bars)
    vol_rank = atr_14.rolling(window=500, min_periods=50).rank(pct=True).fillna(0.5)

    # 3. Session High / Low / Pos (Session resets at 18:00 NY)
    # Session date: if hour >= 18, it belongs to the next session date
    session_date = df.index.to_series().apply(
        lambda dt: dt.date() + pd.Timedelta(days=1) if dt.hour >= 18 else dt.date()
    )
    session_high = highs.groupby(session_date).cummax()
    session_low = lows.groupby(session_date).cummin()
    session_span = session_high - session_low
    session_pos = np.where(session_span > 1e-4, (closes - session_low) / session_span, 0.5)

    # 4. Previous Day Range
    daily_high = highs.groupby(session_date).max()
    daily_low = lows.groupby(session_date).min()
    daily_range = daily_high - daily_low
    prev_day_range_map = daily_range.shift(1)
    prev_day_range = session_date.map(prev_day_range_map).fillna(daily_range.median())

    # 5. 1H EMA(20) Trend Sign (strictly causal: computed on closed 1H bars, shifted by 1 hour)
    df_1h = df["close"].resample("1h", closed="left", label="left").last().dropna()
    ema_1h = df_1h.ewm(span=20, adjust=False).mean()
    ema_1h_slope = np.sign(ema_1h.diff()).shift(1) # shift 1 bar so only closed 1H bars are used
    trend_1h_sign = ema_1h_slope.reindex(df.index, method="ffill").fillna(0.0)

    # 6. Daily EMA(20) Trend Sign (strictly causal: computed on closed daily bars, shifted by 1 day)
    daily_close = closes.groupby(session_date).last()
    ema_daily = daily_close.ewm(span=20, adjust=False).mean()
    ema_daily_slope = np.sign(ema_daily.diff()).shift(1)
    trend_daily_sign = session_date.map(ema_daily_slope).fillna(0.0)

    features_df = pd.DataFrame({
        "atr_14": atr_14,
        "volatility_rank": vol_rank,
        "session_range_pos": session_pos,
        "prev_day_range": prev_day_range,
        "trend_1h_sign": trend_1h_sign,
        "trend_daily_sign": trend_daily_sign
    }, index=df.index)

    return features_df

if __name__ == "__main__":
    df = load_and_clean_data("data/xauusd_15m_long.csv", data_tz="UTC", target_tz="America/New_York", verbose=False)
    df_train = df.loc["2015-01-01":"2021-12-31"]
    import time
    t0 = time.time()
    feats = precompute_market_features(df_train)
    t1 = time.time()
    print(f"Features computed in {t1 - t0:.2f}s for {len(feats):,} bars")
    print(feats.describe())


def test_feature_generation():
    df = load_and_clean_data("data/xauusd_15m_long.csv", data_tz="UTC", target_tz="America/New_York", verbose=False)
    df_sample = df.iloc[:200]
    feats = precompute_market_features(df_sample)
    assert len(feats) == len(df_sample)
    assert "atr_14" in feats.columns
    assert "volatility_rank" in feats.columns
