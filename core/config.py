import os
from datetime import time

# -----------------------------------------------------------------------------
# XAUUSD ICT BASELINE STRATEGY CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PATH = os.environ.get("DATA_PATH", os.path.join("data", "xauusd_15m.csv"))
DATA_TZ = os.environ.get("DATA_TZ", "Europe/Athens")
TARGET_TZ = "America/New_York"

# Causality & Structure Parameters
SWING_LEN = 5  # Confirmed swing requires SWING_LEN closed candles after the pivot
SL_BUFFER = 1.5  # Price buffer beyond sweep extreme
RR = 2.0  # Take Profit 1:2 R:R
ORDER_EXPIRY_BARS = 8  # Limit order expires after 8 bars if unfilled
MIN_GAP_PTS = 0.35  # Minimum FVG gap height in price units
FVG_MIN_GAP = 0.35  # Alias for signals.py
DISPLACEMENT_BODY_RATIO = 0.50  # Minimum body-to-range ratio for displacement candle

# Execution Costs & Conservative Slippage
SPREAD = 0.35  # Bid/Ask spread (0.35 USD / oz)
SLIPPAGE = 0.10  # Stop-loss execution slippage (0.10 USD / oz)
COMMISSION_PER_LOT = 7.0  # Round-turn commission per standard lot in USD
MIN_SL = 3.0  # Minimum stop distance in price units to avoid cost-dominated noise

# Account & Sizing
INITIAL_EQUITY = 10000.0  # Starting capital in USD
RISK_PCT = 0.01  # Risk 1% of equity per trade
RISK_PERCENT = 0.01  # Alias for backtester.py
CONTRACT_SIZE = 100.0  # 1 standard lot = 100 oz of Gold (1.00 USD price move = 100 USD / lot)

# Splits (Anti-Overfitting)
TRAIN_START = "2015-01-01"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2025-12-31"

# In-sample / Out-of-sample fallback for recent broker data
RECENT_TRAIN_START = "2024-09-18"
RECENT_TRAIN_END = "2025-12-31"
RECENT_TEST_START = "2026-01-01"
RECENT_TEST_END = "2026-11-10"

TEST_RUNS_LOG = "test_runs.log"

# Killzone Windows (New York Time)
KILLZONES = {
    "LONDON": (time(2, 0), time(5, 0)),
    "NY_AM": (time(7, 0), time(10, 0)),
}
