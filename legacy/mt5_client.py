import os
import pandas as pd
from datetime import datetime

try:
    import zoneinfo
    NY_TZ = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    try:
        from backports import zoneinfo
        NY_TZ = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        import datetime as dt
        NY_TZ = dt.timezone.utc

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DEFAULT_MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

def is_mt5_available():
    """Checks if MetaTrader5 python library can be imported."""
    try:
        import MetaTrader5 as mt5
        return True
    except ImportError:
        return False

def fetch_mt5_gold_candles(
    login=None,
    password=None,
    server=None,
    symbol=None,
    timeframe="M15",
    count=50,
    terminal_path=None
):
    """
    Connects to MetaTrader 5 desktop terminal, authorizes the account,
    and pulls historical 15m candlestick bars.
    Returns: (df, latest_price, feed_name, error_msg)
    """
    if login is None:
        login = os.environ.get("MT5_LOGIN", 0)
    if password is None:
        password = os.environ.get("MT5_PASSWORD", "")
    if server is None:
        server = os.environ.get("MT5_SERVER", "MetaQuotes-Demo")
    if symbol is None:
        symbol = os.environ.get("MT5_SYMBOL", "XAUUSD")

    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None, 0.0, "MetaTrader 5", "MetaTrader5 Python library is not installed or running in a non-Windows environment (e.g., Linux Cloud)."

    # 1. Resolve terminal path
    path = terminal_path or DEFAULT_MT5_PATH
    if not os.path.exists(path):
        # Search common installation paths
        common_paths = [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\MetaTrader 5\terminal64.exe")
        ]
        found = False
        for p in common_paths:
            if os.path.exists(p):
                path = p
                found = True
                break
        if not found:
            path = None  # let mt5 look in registry / running processes

    # 2. Initialize MT5 with credentials
    init_kwargs = {
        "login": int(login),
        "password": str(password),
        "server": str(server),
        "timeout": 20000
    }
    if path:
        init_kwargs["path"] = path

    initialized = mt5.initialize(**init_kwargs)
    if not initialized:
        # Try without credentials first, then login explicitly
        if path:
            initialized = mt5.initialize(path=path, timeout=15000)
        else:
            initialized = mt5.initialize(timeout=15000)

        if not initialized:
            err = mt5.last_error()
            return None, 0.0, "MetaTrader 5", f"MT5 terminal initialization failed: {err}"

        # Explicit login
        authorized = mt5.login(login=int(login), password=str(password), server=str(server))
        if not authorized:
            err = mt5.last_error()
            mt5.shutdown()
            return None, 0.0, "MetaTrader 5", f"MT5 login failed for account {login} on {server}: {err}"

    # 3. Detect symbol
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    target_tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M15)

    # Check available symbols if requested symbol is missing
    target_sym = symbol
    selected = mt5.symbol_select(target_sym, True)
    if not selected:
        all_syms = mt5.symbols_get()
        if all_syms:
            gold_syms = [s.name for s in all_syms if "XAU" in s.name.upper() or "GOLD" in s.name.upper()]
            if gold_syms:
                target_sym = gold_syms[0]
                mt5.symbol_select(target_sym, True)

    # 4. Copy rates
    rates = mt5.copy_rates_from_pos(target_sym, target_tf, 0, count)
    acc_info = mt5.account_info()
    server_name = acc_info.server if acc_info else server
    acc_login = acc_info.login if acc_info else login

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        return None, 0.0, f"MT5 ({target_sym})", f"No rates returned for {target_sym} on timeframe {timeframe}."

    rows = []
    for r in rates:
        ts = pd.to_datetime(r["time"], unit="s", utc=True).tz_convert(NY_TZ)
        rows.append({
            "Date_NY": ts,
            "Open": round(float(r["open"]), 2),
            "High": round(float(r["high"]), 2),
            "Low": round(float(r["low"]), 2),
            "Close": round(float(r["close"]), 2),
            "Volume": int(r.get("tick_volume", 0)) if hasattr(r, "get") else int(r["tick_volume"])
        })

    df = pd.DataFrame(rows).sort_values("Date_NY").reset_index(drop=True)
    cur_price = df.iloc[-1]["Close"]
    feed_title = f"MT5 ({target_sym} | Acc: {acc_login} @ {server_name})"

    return df, cur_price, feed_title, None
