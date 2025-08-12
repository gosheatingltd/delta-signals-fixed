# ==============================
# Safe OHLC fetcher: project -> Binance (multi-host Spot) -> Binance (multi-host Futures)
# ==============================
import aiohttp

INTERVAL_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1d"
}

BINANCE_SPOT_HOSTS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",
]
BINANCE_FUTURES_HOSTS = [
    "https://fapi.binance.com",
    "https://fstream.binance.com",  # alt futures domain
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0; +https://streamlit.io)"
}

async def _http_json(session: aiohttp.ClientSession, url: str, params: dict):
    try:
        async with session.get(url, params=params, headers=HEADERS, timeout=20) as r:
            if r.status != 200:
                return None, r.status
            return await r.json(), r.status
    except Exception:
        return None, None

def _klines_json_to_df(data):
    # Binance kline array schema
    if not data or isinstance(data, dict):
        return pd.DataFrame()
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","n","tbbav","tbqav","ignore"]
    try:
        df = pd.DataFrame(data, columns=cols)
    except Exception:
        return pd.DataFrame()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["open_time","open","high","low","close","volume"]]

async def _try_binance_group(session, hosts, path, symbol, interval, days):
    iv = INTERVAL_MAP.get(interval, "5m")
    # keep limit modest to avoid 413 / rate limits
    per_day_5m = 12 * 24
    limit = min(1000, max(200, days * per_day_5m))
    params = {"symbol": symbol, "interval": iv, "limit": limit}

    for host in hosts:
        url = f"{host}{path}"
        data, status = await _http_json(session, url, params)
        df = _klines_json_to_df(data)
        if not df.empty:
            return df, host
    return pd.DataFrame(), None

async def _binance_spot_any(session, symbol, interval, days):
    # /api/v3/klines on multiple hosts
    return await _try_binance_group(session, BINANCE_SPOT_HOSTS, "/api/v3/klines", symbol, interval, days)

async def _binance_futures_any(session, symbol, interval, days):
    # /fapi/v1/klines on multiple futures hosts
    return await _try_binance_group(session, BINANCE_FUTURES_HOSTS, "/fapi/v1/klines", symbol, interval, days)

async def get_klines_safe(session: aiohttp.ClientSession, symbol: str, interval: str, days: int):
    """
    Try project's get_klines(); if empty, rotate across multiple Binance Spot hosts,
    then multiple Binance Futures hosts. Returns (df, source_name).
    """
    # 0) shrink days if very large to avoid oversized responses
    days_try = max(1, min(days, 7))

    # 1) project source
    try:
        df = await get_klines(session, symbol, interval=interval, days=days_try)
        if df is not None and not df.empty and "open_time" in df.columns:
            return df, "project"
    except Exception:
        pass

    # 2) Binance Spot (multi-host)
    df_spot, host_spot = await _binance_spot_any(session, symbol, interval, days_try)
    if not df_spot.empty:
        return df_spot, f"binance_spot ({host_spot})"

    # 3) Binance Futures (multi-host)
    df_fut, host_fut = await _binance_futures_any(session, symbol, interval, days_try)
    if not df_fut.empty:
        return df_fut, f"binance_futures ({host_fut})"

    # 4) If all failed and days>1, try a tiny lookback as last resort
    if days_try > 1:
        df_spot, host_spot = await _binance_spot_any(session, symbol, interval, 1)
        if not df_spot.empty:
            return df_spot, f"binance_spot_1d ({host_spot})"
        df_fut, host_fut = await _binance_futures_any(session, symbol, interval, 1)
        if not df_fut.empty:
            return df_fut, f"binance_futures_1d ({host_fut})"

    return pd.DataFrame(), "none"
