import asyncio, aiohttp
from typing import Any, Dict
import pandas as pd
from datetime import datetime, timezone

BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_API = "https://api.binance.com"

async def fetch_json(session: aiohttp.ClientSession, url: str, params: Dict[str, Any] = None, retries=3) -> Any:
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, headers=headers, timeout=20) as r:
                if r.status >= 400:
                    return None
                return await r.json()
        except Exception:
            if attempt == retries - 1:
                return None
            await asyncio.sleep(0.5 * (attempt + 1))

import pandas as pd
from datetime import datetime, timezone

BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_API  = "https://api.binance.com"
KRAKEN_API   = "https://api.kraken.com"

# existing fetch_json stays the same

async def _kraken_ohlc(session, symbol: str, interval: str = "5m", lookback_days: int = 7) -> pd.DataFrame:
    """Fetch OHLC from Kraken as a fallback. Maps BTCUSDT->XBTUSD, ETCUSDT->ETCUSD."""
    # map Binance-style symbols to Kraken pairs
    kr_map = {"BTCUSDT": "XBTUSD", "ETCUSDT": "ETCUSD"}
    pair = kr_map.get(symbol)
    if not pair:
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time","num_trades"])

    # Kraken interval in minutes
    interval_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(interval, 5)

    url = f"{KRAKEN_API}/0/public/OHLC"
    params = {"pair": pair, "interval": interval_min}
    data = await fetch_json(session, url, params)
    if not data or "result" not in data:
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time","num_trades"])

    # result key is the pair (e.g., 'XXBTZUSD' or 'XBTUSD'); grab the first array
    result = data["result"]
    # pick the first list in result that is a list of rows
    rows = None
    for k, v in result.items():
        if isinstance(v, list) and v and isinstance(v[0], list):
            rows = v
            break
    if not rows:
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time","num_trades"])

    # Kraken columns: [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(rows, columns=[
        "t","open","high","low","close","vwap","volume","count"
    ])
    df["open_time"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["close_time"] = df["open_time"]  # we don't need exact close ms for plotting/signals
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["num_trades"] = df["count"].astype(int)
    df = df[["open_time","open","high","low","close","volume","close_time","num_trades"]]
    return df.tail(2000).reset_index(drop=True)  # limit to a reasonable window

async def get_klines(session, symbol: str, interval: str = "5m", days: int = 7):
    """
    Robust kline downloader:
    1) Binance Futures /fapi/v1/klines
    2) Binance ContinuousKlines
    3) Binance Spot /api/v3/klines
    4) Kraken OHLC fallback (XBTUSD/ETCUSD)
    """
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = end - days * 24 * 60 * 60 * 1000
    frames = []
    cur = start
    CHUNK_MS = 1500 * 60 * 60 * 1000

    while cur < end:
        chunk_end = min(cur + CHUNK_MS, end)

        # 1) Futures
        url1 = f"{BINANCE_FAPI}/fapi/v1/klines"
        p1 = {"symbol": symbol, "interval": interval, "limit": 1500, "startTime": cur, "endTime": chunk_end}
        raw = await fetch_json(session, url1, p1)

        # 2) Continuous
        if not raw:
            url2 = f"{BINANCE_FAPI}/fapi/v1/continuousKlines"
            p2 = {"pair": symbol, "contractType": "PERPETUAL", "interval": interval, "limit": 1500, "startTime": cur, "endTime": chunk_end}
            raw = await fetch_json(session, url2, p2)

        # 3) Spot
        if not raw:
            url3 = f"{BINANCE_API}/api/v3/klines"
            p3 = {"symbol": symbol, "interval": interval, "limit": 1000}
            raw = await fetch_json(session, url3, p3)

        if raw:
            df = pd.DataFrame(raw, columns=[
                "open_time","open","high","low","close","volume","close_time",
                "qav","num_trades","taker_base","taker_quote","ignore"
            ])
            if not df.empty:
                df = df[["open_time","open","high","low","close","volume","close_time","num_trades"]].copy()
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                for c in ["open","high","low","close","volume"]:
                    df[c] = df[c].astype(float)
                frames.append(df)
                cur = int(df["close_time"].iloc[-1]) + 1
                continue

        # 4) Kraken fallback (fetch once; no chunking)
        kdf = await _kraken_ohlc(session, symbol, interval=interval, lookback_days=days)
        if not kdf.empty:
            return kdf  # good enough to drive the UI/signals

        # Nothing worked for this chunk; break to avoid infinite loop
        break

    if frames:
        out = pd.concat(frames).drop_duplicates(subset=["open_time"]).reset_index(drop=True)
        return out

    # empty skeleton with expected columns
    return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time","num_trades"])


async def coingecko_price(session, ids):
    url = "https://api.coingecko.com/api/v3/simple/price"
    if isinstance(ids, (list, tuple)):
        ids_list = list(ids)
    else:
        ids_list = [ids]
    params = {"ids": ",".join(ids_list), "vs_currencies": "usd"}
    data = await fetch_json(session, url, params)
    out = {}
    if isinstance(data, dict):
        for k in ids_list:
            v = data.get(k)
            if isinstance(v, dict) and "usd" in v and v["usd"] is not None:
                try:
                    out[k] = float(v["usd"])
                except Exception:
                    pass
    return out
