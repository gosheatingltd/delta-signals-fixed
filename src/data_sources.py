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

async def get_klines(session, symbol: str, interval: str = "5m", days: int = 7):
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = end - days * 24 * 60 * 60 * 1000
    frames = []
    cur = start
    CHUNK_MS = 1500 * 60 * 60 * 1000

    while cur < end:
        chunk_end = min(cur + CHUNK_MS, end)

        # 1) Futures klines
        url1 = f"{BINANCE_FAPI}/fapi/v1/klines"
        p1 = {"symbol": symbol, "interval": interval, "limit": 1500, "startTime": cur, "endTime": chunk_end}
        raw = await fetch_json(session, url1, p1)

        # 2) continuousKlines fallback
        if not raw:
            url2 = f"{BINANCE_FAPI}/fapi/v1/continuousKlines"
            p2 = {"pair": symbol, "contractType": "PERPETUAL", "interval": interval, "limit": 1500,
                  "startTime": cur, "endTime": chunk_end}
            raw = await fetch_json(session, url2, p2)

        # 3) SPOT fallback
        if not raw:
            url3 = f"{BINANCE_API}/api/v3/klines"
            p3 = {"symbol": symbol, "interval": interval, "limit": 1000}
            raw = await fetch_json(session, url3, p3)

        if not raw:
            break

        df = pd.DataFrame(raw, columns=[
            "open_time","open","high","low","close","volume","close_time",
            "qav","num_trades","taker_base","taker_quote","ignore"
        ])
        if df.empty:
            break
        df = df[["open_time","open","high","low","close","volume","close_time","num_trades"]].copy()
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        frames.append(df)

        cur = int(df["close_time"].iloc[-1]) + 1

    if frames:
        out = pd.concat(frames).drop_duplicates(subset=["open_time"]).reset_index(drop=True)
        return out
    return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time","num_trades"])

async def coingecko_price(session, ids):
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": ",".join(ids), "vs_currencies": "usd"}
    data = await fetch_json(session, url, params)
    return data or {}
