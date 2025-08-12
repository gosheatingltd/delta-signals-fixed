import asyncio
import pandas as pd
import streamlit as st
import pytz
from datetime import datetime

from src.utils import load_yaml
from src.data_sources import get_klines, coingecko_price
from src.features_5m import build_features_5m
from src.signals_5m import combine_signals_5m, ScalperConfig

# Load config
@st.cache_data
def load_config():
    return load_yaml("config.yaml")

cfg = load_config()
min_gap = cfg["scalper"]["min_minutes_between_signals"]
max_day = cfg["scalper"]["max_signals_per_day"]

async def fetch_symbol(session, sym: str):
    gecko_map = {
        "BTCUSDT": ["bitcoin"],
        "ETCUSDT": ["ethereum-classic"]
    }

    # latest klines (5m) and features
    ohlc = await get_klines(
        session,
        sym,
        interval=cfg["intraday"]["interval"],
        days=cfg["intraday"]["lookback_days"]
    )

    feats = build_features_5m(ohlc)
    scfg = ScalperConfig(
        min_minutes_between_signals=int(min_gap),
        max_signals_per_day=int(max_day)
    )
    sigs = combine_signals_5m(feats, scfg)

    # price cross-verify
    gecko_px = None
    try:
        cg = await coingecko_price(session, gecko_map[sym])
        if isinstance(cg, dict) and len(cg):
            val = next(iter(cg.values()))
            if isinstance(val, dict):
                gecko_px = float(val.get("usd")) if val.get("usd") is not None else None
            else:
                gecko_px = float(val)
    except Exception:
        gecko_px = None

    last_close = float(feats["close"].iloc[-1])
    diff_bps = (
        abs(last_close - gecko_px) / ((last_close + gecko_px) / 2) * 10000
        if gecko_px is not None else None
    )

    # UK timezone adjustment
    uk_tz = pytz.timezone("Europe/London")
    last_ts_uk = feats.index[-1].tz_localize(pytz.utc).astimezone(uk_tz)

    # latest signal row
    last_sig = int(sigs.iloc[-1]) if len(sigs) else 0
    direction = "LONG" if last_sig > 0 else ("SHORT" if last_sig < 0 else "FLAT")
    latest = {
        "symbol": sym,
        "signal": direction,
        "price_binance": round(float(last_close), 2),
        "price_coingecko": round(float(gecko_px), 2) if gecko_px is not None else None,
        "price_diff_bps": round(float(diff_bps), 1) if diff_bps is not None else None,
        "rsi14": round(float(feats["rsi14"].iloc[-1]), 1),
        "ema9_minus_21": round(
            float(feats["ema_fast"].iloc[-1] - feats["ema_slow"].iloc[-1]), 2
        ),
        "timestamp": last_ts_uk.strftime("%Y-%m-%d %H:%M %Z")
    }

    # last 30 non-zero signals
    nonzero = sigs[sigs != 0].tail(30)
    hist = pd.DataFrame({
        "time": [t.strftime("%Y-%m-%d %H:%M") for t in nonzero.index],
        "signal": ["LONG" if v > 0 else "SHORT" for v in nonzero.values]
    })

    return latest, feats.tail(200)[["close", "ema_fast", "ema_slow", "rsi14"]], hist


async def run():
    import aiohttp
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_symbol(session, sym) for sym in ["BTCUSDT", "ETCUSDT"]]
        results = await asyncio.gather(*tasks)
    return results

st.set_page_config(page_title="⚡ Crypto Edge — 5-Minute Signals", layout="wide")

st.title("⚡ Crypto Edge — 5-Minute Signals (BTC & ETC)")
st.caption("Public data only • Cross-verified prices • Throttled to ~10–20 signals/day")

try:
    results = asyncio.run(run())

    for latest, feats_small, hist in results:
        st.subheader(f"{latest['symbol']} — {latest['signal']}")
        st.write(f"**Binance Price:** {latest['price_binance']}")
        st.write(f"**CoinGecko Price:** {latest['price_coingecko']}")
        st.write(f"**Price Diff (bps):** {latest['price_diff_bps']}")
        st.write(f"**RSI-14:** {latest['rsi14']}")
        st.write(f"**EMA9 − EMA21:** {latest['ema9_minus_21']}")
        st.write(f"**Timestamp:** {latest['timestamp']}")

        st.line_chart(feats_small)

        st.markdown("**Recent Non-Zero Signals:**")
        st.table(hist)

except Exception as e:
    st.error(f"Error: {e}")
