# --- Path setup so imports work on Streamlit Cloud ---
import os, sys
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
SRC_DIR = os.path.join(APP_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# -----------------------------------------------------

import asyncio
from pathlib import Path
import pandas as pd
import streamlit as st
import pytz
import altair as alt

from src.utils import load_yaml
from src.data_sources import get_klines, coingecko_price
from src.features_5m import build_features_5m
from src.signals_5m import combine_signals_5m, ScalperConfig

st.set_page_config(page_title="⚡ Crypto Edge — 5-Minute Signals", layout="wide")
st.title("⚡ Crypto Edge — 5-Minute Signals (BTC & ETC)")
st.caption("Public data only • Cross-verified prices • Throttled to ~10–20 signals/day")

# ==============================
# CONFIG (robust)
# ==============================
@st.cache_data(ttl=60)
def load_config():
    p = Path(APP_DIR) / "config.yaml"
    if p.exists():
        return load_yaml(str(p))
    # sane defaults so app still boots
    return {
        "symbols": ["BTCUSDT", "ETCUSDT"],
        "signals": {"min_gap_minutes": 20, "max_per_day": 20, "news_quorum_sources": 2},
        "intraday": {
            "interval": "5m",
            "lookback_days": 7,
            "min_minutes_between_signals": 20,
            "max_signals_per_day": 20,
        },
        "cross_verify": {"price_tolerance_bps": 15},
        "schedule": {"poll_interval_seconds": 30},
    }

cfg = load_config()

# derive throttles with fallbacks
min_gap = int(
    cfg.get("signals", {}).get(
        "min_gap_minutes",
        cfg.get("intraday", {}).get("min_minutes_between_signals", 20),
    )
)
max_day = int(
    cfg.get("signals", {}).get(
        "max_per_day",
        cfg.get("intraday", {}).get("max_signals_per_day", 20),
    )
)

symbols = st.multiselect(
    "Symbols", ["BTCUSDT", "ETCUSDT"], default=cfg.get("symbols", ["BTCUSDT", "ETCUSDT"])
)
colA, colB, colC = st.columns(3)
with colA:
    poll = st.number_input(
        "Refresh (seconds)",
        min_value=15,
        max_value=120,
        value=int(cfg.get("schedule", {}).get("poll_interval_seconds", 30)),
        step=5,
    )
with colB:
    min_gap = st.number_input("Min minutes between signals", 5, 120, min_gap, 5)
with colC:
    max_day = st.number_input("Max signals/day", 5, 50, max_day, 1)

# ==============================
# Helpers
# ==============================
def _empty_symbol(sym: str):
    latest = {
        "symbol": sym,
        "signal": "FLAT",
        "price_binance": None,
        "price_coingecko": None,
        "price_diff_bps": None,
        "rsi14": None,
        "ema9_minus_21": None,
        "timestamp": "",
    }
    empty_feats = pd.DataFrame({"close": [], "ema_fast": [], "ema_slow": [], "rsi14": []})
    empty_hist = pd.DataFrame({"time": [], "signal": []})
    return latest, empty_feats, empty_hist

def to_london(ts):
    uk = pytz.timezone("Europe/London")
    if getattr(ts, "tzinfo", None) is None:
        ts = pytz.utc.localize(ts)
    return ts.astimezone(uk)

# ==============================
# Per-symbol pipeline
# ==============================
async def fetch_symbol(session, sym: str):
    gecko_map = {"BTCUSDT": ["bitcoin"], "ETCUSDT": ["ethereum-classic"]}

    # 1) OHLC
    ohlc = await get_klines(
        session,
        sym,
        interval=cfg["intraday"]["interval"],
        days=int(cfg["intraday"]["lookback_days"]),
    )
    if ohlc is None or ohlc.empty or "open_time" not in ohlc.columns:
        st.warning(f"{sym}: no market data returned. Retrying on next refresh.")
        return _empty_symbol(sym)

    # 2) Features
    feats = build_features_5m(ohlc)
    if feats is None or feats.empty:
        st.warning(f"{sym}: insufficient data for features (5m).")
        return _empty_symbol(sym)

    # 3) Signals
    scfg = ScalperConfig(
        min_minutes_between_signals=int(min_gap), max_signals_per_day=int(max_day)
    )
    sigs = combine_signals_5m(feats, scfg)

    # 4) CoinGecko (robust parse)
    gecko_px = None
    try:
        cg = await coingecko_price(session, gecko_map[sym])
        if isinstance(cg, dict) and len(cg):
            val = next(iter(cg.values()))
            gecko_px = float(val["usd"]) if isinstance(val, dict) and "usd" in val else float(val)
    except Exception:
        gecko_px = None

    last_close = float(feats["close"].iloc[-1])
    diff_bps = (
        abs(last_close - gecko_px) / ((last_close + gecko_px) / 2) * 10000
        if gecko_px is not None
        else None
    )

    # 5) Latest row (UK time)
    last_ts = feats.index[-1]
    last_ts_uk = to_london(last_ts)

    last_sig = int(sigs.iloc[-1]) if len(sigs) else 0
    direction = "LONG" if last_sig > 0 else ("SHORT" if last_sig < 0 else "FLAT")
    latest = {
        "symbol": sym,
        "signal": direction,
        "price_binance": round(last_close, 2),
        "price_coingecko": None if gecko_px is None else round(float(gecko_px), 2),
        "price_diff_bps": None if diff_bps is None else round(float(diff_bps), 1),
        "rsi14": round(float(feats["rsi14"].iloc[-1]), 1),
        "ema9_minus_21": round(
            float(feats["ema_fast"].iloc[-1] - feats["ema_slow"].iloc[-1]), 2
        ),
        "timestamp": last_ts_uk.strftime("%Y-%m-%d %H:%M %Z"),
    }

    nonzero = sigs[sigs != 0].tail(30)
    hist = pd.DataFrame(
        {
            "time": [t.strftime("%Y-%m-%d %H:%M") for t in nonzero.index],
            "signal": ["LONG" if v > 0 else "SHORT" for v in nonzero.values],
        }
    )

    return latest, feats.tail(200)[["close", "ema_fast", "ema_slow", "rsi14"]], hist

# ==============================
# Main
# ==============================
async def run():
    import aiohttp
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_symbol(session, sym) for sym in symbols]
        return await asyncio.gather(*tasks)

results = asyncio.run(run())

# ==============================
# Render
# ==============================
st.subheader("Live Signals")
rows = [latest for (latest, _, _) in results]
st.dataframe(pd.DataFrame(rows), use_container_width=True)

left, right = st.columns(2)
for i, (latest, feats, hist) in enumerate(results):
    with (left if i % 2 == 0 else right):
        st.markdown(f"### {latest['symbol']}")

        # ---- PRICE CHART: Close + EMA9 + EMA21 (Altair) ----
        if not feats.empty:
            plot_df = feats[["close", "ema_fast", "ema_slow"]].copy().reset_index()
            plot_df = plot_df.rename(columns={"ts": "time"})
            long_df = plot_df.melt("time", var_name="series", value_name="value")
            name_map = {"close": "Close", "ema_fast": "EMA 9", "ema_slow": "EMA 21"}
            long_df["series"] = long_df["series"].map(name_map)

            line = (
                alt.Chart(long_df)
                .mark_line()
                .encode(
                    x=alt.X("time:T", title="Time (UK)"),
                    y=alt.Y("value:Q", title="Price (USD)"),
                    color=alt.Color(
                        "series:N",
                        title="Legend",
                        scale=alt.Scale(
                            domain=["Close", "EMA 9", "EMA 21"],
                            range=["#1f77b4", "#e45756", "#4ea8de"],  # dark blue, red, light blue
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("time:T", title="Time"),
                        alt.Tooltip("series:N", title="Series"),
                        alt.Tooltip("value:Q", title="Value", format=",.2f"),
                    ],
                )
                .properties(height=260)
                .interactive()
            )
            st.altair_chart(line, use_container_width=True)

            # RSI bar chart
            st.bar_chart(feats[["rsi14"]])

        # Recent signals table
        if not hist.empty:
            st.markdown("**Recent non-zero signals**")
            st.dataframe(hist, use_container_width=True, height=240)
