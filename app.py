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
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import pytz
import altair as alt

from src.utils import load_yaml
from src.data_sources import get_klines, coingecko_price
from src.features_5m import build_features_5m
from src.signals_5m import combine_signals_5m, ScalperConfig

UK_TZ = pytz.timezone("Europe/London")

st.set_page_config(page_title="⚡ Crypto Edge — 5-Min Signals (BTC & ETH)", layout="wide")
st.title("⚡ Crypto Edge — 5-Minute Signals (BTC & ETH)")
st.caption("Public data only • Cross-verified prices • 5m timeframe • 12-hour clock")

# ==============================
# CONFIG
# ==============================
@st.cache_data(ttl=60)
def load_config():
    p = Path(APP_DIR) / "config.yaml"
    if p.exists():
        return load_yaml(str(p))
    return {
        "symbols": ["ETHUSDT", "BTCUSDT"],  # ETH first by default
        "signals": {"min_gap_minutes": 20, "max_per_day": 20},
        "intraday": {"interval": "5m", "lookback_days": 7,
                     "min_minutes_between_signals": 20, "max_signals_per_day": 20},
        "cross_verify": {"price_tolerance_bps": 15},
        "schedule": {"poll_interval_seconds": 30},
    }

cfg = load_config()

# derive throttles with fallbacks
min_gap = int(cfg.get("signals", {}).get(
    "min_gap_minutes", cfg.get("intraday", {}).get("min_minutes_between_signals", 20)
))
max_day = int(cfg.get("signals", {}).get(
    "max_per_day", cfg.get("intraday", {}).get("max_signals_per_day", 20)
))

# ==============================
# Controls
# ==============================
symbols = st.multiselect(
    "Symbols",
    ["ETHUSDT", "BTCUSDT"],                    # ETH + BTC only
    default=cfg.get("symbols", ["ETHUSDT"])    # default ETH
)
colA, colB, colC = st.columns(3)
with colA:
    poll = st.number_input("Refresh (seconds)", 15, 120,
                           int(cfg.get("schedule", {}).get("poll_interval_seconds", 30)), 5)
with colB:
    min_gap = st.number_input("Min minutes between signals", 5, 120, min_gap, 5)
with colC:
    max_day = st.number_input("Max signals/day", 5, 50, max_day, 1)

show_provisional = st.checkbox("Show provisional signal (forming candle) — may change on close", value=True)

# ==============================
# Helpers (UK time, countdown)
# ==============================
def to_london(ts):
    """Convert timestamp to Europe/London correctly and return pandas.Timestamp."""
    if isinstance(ts, pd.Timestamp):
        if ts.tz is not None:
            return ts.tz_convert(UK_TZ)
        return ts.tz_localize("UTC").tz_convert(UK_TZ)
    if getattr(ts, "tzinfo", None) is not None:
        return pd.Timestamp(ts.astimezone(UK_TZ))
    return pd.Timestamp(pytz.utc.localize(ts).astimezone(UK_TZ))

def fmt12(ts):  # pretty 12-hour format
    return to_london(ts).strftime("%Y-%m-%d %I:%M %p %Z")

def fmt12_no_tz(ts):
    return to_london(ts).strftime("%Y-%m-%d %I:%M %p")

def now_uk_floor():
    return datetime.now(UK_TZ).replace(second=0, microsecond=0)

def next_five_minute_boundary(dt: datetime) -> datetime:
    minute = (dt.minute // 5 + 1) * 5
    if minute >= 60:
        return dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return dt.replace(minute=minute, second=0, microsecond=0)

def countdown_to_next_bar():
    now = datetime.now(UK_TZ)
    nxt = next_five_minute_boundary(now)
    delta = (nxt - now)
    return nxt.strftime("%I:%M %p %Z"), f"{delta.seconds//60:02d}:{delta.seconds%60:02d}"

def latest_nonzero(series: pd.Series):
    nz = series[series != 0]
    if len(nz) == 0:
        return None, 0
    idx = nz.index[-1]
    return idx, int(nz.iloc[-1])

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
        "provisional": "FLAT"
    }
    empty_feats = pd.DataFrame({"close": [], "ema_fast": [], "ema_slow": [], "rsi14": []})
    empty_hist = pd.DataFrame({"time": [], "signal": []})
    return latest, empty_feats, empty_hist, 0

# ==============================
# Per-symbol pipeline (RAW strategy; ETH + BTC)
# ==============================
async def fetch_symbol(session, sym: str):
    gecko_map = {
        "BTCUSDT": ["bitcoin"],
        "ETHUSDT": ["ethereum"],
    }

    # 1) OHLC
    ohlc = await get_klines(session, sym, interval=cfg["intraday"]["interval"], days=int(cfg["intraday"]["lookback_days"]))
    if ohlc is None or ohlc.empty or "open_time" not in ohlc.columns:
        st.warning(f"{sym}: no market data returned. Retrying on next refresh.")
        return _empty_symbol(sym)

    # 2) Features
    feats = build_features_5m(ohlc)
    if feats is None or feats.empty:
        st.warning(f"{sym}: insufficient data for features (5m).")
        return _empty_symbol(sym)

    # Ensure index is UK tz-aware for display
    try:
        if getattr(feats.index, "tz", None) is None and getattr(feats.index, "tzinfo", None) is None:
            feats.index = feats.index.tz_localize("UTC")
        else:
            try:
                feats.index = feats.index.tz_convert("UTC")
            except Exception:
                pass
        feats.index = feats.index.tz_convert(UK_TZ)
    except Exception:
        feats.index = pd.to_datetime(feats.index).tz_localize(UK_TZ)

    # 3) Signals (raw) + throttling
    scfg = ScalperConfig(min_minutes_between_signals=int(min_gap), max_signals_per_day=int(max_day))
    sigs = combine_signals_5m(feats, scfg)  # -1/0/+1

    # 4) Provisional preview (forming candle)
    # NOTE: Without websocket/partial candles, this is effectively the last closed bar.
    # It still gives a heads-up, but can only change on the next close.
    provisional_val = int(sigs.iloc[-1]) if len(sigs) else 0
    provisional = "LONG" if provisional_val > 0 else ("SHORT" if provisional_val < 0 else "FLAT")

    # 5) Cross-verify price
    gecko_px = None
    try:
        cg = await coingecko_price(session, gecko_map[sym])
        if isinstance(cg, dict) and len(cg):
            v = next(iter(cg.values()))
            gecko_px = float(v["usd"]) if isinstance(v, dict) and "usd" in v else float(v)
    except Exception:
        gecko_px = None

    last_close = float(feats["close"].iloc[-1])
    diff_bps = (abs(last_close - gecko_px) / ((last_close + gecko_px) / 2) * 10000) if gecko_px is not None else None

    # 6) Latest closed-bar stance (official)
    idx_last, val_last = latest_nonzero(sigs)
    if idx_last is None:
        direction = "FLAT"
        ts_label = fmt12(feats.index[-1])
    else:
        direction = "LONG" if val_last > 0 else "SHORT"
        ts_label = fmt12(idx_last)

    latest = {
        "symbol": sym,
        "signal": direction,
        "price_binance": round(last_close, 2),
        "price_coingecko": None if gecko_px is None else round(float(gecko_px), 2),
        "price_diff_bps": None if diff_bps is None else round(float(diff_bps), 1),
        "rsi14": round(float(feats["rsi14"].iloc[-1]), 1),
        "ema9_minus_21": round(float(feats["ema_fast"].iloc[-1] - feats["ema_slow"].iloc[-1]), 2),
        "timestamp": ts_label,     # 12-hour clock
        "provisional": provisional # preview for forming candle
    }

    # 7) History (UK, 12-hour)
    nonzero = sigs[sigs != 0].tail(30)
    hist = pd.DataFrame({
        "time": [fmt12_no_tz(t) for t in nonzero.index],
        "signal": ["LONG" if v > 0 else "SHORT" for v in nonzero.values],
    })

    return latest, feats.tail(200)[["close", "ema_fast", "ema_slow", "rsi14"]], hist, provisional_val

# ==============================
# Fetch & Header: Trade-Now panel
# ==============================
async def run():
    import aiohttp
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_symbol(session, sym) for sym in symbols]
        return await asyncio.gather(*tasks)

results = asyncio.run(run())

next_time, tminus = countdown_to_next_bar()
st.info(f"🕒 Current UK time: **{now_uk_floor().strftime('%I:%M %p %Z')}**  |  Next 5-min bar: **{next_time}** (T–{tminus})")

# ==============================
# Summary: current stance (official) + provisional
# ==============================
rows = []
for (latest, feats, hist, last_sig_val) in results:
    rows.append({
        "symbol": latest["symbol"],
        "trade_now (official)": latest["signal"],
        "provisional (forming)": latest["provisional"] if show_provisional else "—",
        "last_bar (UK)": feats.index[-1].strftime("%Y-%m-%d %I:%M %p %Z"),
        "active_since": latest["timestamp"],  # when the current posture began
        "price_binance": latest["price_binance"],
        "price_coingecko": latest["price_coingecko"],
        "diff_bps": latest["price_diff_bps"],
        "rsi14": latest["rsi14"],
        "ema9-21": latest["ema9_minus_21"],
    })

st.subheader("🚦 Trade Now (by symbol)")
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ==============================
# Per-symbol panels
# ==============================
left, right = st.columns(2)
for i, (latest, feats, hist, last_sig_val) in enumerate(results):
    with (left if i % 2 == 0 else right):
        st.markdown(f"### {latest['symbol']} — Now: **{latest['signal']}**")
        if show_provisional:
            preview = "LONG" if last_sig_val > 0 else ("SHORT" if last_sig_val < 0 else "FLAT")
            st.caption(f"Provisional (forming candle): **{preview}** — may change on close")

        st.caption(f"Active since: **{latest['timestamp']}**  •  Last bar: **{feats.index[-1].strftime('%Y-%m-%d %I:%M %p %Z')}**")

        # ---- PRICE CHART: Close + EMA9 + EMA21 (Altair) ----
        if not feats.empty:
            plot_df = feats[["close", "ema_fast", "ema_slow"]].copy().reset_index()
            plot_df = plot_df.rename(columns={"ts": "time"})
            long_df = plot_df.melt("time", var_name="series", value_name="value")
            long_df["series"] = long_df["series"].map({"close": "Close", "ema_fast": "EMA 9", "ema_slow": "EMA 21"})

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

        # Recent signals table (raw)
        if not hist.empty:
            st.markdown("**Recent non-zero signals (raw)**")
            st.dataframe(hist.sort_values("time", ascending=False).reset_index(drop=True),
                         use_container_width=True, height=240)
