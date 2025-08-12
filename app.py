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
import aiohttp  # used by safe fetchers

from src.utils import load_yaml
from src.data_sources import get_klines, coingecko_price
from src.features_5m import build_features_5m
from src.signals_5m import combine_signals_5m, ScalperConfig

UK_TZ = pytz.timezone("Europe/London")

st.set_page_config(page_title="⚡ Crypto Edge — 5-Min Signals (ETH & BTC)", layout="wide")
st.title("⚡ Crypto Edge — 5-Minute Signals (ETH & BTC)")
st.caption("Public data only • Cross-verified prices • 5m timeframe • 12-hour clock • Filtered + Raw")

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
# Symbol selector (ETH/BTC only) — robust defaults
# ==============================
OPTIONS = ["ETHUSDT", "BTCUSDT"]
cfg_syms = cfg.get("symbols", ["ETHUSDT"])
default_syms = [s for s in cfg_syms if s in OPTIONS] or ["ETHUSDT"]

symbols = st.multiselect(
    "Symbols",
    OPTIONS,
    default=default_syms
)

# If user clears everything, auto-restore ETH to avoid empty results
if not symbols:
    st.warning("No symbols selected — defaulting to ETHUSDT.")
    symbols = ["ETHUSDT"]

# Other controls
colA, colB, colC = st.columns(3)
with colA:
    poll = st.number_input("Refresh (seconds)", 15, 120,
                           int(cfg.get("schedule", {}).get("poll_interval_seconds", 30)), 5)
with colB:
    min_gap = st.number_input("Min minutes between signals", 5, 120, min_gap, 5)
with colC:
    max_day = st.number_input("Max signals/day", 5, 50, max_day, 1)

show_provisional = st.checkbox("Show provisional (forming candle) — may change on close", value=True)
show_raw_table = st.checkbox("Show raw (unfiltered) table under each chart", value=True)

# ==============================
# Helpers (UK time, formatting, countdown)
# ==============================
def to_london(ts):
    if isinstance(ts, pd.Timestamp):
        if ts.tz is not None:
            return ts.tz_convert(UK_TZ)
        return ts.tz_localize("UTC").tz_convert(UK_TZ)
    if getattr(ts, "tzinfo", None) is not None:
        return pd.Timestamp(ts.astimezone(UK_TZ))
    return pd.Timestamp(pytz.utc.localize(ts).astimezone(UK_TZ))

def fmt12(ts):        # "YYYY-mm-dd HH:MM AM/PM TZ"
    return to_london(ts).strftime("%Y-%m-%d %I:%M %p %Z")

def fmt12_no_tz(ts):  # "YYYY-mm-dd HH:MM AM/PM"
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
        "signal_filtered": "FLAT",
        "signal_raw": "FLAT",
        "price_binance": None,
        "price_coingecko": None,
        "price_diff_bps": None,
        "rsi14": None,
        "ema9_minus_21": None,
        "active_since_filtered": "",
        "active_since_raw": "",
        "provisional_filtered": "FLAT",
        "provisional_raw": "FLAT",
        "trend": "N/A",
        "volume_ok": None,
        "last_bar": ""
    }
    empty_feats = pd.DataFrame({"close": [], "ema_fast": [], "ema_slow": [], "rsi14": []})
    empty_hist = pd.DataFrame({"time": [], "signal": []})
    return latest, empty_feats, empty_hist, empty_hist

# ==============================
# Safe OHLC fetcher: your get_klines() → Binance Spot → Binance Futures
# ==============================
INTERVAL_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "2h": "2h", "4h": "4h",
    "1d": "1d"
}

async def _binance_spot_klines(session: aiohttp.ClientSession, symbol: str, interval: str, days: int):
    iv = INTERVAL_MAP.get(interval, "5m")
    per_day_5m = 12 * 24
    limit = min(1500, max(100, days * per_day_5m))
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": iv, "limit": limit}
    async with session.get(url, params=params, timeout=20) as r:
        if r.status != 200:
            return pd.DataFrame()
        data = await r.json()
    if not data:
        return pd.DataFrame()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","n","tbbav","tbqav","ignore"]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["open_time","open","high","low","close","volume"]]

async def _binance_futures_klines(session: aiohttp.ClientSession, symbol: str, interval: str, days: int):
    iv = INTERVAL_MAP.get(interval, "5m")
    per_day_5m = 12 * 24
    limit = min(1500, max(100, days * per_day_5m))
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": iv, "limit": limit}
    async with session.get(url, params=params, timeout=20) as r:
        if r.status != 200:
            return pd.DataFrame()
        data = await r.json()
    if not data or isinstance(data, dict):
        return pd.DataFrame()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","n","tbbav","tbqav","ignore"]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["open_time","open","high","low","close","volume"]]

async def get_klines_safe(session: aiohttp.ClientSession, symbol: str, interval: str, days: int) -> pd.DataFrame:
    """
    Try project's get_klines(); if empty, fall back to Binance Spot, then Futures.
    Ensures columns: ['open_time','open','high','low','close','volume'] (open_time is tz-aware UTC).
    """
    # 1) Your existing source
    try:
        df = await get_klines(session, symbol, interval=interval, days=days)
        if df is not None and not df.empty and "open_time" in df.columns:
            return df
    except Exception:
        pass
    # 2) Binance Spot
    df_spot = await _binance_spot_klines(session, symbol, interval, days)
    if not df_spot.empty:
        return df_spot
    # 3) Binance Futures
    df_fut = await _binance_futures_klines(session, symbol, interval, days)
    if not df_fut.empty:
        return df_fut
    # 4) Nothing
    return pd.DataFrame()

# ==============================
# Per-symbol pipeline (Raw + Filters)
# ==============================
async def fetch_symbol(session, sym: str):
    gecko_map = {
        "BTCUSDT": ["bitcoin"],
        "ETHUSDT": ["ethereum"],
    }

    # 1) OHLC (safe fetch)
    ohlc = await get_klines_safe(session, sym, interval=cfg["intraday"]["interval"], days=int(cfg["intraday"]["lookback_days"]))
    if ohlc is None or ohlc.empty or "open_time" not in ohlc.columns:
        st.warning(f"{sym}: no market data returned. Retrying on next refresh.")
        return _empty_symbol(sym)

    # 2) Features
    feats = build_features_5m(ohlc)
    if feats is None or feats.empty:
        st.warning(f"{sym}: insufficient data for features (5m).")
        return _empty_symbol(sym)

    # Ensure index UK tz-aware for display
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
    sigs_raw = combine_signals_5m(feats, scfg)  # -1/0/+1

    # 4) Filters (Volume + Trend)
    feats["ema50"] = feats["close"].ewm(span=50, adjust=False).mean()
    feats["ema200"] = feats["close"].ewm(span=200, adjust=False).mean()
    if "volume" not in feats.columns and "volume" in ohlc.columns:
        feats["volume"] = ohlc["volume"].values
    elif "volume" not in feats.columns:
        feats["volume"] = pd.Series(index=feats.index, dtype="float64").fillna(0.0)

    vol_mean20 = feats["volume"].rolling(20, min_periods=20).mean()
    vol_ok_series = feats["volume"] > (1.5 * vol_mean20)
    uptrend_series = feats["ema50"] > feats["ema200"]
    downtrend_series = feats["ema50"] < feats["ema200"]

    sigs_filt = sigs_raw.copy()
    sigs_filt[(sigs_raw > 0) & ~(vol_ok_series & uptrend_series)] = 0
    sigs_filt[(sigs_raw < 0) & ~(vol_ok_series & downtrend_series)] = 0

    # 5) Provisional previews (forming candle)
    prov_raw_val = int(sigs_raw.iloc[-1]) if len(sigs_raw) else 0
    prov_filt_val = int(sigs_filt.iloc[-1]) if len(sigs_filt) else 0
    provisional_raw = "LONG" if prov_raw_val > 0 else ("SHORT" if prov_raw_val < 0 else "FLAT")
    provisional_filt = "LONG" if prov_filt_val > 0 else ("SHORT" if prov_filt_val < 0 else "FLAT")

    # 6) Cross-verify price
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

    # 7) Latest snapshots
    idx_filt, val_filt = latest_nonzero(sigs_filt)
    idx_raw,  val_raw  = latest_nonzero(sigs_raw)

    latest = {
        "symbol": sym,
        "signal_filtered": "FLAT" if idx_filt is None else ("LONG" if val_filt > 0 else "SHORT"),
        "signal_raw":      "FLAT" if idx_raw  is None else ("LONG" if val_raw  > 0 else "SHORT"),
        "price_binance": round(last_close, 2),
        "price_coingecko": None if gecko_px is None else round(float(gecko_px), 2),
        "price_diff_bps": None if diff_bps is None else round(float(diff_bps), 1),
        "rsi14": round(float(feats["rsi14"].iloc[-1]), 1),
        "ema9_minus_21": round(float(feats["ema_fast"].iloc[-1] - feats["ema_slow"].iloc[-1]), 2),
        "active_since_filtered": "—" if idx_filt is None else fmt12(idx_filt),
        "active_since_raw":      "—" if idx_raw  is None else fmt12(idx_raw),
        "provisional_filtered": provisional_filt,
        "provisional_raw": provisional_raw,
        "trend": "Uptrend" if uptrend_series.iloc[-1] else ("Downtrend" if downtrend_series.iloc[-1] else "Sideways"),
        "volume_ok": bool(vol_ok_series.iloc[-1]) if not vol_ok_series.isna().iloc[-1] else None,
        "last_bar": fmt12(feats.index[-1]),
    }

    # 8) Histories (UK, 12-hour)
    nz_filt = sigs_filt[sigs_filt != 0].tail(30)
    hist_filt = pd.DataFrame({
        "time": [fmt12_no_tz(t) for t in nz_filt.index],
        "signal": ["LONG" if v > 0 else "SHORT" for v in nz_filt.values],
    })

    nz_raw = sigs_raw[sigs_raw != 0].tail(30)
    hist_raw = pd.DataFrame({
        "time": [fmt12_no_tz(t) for t in nz_raw.index],
        "signal": ["LONG" if v > 0 else "SHORT" for v in nz_raw.values],
    })

    plot_feats = feats.tail(200)[["close", "ema_fast", "ema_slow", "rsi14"]].copy()
    return latest, plot_feats, hist_filt, hist_raw

# ==============================
# Fetch & header
# ==============================
async def run():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_symbol(session, sym) for sym in symbols]
        return await asyncio.gather(*tasks)

results = asyncio.run(run())

next_time, tminus = countdown_to_next_bar()
st.info(f"🕒 Current UK time: **{now_uk_floor().strftime('%I:%M %p %Z')}**  |  Next 5-min bar: **{next_time}** (T–{tminus})")

# ==============================
# Summary: filtered + raw stance
# ==============================
rows = []
for (latest, feats, hist_filt, hist_raw) in results:
    rows.append({
        "symbol": latest["symbol"],
        "trade_now (filtered)": latest["signal_filtered"],
        "provisional (filtered)": latest["provisional_filtered"] if show_provisional else "—",
        "trade_now (raw)": latest["signal_raw"],
        "provisional (raw)": latest["provisional_raw"] if show_provisional else "—",
        "trend": latest["trend"],
        "volume_ok": latest["volume_ok"],
        "last_bar (UK)": latest["last_bar"],
        "active_since (filtered)": latest["active_since_filtered"],
        "active_since (raw)": latest["active_since_raw"],
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
for i, (latest, feats, hist_filt, hist_raw) in enumerate(results):
    with (left if i % 2 == 0 else right):
        st.markdown(f"### {latest['symbol']} — Filtered: **{latest['signal_filtered']}** • Raw: **{latest['signal_raw']}**")
        st.caption(f"Trend: {latest['trend']} • Volume OK: {latest['volume_ok']} • Last bar: {latest['last_bar']}")

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

        # Filtered recent signals (UK 12h)
        if hist_filt is not None and not hist_filt.empty:
            st.markdown("**Recent non-zero signals (filtered)**")
            st.dataframe(hist_filt.sort_values("time", ascending=False).reset_index(drop=True),
                         use_container_width=True, height=240)

        # Raw recent signals (optional)
        if show_raw_table and hist_raw is not None and not hist_raw.empty:
            st.markdown("**Recent non-zero signals (raw, unfiltered)**")
            st.dataframe(hist_raw.sort_values("time", ascending=False).reset_index(drop=True),
                         use_container_width=True, height=240)
