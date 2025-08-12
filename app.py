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
from datetime import datetime
import pandas as pd
import streamlit as st
import pytz
import altair as alt
from streamlit_autorefresh import st_autorefresh

from src.utils import load_yaml
from src.data_sources import get_klines, coingecko_price
from src.features_5m import build_features_5m
from src.signals_5m import combine_signals_5m, ScalperConfig

UK_TZ = pytz.timezone("Europe/London")

st.set_page_config(page_title="⚡ Crypto Edge — 5-Minute Signals", layout="wide")
st.title("⚡ Crypto Edge — 5-Minute Signals (BTC & ETC)")
st.caption("Public data only • Cross-verified prices • 5m timeframe • UK time")

# ==============================
# CONFIG (short cache)
# ==============================
@st.cache_data(ttl=60)
def load_config():
    p = Path(APP_DIR) / "config.yaml"
    if p.exists():
        return load_yaml(str(p))
    return {
        "symbols": ["BTCUSDT", "ETCUSDT"],
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

# UI controls
symbols = st.multiselect("Symbols",
                         ["BTCUSDT", "ETCUSDT"],
                         default=cfg.get("symbols", ["BTCUSDT", "ETCUSDT"]))
colA, colB, colC = st.columns(3)
with colA:
    poll = st.number_input("Refresh (seconds)", 15, 120,
                           int(cfg.get("schedule", {}).get("poll_interval_seconds", 30)), 5)
with colB:
    min_gap = st.number_input("Min minutes between signals", 5, 120, min_gap, 5)
with colC:
    max_day = st.number_input("Max signals/day", 5, 50, max_day, 1)

# Auto-refresh + manual force-refresh
st_autorefresh(interval=int(poll) * 1000, key="auto")
if st.button("Force refresh data now"):
    st.cache_data.clear()
    st.experimental_rerun()

show_raw = st.checkbox("Show raw (unfiltered) signal history", value=False)

# ==============================
# Helpers
# ==============================
def to_london(ts):
    """Convert a timestamp to Europe/London (keeps tz-aware)."""
    if getattr(ts, "tzinfo", None) is None:
        ts = pytz.utc.localize(ts)
    return ts.astimezone(UK_TZ)

def now_london_floor_minute():
    return datetime.now(UK_TZ).replace(second=0, microsecond=0)

def latest_nonzero(series: pd.Series):
    nz = series[series != 0]
    if len(nz) == 0:
        return None, 0
    idx = nz.index[-1]
    return idx, int(nz.iloc[-1])

def _empty_symbol(sym: str):
    latest = {
        "symbol": sym, "signal": "FLAT", "price_binance": None, "price_coingecko": None,
        "price_diff_bps": None, "rsi14": None, "ema9_minus_21": None,
        "timestamp": ""
    }
    empty = pd.DataFrame({"close": [], "ema_fast": [], "ema_slow": [], "rsi14": []})
    hist = pd.DataFrame({"time": [], "signal": []})
    return latest, empty, hist, hist

def pick_active(hist_df: pd.DataFrame, now_uk: datetime):
    """Return the row at/just before 'now' (UK). hist_df['time'] is UK 'YYYY-MM-DD HH:MM'."""
    if hist_df is None or hist_df.empty:
        return None
    df = hist_df.copy()
    df["t"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M", errors="coerce")
    df = df.dropna(subset=["t"])
    # localize to UK (strings already represent UK local)
    df["t"] = df["t"].apply(lambda t: UK_TZ.localize(t) if t.tzinfo is None else t.astimezone(UK_TZ))
    df = df.sort_values("t")
    df_now = df[df["t"] <= now_uk]
    if df_now.empty:
        return None
    return df_now.iloc[-1][["time", "signal"]].to_dict()

# ==============================
# Per-symbol pipeline (BASELINE)
# ==============================
async def fetch_symbol(session, sym: str):
    gecko_map = {"BTCUSDT": ["bitcoin"], "ETCUSDT": ["ethereum-classic"]}

    # 1) Fetch OHLC
    ohlc = await get_klines(session, sym,
                            interval=cfg["intraday"]["interval"],
                            days=int(cfg["intraday"]["lookback_days"]))
    if ohlc is None or ohlc.empty or "open_time" not in ohlc.columns:
        st.warning(f"{sym}: no market data returned. Retrying on next refresh.")
        return _empty_symbol(sym)

    # 2) Build features (RSI, EMA9/21, etc.)
    feats = build_features_5m(ohlc)
    if feats is None or feats.empty:
        st.warning(f"{sym}: insufficient data for features (5m).")
        return _empty_symbol(sym)

    # --- Force feature index to UK local time ---
    # Usually feats.index is UTC; ensure tz-aware and convert once.
    try:
        if getattr(feats.index, "tz", None) is None and getattr(feats.index, "tzinfo", None) is None:
            feats.index = feats.index.tz_localize("UTC")
        else:
            try:
                feats.index = feats.index.tz_convert("UTC")
            except Exception:
                pass
        feats.index = feats.index.tz_convert("Europe/London")
    except Exception:
        # last resort: treat as naive UK
        feats.index = pd.to_datetime(feats.index)
        feats.index = feats.index.tz_localize("Europe/London")

    # 3) Core signals (no extra filters) + throttling
    scfg = ScalperConfig(min_minutes_between_signals=int(min_gap),
                         max_signals_per_day=int(max_day))
    sigs = combine_signals_5m(feats, scfg)  # -1/0/+1

    # 4) Cross-verify price (robust)
    gecko_px = None
    try:
        cg = await coingecko_price(session, gecko_map[sym])
        if isinstance(cg, dict) and len(cg):
            v = next(iter(cg.values()))
            gecko_px = float(v["usd"]) if isinstance(v, dict) and "usd" in v else float(v)
    except Exception:
        gecko_px = None

    last_close = float(feats["close"].iloc[-1])
    diff_bps = (
        abs(last_close - gecko_px) / ((last_close + gecko_px) / 2) * 10000
        if gecko_px is not None else None
    )

    # 5) Latest signal snapshot
    idx_last, val_last = latest_nonzero(sigs)
    if idx_last is None:
        latest = {
            "symbol": sym, "signal": "FLAT",
            "price_binance": round(last_close, 2),
            "price_coingecko": None if gecko_px is None else round(float(gecko_px), 2),
            "price_diff_bps": None if diff_bps is None else round(float(diff_bps), 1),
            "rsi14": round(float(feats["rsi14"].iloc[-1]), 1),
            "ema9_minus_21": round(float(feats["ema_fast"].iloc[-1] - feats["ema_slow"].iloc[-1]), 2),
            "timestamp": feats.index[-1].strftime("%Y-%m-%d %H:%M %Z"),
        }
    else:
        direction = "LONG" if val_last > 0 else "SHORT"
        latest = {
            "symbol": sym, "signal": direction,
            "price_binance": round(float(feats.loc[idx_last, "close"]), 2),
            "price_coingecko": None if gecko_px is None else round(float(gecko_px), 2),
            "price_diff_bps": None if diff_bps is None else round(float(diff_bps), 1),
            "rsi14": round(float(feats.loc[idx_last, "rsi14"]), 1),
            "ema9_minus_21": round(float(feats.loc[idx_last, "ema_fast"] - feats.loc[idx_last, "ema_slow"]), 2),
            "timestamp": idx_last.strftime("%Y-%m-%d %H:%M %Z"),
        }

    # 6) Histories (non-zero signals only)
    nonzero = sigs[sigs != 0].tail(30)
    hist = pd.DataFrame({
        "time": [t.strftime("%Y-%m-%d %H:%M") for t in nonzero.index],  # feats index is already UK tz
        "signal": ["LONG" if v > 0 else "SHORT" for v in nonzero.values]
    })

    # Plot features (last 200 rows)
    plot_feats = feats.tail(200)[["close", "ema_fast", "ema_slow", "rsi14"]].copy()
    return latest, plot_feats, hist, hist  # (we reuse hist as "raw" too for baseline)

# ==============================
# Main fetch
# ==============================
async def run():
    import aiohttp
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_symbol(session, sym) for sym in symbols]
        return await asyncio.gather(*tasks)

results = asyncio.run(run())

# ==============================
# Active signal "NOW" banner
# ==============================
now_uk = now_london_floor_minute()
active_rows = []
for (latest, feats, hist, _raw) in results:
    act = None
    if hist is not None and not hist.empty:
        act = (lambda h: (lambda r: {"time": r["time"], "signal": r["signal"]} if r is not None else None)(
            (lambda df: df.iloc[-1][["time","signal"]] if not df.empty else None)(
                (lambda df0: df0.assign(t=pd.to_datetime(df0["time"])).sort_values("t")[df0.assign(
                    t=pd.to_datetime(df0["time"])).assign(t=lambda d:d["t"].apply(lambda t: UK_TZ.localize(t))).loc[:, "t"] <= now_uk]
                ))))(hist)
    if act:
        active_rows.append({
            "symbol": latest["symbol"],
            "active_signal": act["signal"],
            "active_time (UK)": act["time"],
            "price (approx)": latest["price_binance"]
        })

if active_rows:
    st.subheader("🚨 Active signal now")
    st.dataframe(pd.DataFrame(active_rows), use_container_width=True)

# ==============================
# Render per symbol
# ==============================
st.subheader("Live Signals")
rows = [latest for (latest, _, _, _) in results]
st.dataframe(pd.DataFrame(rows), use_container_width=True)

left, right = st.columns(2)
for i, (latest, feats, hist, raw_hist) in enumerate(results):
    with (left if i % 2 == 0 else right):
        st.markdown(f"### {latest['symbol']} — {latest['signal']}")
        st.caption(f"Time: {latest['timestamp']}")

        # ---- PRICE CHART: Close + EMA9 + EMA21 + Signal Arrows ----
        if not feats.empty:
            plot_df = feats[["close", "ema_fast", "ema_slow"]].copy().reset_index()
            plot_df = plot_df.rename(columns={"ts": "time"})

            # Build arrow points from history (SAFE MERGE on UK minute key)
            sigs_for_plot = hist.copy()
            if not sigs_for_plot.empty:
                sigs_for_plot["time_key"] = sigs_for_plot["time"].astype(str)

                ts = pd.to_datetime(plot_df["time"], errors="coerce")
                # If naive, treat as UK; else convert to UK
                if getattr(ts.dt, "tz", None) is None:
                    ts = ts.dt.tz_localize("Europe/London", nonexistent="NaT", ambiguous="NaT")
                else:
                    ts = ts.dt.tz_convert("Europe/London")
                plot_df["time_key"] = ts.dt.strftime("%Y-%m-%d %H:%M")

                join_df = plot_df[["time", "close", "time_key"]].merge(
                    sigs_for_plot[["time_key", "signal"]],
                    on="time_key",
                    how="inner",
                    validate="m:1"
                )
                join_df["arrow_y"] = join_df.apply(
                    lambda r: r["close"] * (1.005 if r["signal"] == "LONG" else 0.995),
                    axis=1
                )
            else:
                join_df = pd.DataFrame(columns=["time", "close", "signal", "arrow_y"])

            # Long-form for EMAs/Close
            long_df = plot_df.melt("time", var_name="series", value_name="value")
            name_map = {"close": "Close", "ema_fast": "EMA 9", "ema_slow": "EMA 21"}
            long_df["series"] = long_df["series"].map(name_map)

            # Price + EMAs
            line_chart = alt.Chart(long_df).mark_line().encode(
                x=alt.X("time:T", title="Time (UK)"),
                y=alt.Y("value:Q", title="Price (USD)"),
                color=alt.Color(
                    "series:N",
                    title="Legend",
                    scale=alt.Scale(
                        domain=["Close", "EMA 9", "EMA 21"],
                        range=["#1f77b4", "#e45756", "#4ea8de"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("time:T", title="Time"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("value:Q", title="Value", format=",.2f"),
                ],
            )

            # Signal arrows
            arrow_chart = alt.Chart(join_df).mark_text(
                align="center", baseline="middle", fontSize=14, fontWeight="bold"
            ).encode(
                x="time:T",
                y="arrow_y:Q",
                text=alt.condition(
                    alt.datum.signal == "LONG", alt.value("▲"), alt.value("▼")
                ),
                color=alt.condition(
                    alt.datum.signal == "LONG", alt.value("green"), alt.value("red")
                ),
                tooltip=[
                    alt.Tooltip("time:T", title="Signal Time"),
                    alt.Tooltip("signal:N", title="Signal Type"),
                    alt.Tooltip("close:Q", title="Close", format=",.2f"),
                ]
            )

            final_chart = (line_chart + arrow_chart).properties(height=260).interactive()
            st.altair_chart(final_chart, use_container_width=True)

            # RSI bar chart
            st.bar_chart(feats[["rsi14"]])

        # Recent signals table
        if not hist.empty:
            act = pick_active(hist, now_london_floor_minute())
            hf = hist.copy()
            if act:
                hf["ACTIVE_NOW"] = hf.apply(
                    lambda r: "◉ NOW" if (r["time"] == act["time"] and r["signal"] == act["signal"]) else "",
                    axis=1
                )
                hf = hf.sort_values(by=["ACTIVE_NOW", "time"], ascending=[False, False]).reset_index(drop=True)
            st.markdown("**Recent non-zero signals**")
            st.dataframe(hf, use_container_width=True, height=260)

        if show_raw and not raw_hist.empty:
            st.markdown("**Raw (unfiltered) signals**")
            st.dataframe(raw_hist, use_container_width=True, height=240)
