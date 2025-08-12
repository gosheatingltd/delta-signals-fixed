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
st.caption("Public data only • Cross-verified prices • Filters: Volume + Trend • 5m timeframe")

# ==============================
# CONFIG (robust)
# ==============================
@st.cache_data(ttl=60)
def load_config():
    p = Path(APP_DIR) / "config.yaml"
    if p.exists():
        return load_yaml(str(p))
    # sane defaults so the app still boots
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

show_raw = st.checkbox("Show raw (unfiltered) signal history", value=False)

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
        "volume_ok": None,
        "trend": "N/A",
    }
    empty_feats = pd.DataFrame({"close": [], "ema_fast": [], "ema_slow": [], "rsi14": []})
    empty_hist = pd.DataFrame({"time": [], "signal": []})
    return latest, empty_feats, empty_hist, empty_hist

def to_london(ts):
    uk = pytz.timezone("Europe/London")
    if getattr(ts, "tzinfo", None) is None:
        ts = pytz.utc.localize(ts)
    return ts.astimezone(uk)

def latest_nonzero(series: pd.Series):
    nz = series[series != 0]
    if len(nz) == 0:
        return None, 0
    idx = nz.index[-1]
    return idx, int(nz.iloc[-1])

# ==============================
# Per-symbol pipeline (with filters)
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

    # Ensure volume is present
    if "volume" not in feats.columns:
        feats["volume"] = ohlc["volume"].values

    # Add trend EMAs for filters
    feats["ema50"] = feats["close"].ewm(span=50, adjust=False).mean()
    feats["ema200"] = feats["close"].ewm(span=200, adjust=False).mean()

    # 3) Raw signals (original strategy + throttling)
    scfg = ScalperConfig(
        min_minutes_between_signals=int(min_gap), max_signals_per_day=int(max_day)
    )
    sigs_raw = combine_signals_5m(feats, scfg)  # -1/0/+1

    # 4) Filters
    # Volume filter: last bar volume > 1.5x rolling 20-bar mean
    vol_mean20 = feats["volume"].rolling(20, min_periods=20).mean()
    vol_ok_series = feats["volume"] > (1.5 * vol_mean20)

    # Trend filter: LONG only if EMA50>EMA200, SHORT only if EMA50<EMA200
    uptrend_series = feats["ema50"] > feats["ema200"]
    downtrend_series = feats["ema50"] < feats["ema200"]

    # Apply filters to every bar of the raw signal
    sigs_filt = sigs_raw.copy()
    # LONGs must satisfy vol_ok & uptrend
    sigs_filt[(sigs_raw > 0) & ~(vol_ok_series & uptrend_series)] = 0
    # SHORTs must satisfy vol_ok & downtrend
    sigs_filt[(sigs_raw < 0) & ~(vol_ok_series & downtrend_series)] = 0

    # 5) Price cross-verify (robust parse)
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
        if gecko_px is not None else None
    )

    # 6) Determine latest filtered signal
    idx_filt, val_filt = latest_nonzero(sigs_filt)
    if idx_filt is None:
        # No valid signal after filters
        latest = {
            "symbol": sym,
            "signal": "FLAT",
            "price_binance": round(last_close, 2),
            "price_coingecko": None if gecko_px is None else round(float(gecko_px), 2),
            "price_diff_bps": None if diff_bps is None else round(float(diff_bps), 1),
            "rsi14": round(float(feats["rsi14"].iloc[-1]), 1),
            "ema9_minus_21": round(float(feats["ema_fast"].iloc[-1] - feats["ema_slow"].iloc[-1]), 2),
            "timestamp": to_london(feats.index[-1]).strftime("%Y-%m-%d %H:%M %Z"),
            "volume_ok": bool(vol_ok_series.iloc[-1]) if not vol_ok_series.isna().iloc[-1] else None,
            "trend": "Uptrend" if uptrend_series.iloc[-1] else ("Downtrend" if downtrend_series.iloc[-1] else "Sideways"),
        }
    else:
        direction = "LONG" if val_filt > 0 else "SHORT"
        last_ts_uk = to_london(idx_filt)
        latest = {
            "symbol": sym,
            "signal": direction,
            "price_binance": round(float(feats.loc[idx_filt, "close"]), 2),
            "price_coingecko": None if gecko_px is None else round(float(gecko_px), 2),
            "price_diff_bps": None if diff_bps is None else round(float(diff_bps), 1),
            "rsi14": round(float(feats.loc[idx_filt, "rsi14"]), 1),
            "ema9_minus_21": round(float(feats.loc[idx_filt, "ema_fast"] - feats.loc[idx_filt, "ema_slow"]), 2),
            "timestamp": last_ts_uk.strftime("%Y-%m-%d %H:%M %Z"),
            "volume_ok": bool(vol_ok_series.loc[idx_filt]) if not pd.isna(vol_ok_series.loc[idx_filt]) else None,
            "trend": "Uptrend" if uptrend_series.loc[idx_filt] else ("Downtrend" if downtrend_series.loc[idx_filt] else "Sideways"),
        }

    # 7) Histories (filtered vs raw)
    # Filtered history: only non-zero filtered signals
    nonzero_filt = sigs_filt[sigs_filt != 0].tail(30)
    hist_filt = pd.DataFrame({
        "time": [to_london(t).strftime("%Y-%m-%d %H:%M") for t in nonzero_filt.index],
        "signal": ["LONG" if v > 0 else "SHORT" for v in nonzero_filt.values]
    })

    # Raw history (for optional comparison)
    nonzero_raw = sigs_raw[sigs_raw != 0].tail(30)
    hist_raw = pd.DataFrame({
        "time": [to_london(t).strftime("%Y-%m-%d %H:%M") for t in nonzero_raw.index],
        "signal": ["LONG" if v > 0 else "SHORT" for v in nonzero_raw.values]
    })

    # Return feats with columns used for plotting
    plot_feats = feats.tail(200)[["close", "ema_fast", "ema_slow", "rsi14"]].copy()
    return latest, plot_feats, hist_filt, hist_raw

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
st.subheader("Live Signals (filtered)")
rows = [latest for (latest, _, _, _) in results]
st.dataframe(pd.DataFrame(rows), use_container_width=True)

left, right = st.columns(2)
for i, (latest, feats, hist_filt, hist_raw) in enumerate(results):
    with (left if i % 2 == 0 else right):
        st.markdown(f"### {latest['symbol']} — {latest['signal']}")
        st.caption(f"Trend: {latest['trend']} • Volume OK: {latest['volume_ok']} • Time: {latest['timestamp']}")

        # ---- PRICE CHART: Close + EMA9 + EMA21 + Signal Arrows ----
        if not feats.empty:
            plot_df = feats[["close", "ema_fast", "ema_slow"]].copy().reset_index()
            plot_df = plot_df.rename(columns={"ts": "time"})

            # Build arrow points from filtered history (SAFE MERGE on UK minute key)
            sigs_for_plot = hist_filt.copy()
            if not sigs_for_plot.empty:
                # hist_filt["time"] is UK formatted strings like "YYYY-MM-DD HH:MM"
                sigs_for_plot["time_key"] = sigs_for_plot["time"].astype(str)

                # Create a matching key from plot_df times → UK strings at minute resolution
                plot_df["time_key"] = (
                    pd.to_datetime(plot_df["time"])               # ensure datetime
                    .dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
                    .dt.tz_convert("Europe/London")
                    .dt.strftime("%Y-%m-%d %H:%M")
                )

                # Merge on the uniform string key
                join_df = plot_df[["time", "close", "time_key"]].merge(
                    sigs_for_plot[["time_key", "signal"]],
                    on="time_key",
                    how="inner",
                    validate="m:1"
                )

                # Place arrows slightly above/below price
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
                    alt.datum.signal == "LONG",
                    alt.value("▲"),
                    alt.value("▼")
                ),
                color=alt.condition(
                    alt.datum.signal == "LONG",
                    alt.value("green"),
                    alt.value("red")
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

        # Recent signals table (filtered by default; raw optional)
        if not hist_filt.empty:
            st.markdown("**Recent non-zero signals (filtered)**")
            st.dataframe(hist_filt, use_container_width=True, height=240)

        if show_raw and not hist_raw.empty:
            st.markdown("**Recent non-zero signals (raw, unfiltered)**")
            st.dataframe(hist_raw, use_container_width=True, height=240)
