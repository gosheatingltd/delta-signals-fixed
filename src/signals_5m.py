from dataclasses import dataclass
import pandas as pd

@dataclass
class ScalperConfig:
    min_minutes_between_signals: int = 20
    max_signals_per_day: int = 20

def ma_cross_momentum(df: pd.DataFrame):
    long_cond = (df["ema_fast"] > df["ema_slow"]) & (df["ret_5m"] > 0) & (df["rsi14"] > 45)
    short_cond = (df["ema_fast"] < df["ema_slow"]) & (df["ret_5m"] < 0) & (df["rsi14"] < 55)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def range_breakout(df: pd.DataFrame):
    long_cond = (df["close"] > df["roll_high_4h"]) & (df["rsi14"] > 55)
    short_cond = (df["close"] < df["roll_low_4h"]) & (df["rsi14"] < 45)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def combine_signals_5m(df: pd.DataFrame, cfg: ScalperConfig):
    m = ma_cross_momentum(df)
    r = range_breakout(df)
    s = (m + r).clip(-1, 1)

    last_fire = None
    fires_day = {}
    out = pd.Series(0, index=df.index, dtype=int)

    for ts, val in s.items():
        if val == 0:
            continue
        day_key = ts.date()
        fires_day.setdefault(day_key, 0)

        if last_fire is not None:
            minutes = (ts - last_fire).total_seconds() / 60.0
            if minutes < cfg.min_minutes_between_signals:
                continue
        if fires_day[day_key] >= cfg.max_signals_per_day:
            continue

        out.loc[ts] = val
        fires_day[day_key] += 1
        last_fire = ts

    return out
