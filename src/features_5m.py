import pandas as pd
import numpy as np

def ema(series, span=20):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def build_features_5m(ohlc: pd.DataFrame):
    if ohlc is None or ohlc.empty or "open_time" not in ohlc.columns:
        return pd.DataFrame()

    df = ohlc.copy()
    df = df.rename(columns={"open_time": "ts"}).set_index("ts").sort_index()

    df["ret_5m"] = df["close"].pct_change()
    df["ret_1h"] = df["close"].pct_change(12)
    df["vol_1h"] = df["ret_5m"].rolling(12).std().fillna(0)
    df["ema_fast"] = ema(df["close"], 9)
    df["ema_slow"] = ema(df["close"], 21)
    df["rsi14"] = rsi(df["close"], 14)
    df["roll_high_4h"] = df["high"].rolling(48).max()
    df["roll_low_4h"]  = df["low"].rolling(48).min()
    return df.dropna()
