"""
feature_engineering.py
-----------------------
Builds technical-analysis features used as model inputs, and the
prediction target (next day's closing price).
"""

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages, momentum, and volatility indicators."""
    out = df.copy()

    out["Return_1d"] = out["Close"].pct_change()
    out["LogReturn_1d"] = np.log(out["Close"] / out["Close"].shift(1))

    for w in (5, 10, 20, 50):
        out[f"SMA_{w}"] = out["Close"].rolling(w).mean()
        out[f"STD_{w}"] = out["Close"].rolling(w).std()

    out["EMA_12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA_26"] = out["Close"].ewm(span=26, adjust=False).mean()

    macd_line, signal_line, hist = _macd(out["Close"])
    out["MACD"] = macd_line
    out["MACD_Signal"] = signal_line
    out["MACD_Hist"] = hist

    out["RSI_14"] = _rsi(out["Close"], 14)

    # Bollinger Bands
    out["BB_Mid"] = out["SMA_20"]
    out["BB_Upper"] = out["BB_Mid"] + 2 * out["STD_20"]
    out["BB_Lower"] = out["BB_Mid"] - 2 * out["STD_20"]

    # Price range / volume features
    out["High_Low_Pct"] = (out["High"] - out["Low"]) / out["Close"]
    out["Volume_Change"] = out["Volume"].pct_change()
    out["Volume_SMA_10"] = out["Volume"].rolling(10).mean()

    # Lagged closing prices (autoregressive signal)
    for lag in (1, 2, 3, 5):
        out[f"Close_Lag_{lag}"] = out["Close"].shift(lag)

    return out


def build_feature_target(df: pd.DataFrame, horizon: int = 1):
    """
    Adds technical indicators, builds the target column (Close price
    `horizon` days ahead), and drops rows with NaNs created by rolling
    windows / shifting.

    Returns
    -------
    features_df : DataFrame indexed like df, feature columns only
    target : Series, the prediction target
    full_df : DataFrame with everything (features + target + Date/Close)
    """
    feat = add_technical_indicators(df)
    feat["Target"] = feat["Close"].shift(-horizon)

    full = feat.dropna().reset_index(drop=True)

    non_feature_cols = {"Date", "Target", "Open", "High", "Low", "Close", "Volume"}
    feature_cols = [c for c in full.columns if c not in non_feature_cols]

    X = full[feature_cols]
    y = full["Target"]

    return X, y, full, feature_cols
