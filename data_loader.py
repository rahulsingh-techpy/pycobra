"""
data_loader.py
--------------
Handles data acquisition for the stock price prediction app.

Priority order:
    1. If a CSV path is given, load it directly.
    2. Else if a ticker is given, try to download real data via yfinance
       (requires internet access).
    3. Else, fall back to generating a realistic synthetic stock price
       series so the app can always be demoed / tested offline.
"""

import os
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """Load OHLCV data from a local CSV file."""
    df = pd.read_csv(csv_path)

    # Try to standardize column names (case-insensitive)
    df.columns = [c.strip().title() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df[REQUIRED_COLUMNS]


def load_from_yfinance(ticker: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """Download real OHLCV data using yfinance (needs internet access)."""
    import yfinance as yf

    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    data = data.reset_index()
    data = data.rename(columns={"Adj Close": "Adj_Close"})
    data["Date"] = pd.to_datetime(data["Date"])
    return data[["Date", "Open", "High", "Low", "Close", "Volume"]]


def generate_synthetic_data(
    n_days: int = 1000,
    start_price: float = 150.0,
    annual_drift: float = 0.08,
    annual_volatility: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic-looking synthetic OHLCV series using
    Geometric Brownian Motion plus mild weekly seasonality and
    volume noise. Useful for offline demos and testing.
    """
    rng = np.random.default_rng(seed)

    dt = 1 / 252
    drift = (annual_drift - 0.5 * annual_volatility ** 2) * dt
    shock_scale = annual_volatility * np.sqrt(dt)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    log_returns = drift + shock_scale * rng.standard_normal(n_days)
    # add slight weekly seasonality (Monday dip / Friday rally, illustrative only)
    weekday_effect = dates.weekday.map({0: -0.0005, 1: 0.0001, 2: 0.0002, 3: 0.0001, 4: 0.0004}).fillna(0).to_numpy()
    log_returns = log_returns + weekday_effect

    close = start_price * np.exp(np.cumsum(log_returns))

    daily_range_pct = np.abs(rng.normal(0.012, 0.006, n_days))
    high = close * (1 + daily_range_pct * rng.uniform(0.4, 1.0, n_days))
    low = close * (1 - daily_range_pct * rng.uniform(0.4, 1.0, n_days))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n_days)

    base_volume = 5_000_000
    volume = base_volume * (1 + 0.5 * np.abs(rng.standard_normal(n_days))) * (
        1 + 2 * np.abs(log_returns) * 20
    )

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Close": close.round(2),
            "Volume": volume.astype(int),
        }
    )
    return df


def load_data(ticker: str = None, csv_path: str = None, period: str = "3y") -> pd.DataFrame:
    """
    Unified entry point. Returns a clean OHLCV DataFrame sorted by date.
    """
    if csv_path and os.path.exists(csv_path):
        print(f"[data_loader] Loading data from CSV: {csv_path}")
        return load_from_csv(csv_path)

    if ticker:
        try:
            print(f"[data_loader] Attempting to download '{ticker}' via yfinance...")
            return load_from_yfinance(ticker, period=period)
        except Exception as e:
            print(f"[data_loader] yfinance download failed ({e}). Falling back to synthetic data.")

    print("[data_loader] No ticker/CSV available or reachable — generating synthetic demo data.")
    return generate_synthetic_data()
