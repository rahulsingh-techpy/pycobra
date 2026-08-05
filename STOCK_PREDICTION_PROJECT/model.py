"""
model.py
--------
Model training, evaluation, and persistence utilities.

Two models are provided out of the box:
    - Linear Regression   (fast, interpretable baseline)
    - Random Forest       (captures non-linear patterns, gives
                            feature importances)

Both are evaluated with a time-respecting train/test split (no
shuffling, since this is a time series).
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def time_series_split(X, y, test_size: float = 0.2):
    """Split preserving chronological order (no shuffling)."""
    n_test = int(len(X) * test_size)
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_scaled, y_train)
    return model, scaler


def train_random_forest(X_train, y_train, n_estimators=300, max_depth=8, random_state=42):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, None  # no scaler needed for tree models


def evaluate(model, scaler, X_test, y_test):
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    preds = model.predict(X_eval)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}
    return preds, metrics


def save_model(model, scaler, feature_cols, path_prefix: str):
    joblib.dump(model, f"{path_prefix}_model.joblib")
    if scaler is not None:
        joblib.dump(scaler, f"{path_prefix}_scaler.joblib")
    joblib.dump(feature_cols, f"{path_prefix}_features.joblib")


def load_model(path_prefix: str):
    model = joblib.load(f"{path_prefix}_model.joblib")
    try:
        scaler = joblib.load(f"{path_prefix}_scaler.joblib")
    except FileNotFoundError:
        scaler = None
    feature_cols = joblib.load(f"{path_prefix}_features.joblib")
    return model, scaler, feature_cols


def recursive_forecast(model, scaler, last_row: pd.Series, feature_cols, df_history: pd.DataFrame, days: int):
    """
    Forecast `days` steps ahead by iteratively predicting the next close,
    appending it to a rolling history, and recomputing technical
    indicators for the next step. This is a simple but effective way to
    get a multi-day forecast out of a single-step model.
    """
    from feature_engineering import add_technical_indicators

    history = df_history.copy()
    forecasts = []

    for _ in range(days):
        feat_hist = add_technical_indicators(history)
        row = feat_hist.iloc[[-1]][feature_cols]
        X_input = scaler.transform(row) if scaler is not None else row
        next_close = float(model.predict(X_input)[0])

        next_date = history["Date"].iloc[-1] + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:  # skip weekends
            next_date += pd.Timedelta(days=1)

        new_row = {
            "Date": next_date,
            "Open": history["Close"].iloc[-1],
            "High": max(history["Close"].iloc[-1], next_close),
            "Low": min(history["Close"].iloc[-1], next_close),
            "Close": next_close,
            "Volume": history["Volume"].iloc[-5:].mean(),
        }
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        forecasts.append((next_date, next_close))

    return forecasts
