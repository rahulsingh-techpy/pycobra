"""
visualize.py
------------
Plotting helpers. All functions save PNGs to the given output path
(no interactive display needed, so this works in headless environments).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_price_history(df: pd.DataFrame, out_path: str, ticker_label: str = "Stock"):
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Close"], label="Close Price", color="#2563eb", linewidth=1.2)
    if "SMA_20" in df.columns:
        plt.plot(df["Date"], df["SMA_20"], label="SMA 20", color="#f59e0b", linewidth=1)
    if "SMA_50" in df.columns:
        plt.plot(df["Date"], df["SMA_50"], label="SMA 50", color="#10b981", linewidth=1)
    plt.title(f"{ticker_label} — Price History")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_actual_vs_predicted(dates, y_true, y_pred, out_path: str, model_name: str = "Model"):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual", color="#111827", linewidth=1.5)
    plt.plot(dates, y_pred, label="Predicted", color="#ef4444", linewidth=1.3, linestyle="--")
    plt.title(f"Actual vs Predicted Closing Price — {model_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_cols, out_path: str, top_n: int = 15):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(9, 6))
    plt.barh(
        [feature_cols[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#6366f1",
    )
    plt.title("Top Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_forecast(history_dates, history_close, forecast_dates, forecast_close, out_path: str):
    plt.figure(figsize=(12, 5))
    plt.plot(history_dates, history_close, label="Historical Close", color="#111827")
    plt.plot(forecast_dates, forecast_close, label="Forecast", color="#ef4444", linestyle="--", marker="o", markersize=3)
    plt.axvline(x=history_dates.iloc[-1], color="gray", linestyle=":", linewidth=1)
    plt.title("Future Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
