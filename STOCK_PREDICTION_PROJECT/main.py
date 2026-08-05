"""
main.py
-------
Command-line entry point for the Stock Price Prediction app.

Examples
--------
# Use a ticker (needs internet; falls back to synthetic data if offline)
python src/main.py --ticker AAPL --model both --forecast-days 10

# Use your own CSV (must have Date, Open, High, Low, Close, Volume columns)
python src/main.py --csv data/my_stock.csv --model rf --forecast-days 5

# Just try it out with generated demo data
python src/main.py --model both
"""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(__file__))

from data_loader import load_data
from feature_engineering import build_feature_target
from model import (
    evaluate,
    recursive_forecast,
    save_model,
    time_series_split,
    train_linear_regression,
    train_random_forest,
)
from visualize import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_forecast,
    plot_price_history,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")


def parse_args():
    p = argparse.ArgumentParser(description="Stock Price Prediction App")
    p.add_argument("--ticker", type=str, default=None, help="Stock ticker symbol, e.g. AAPL")
    p.add_argument("--csv", type=str, default=None, help="Path to a local OHLCV CSV file")
    p.add_argument("--period", type=str, default="3y", help="History period for yfinance download (e.g. 1y, 3y, 5y)")
    p.add_argument(
        "--model",
        type=str,
        choices=["linear", "rf", "both"],
        default="both",
        help="Which model(s) to train",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Fraction of data held out for testing")
    p.add_argument("--forecast-days", type=int, default=10, help="Number of future business days to forecast")
    return p.parse_args()


def run_pipeline(model_name, X_train, X_test, y_train, y_test, full, feature_cols, ticker_label):
    print(f"\n=== Training {model_name.upper()} model ===")
    if model_name == "linear":
        model, scaler = train_linear_regression(X_train, y_train)
    else:
        model, scaler = train_random_forest(X_train, y_train)

    preds, metrics = evaluate(model, scaler, X_test, y_test)
    print(f"{model_name.upper()} metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    test_dates = full["Date"].iloc[-len(y_test):]
    plot_path = os.path.join(PLOTS_DIR, f"{model_name}_actual_vs_predicted.png")
    plot_actual_vs_predicted(test_dates, y_test.values, preds, plot_path, model_name.upper())
    print(f"Saved plot: {plot_path}")

    if model_name == "rf":
        fi_path = os.path.join(PLOTS_DIR, "rf_feature_importance.png")
        plot_feature_importance(model, feature_cols, fi_path)
        print(f"Saved plot: {fi_path}")

    save_model(model, scaler, feature_cols, os.path.join(MODELS_DIR, model_name))
    print(f"Saved model artifacts to: {MODELS_DIR}/{model_name}_*.joblib")

    return model, scaler, metrics


def main():
    args = parse_args()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    ticker_label = args.ticker or (os.path.basename(args.csv) if args.csv else "Synthetic Demo Data")

    df = load_data(ticker=args.ticker, csv_path=args.csv, period=args.period)
    print(f"\nLoaded {len(df)} rows from {df['Date'].min().date()} to {df['Date'].max().date()}")

    X, y, full, feature_cols = build_feature_target(df, horizon=1)
    print(f"Built {X.shape[1]} features across {len(X)} usable rows after cleaning.")

    hist_plot_path = os.path.join(PLOTS_DIR, "price_history.png")
    plot_price_history(full, hist_plot_path, ticker_label)
    print(f"Saved plot: {hist_plot_path}")

    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=args.test_size)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    models_to_run = ["linear", "rf"] if args.model == "both" else [args.model]
    results = {}
    best_model, best_scaler, best_name = None, None, None
    best_rmse = float("inf")

    for m in models_to_run:
        model, scaler, metrics = run_pipeline(m, X_train, X_test, y_train, y_test, full, feature_cols, ticker_label)
        results[m] = metrics
        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_model, best_scaler, best_name = model, scaler, m

    print("\n=== Summary ===")
    summary_df = pd.DataFrame(results).T
    print(summary_df.round(4))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"))

    print(f"\nBest model by RMSE: {best_name.upper()}")

    if args.forecast_days > 0:
        print(f"\n=== Forecasting next {args.forecast_days} business days with {best_name.upper()} ===")
        forecasts = recursive_forecast(
            best_model, best_scaler, full.iloc[-1], feature_cols, df, args.forecast_days
        )
        for d, price in forecasts:
            print(f"  {d.date()}: {price:.2f}")

        forecast_df = pd.DataFrame(forecasts, columns=["Date", "Predicted_Close"])
        forecast_df.to_csv(os.path.join(OUTPUT_DIR, "forecast.csv"), index=False)

        forecast_plot_path = os.path.join(PLOTS_DIR, "forecast.png")
        plot_forecast(
            df["Date"].iloc[-60:], df["Close"].iloc[-60:],
            forecast_df["Date"], forecast_df["Predicted_Close"],
            forecast_plot_path,
        )
        print(f"Saved plot: {forecast_plot_path}")
        print(f"Saved forecast table: {os.path.join(OUTPUT_DIR, 'forecast.csv')}")

    print("\nDone. See the 'outputs/' folder for plots, models, and CSV results.")


if __name__ == "__main__":
    main()
