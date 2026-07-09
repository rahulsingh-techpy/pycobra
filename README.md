# Stock Price Prediction App

A Python data-analytics application that predicts next-day (and multi-day)
stock closing prices using engineered technical indicators and machine
learning (Linear Regression and Random Forest).

## Features

- **Flexible data input**: real ticker data via `yfinance`, your own CSV,
  or auto-generated synthetic data (for offline demos/testing).
- **Technical indicator engineering**: SMA/EMA, MACD, RSI, Bollinger Bands,
  volatility, volume features, lagged closes.
- **Two models out of the box**: Linear Regression (fast baseline) and
  Random Forest (non-linear, gives feature importances). Easy to extend
  with your own model in `src/model.py`.
- **Time-respecting evaluation**: chronological train/test split (no
  shuffling), with MAE, RMSE, R², and MAPE.
- **Multi-day forecasting**: recursive forecasting to project prices
  several business days into the future.
- **Auto-generated plots**: price history, actual vs. predicted, feature
  importance, and forecast charts saved to `outputs/plots/`.
- **Model persistence**: trained models saved with `joblib` for reuse.

## Project Structure

```
stock_price_prediction/
├── README.md
├── requirements.txt
├── data/
│   └── sample_stock_data.csv     # bundled synthetic demo dataset
├── src/
│   ├── data_loader.py            # data acquisition (yfinance / CSV / synthetic)
│   ├── feature_engineering.py    # technical indicators + target creation
│   ├── model.py                  # training, evaluation, forecasting, persistence
│   ├── visualize.py              # plotting helpers
│   └── main.py                   # CLI entry point
└── outputs/
    ├── plots/                    # generated charts (PNG)
    ├── models/                   # saved model artifacts (.joblib)
    ├── metrics_summary.csv       # model comparison metrics
    └── forecast.csv              # future price predictions
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Try it instantly with the bundled sample data

```bash
python src/main.py --csv data/sample_stock_data.csv --model both --forecast-days 10
```

### 2. Use a real ticker (requires internet access)

```bash
python src/main.py --ticker AAPL --period 3y --model both --forecast-days 10
```

If `yfinance` can't reach the internet, the app automatically falls back
to synthetic demo data so it never crashes.

### 3. Use your own CSV

Your CSV must contain these columns (case-insensitive):
`Date, Open, High, Low, Close, Volume`

```bash
python src/main.py --csv path/to/your_data.csv --model rf --forecast-days 5
```

### CLI Options

| Flag              | Description                                         | Default |
|-------------------|------------------------------------------------------|---------|
| `--ticker`        | Stock ticker symbol (e.g. `AAPL`, `TSLA`)             | None    |
| `--csv`           | Path to a local OHLCV CSV file                        | None    |
| `--period`        | History period for yfinance download (`1y`, `3y`...)  | `3y`    |
| `--model`         | `linear`, `rf`, or `both`                             | `both`  |
| `--test-size`     | Fraction of data held out for testing                | `0.2`   |
| `--forecast-days` | Number of future business days to forecast            | `10`    |

## How It Works

1. **Data loading** — pulls OHLCV data from your chosen source.
2. **Feature engineering** — computes technical indicators (moving
   averages, MACD, RSI, Bollinger Bands, volatility, volume trends, lagged
   prices) and creates the next-day closing price as the prediction target.
3. **Train/test split** — chronological split so the model is always
   evaluated on data *after* what it trained on (avoids lookahead bias).
4. **Training & evaluation** — fits Linear Regression and/or Random
   Forest, reports MAE, RMSE, R², and MAPE.
5. **Forecasting** — the best-performing model (by RMSE) is used to
   recursively forecast closing prices for the requested number of future
   business days, re-deriving technical indicators at each step.
6. **Outputs** — plots, a metrics summary, a forecast table, and the
   trained model artifacts are written to `outputs/`.

## Extending the App

- **Add a new model**: implement a `train_xxx(X_train, y_train)` function
  in `src/model.py` returning `(model, scaler_or_None)`, then wire it into
  `main.py`'s `models_to_run` list.
- **Add new indicators**: extend `add_technical_indicators()` in
  `src/feature_engineering.py`.
- **Deep learning (LSTM)**: the modular structure (features → train →
  evaluate → forecast) is designed so you can swap in a Keras/PyTorch
  sequence model in `model.py` without touching the rest of the pipeline.

## Disclaimer

This project is for **educational purposes only**. Stock price prediction
is inherently uncertain; technical indicators and the models here do not
account for news, earnings, macroeconomic events, or market sentiment.
**Nothing in this app constitutes financial advice.** Do not use it as the
sole basis for real investment decisions.
