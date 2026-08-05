# 🛡️ Fraud Shield — Credit Card Fraud Detection Dashboard

Python + data-analytics application built with **Streamlit**, **pandas**, **scikit-learn**, and **matplotlib/seaborn**.

## Features
- Synthetic dataset generator (same schema as Kaggle's `creditcard.csv`: `Time`, `V1`–`V28`, `Amount`, `Class`)
- Interactive EDA — class balance, amount distribution, time patterns, top correlated features
- Two trained models (Logistic Regression + Random Forest) with ROC curve, confusion matrix, precision/recall trade-off
- Adjustable decision threshold to explore the precision vs. recall trade-off live
- Live transaction scoring — draw a random transaction and see its fraud probability
- Dark, gradient-accented UI theme with custom cards and badges

## How to run (apne computer par)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

It will open automatically at `http://localhost:8501`.

## Using your own real data
Replace the call to `generate_data(...)` near the top of `app.py` with:
```python
df = pd.read_csv("creditcard.csv")
```
as long as the CSV has the same column names (`Time`, `V1`...`V28`, `Amount`, `Class`), everything else works unchanged — including the popular [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Requirements
- Python 3.9+
- See `requirements.txt`
