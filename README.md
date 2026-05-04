# SP500_Prediction

A machine-learning project that predicts the **next-day direction** (up / down) of the S&P 500 index using historical price data and technical indicators, powered by a **Random Forest classifier**.

---

## Features

| Category | Indicators |
|---|---|
| Returns | 1-day, 5-day, 20-day percentage returns |
| Moving averages | SMA & EMA (5, 10, 20, 50, 200 days) |
| Bollinger Bands | Upper/lower bands, width, %B |
| Momentum | RSI (14-day), MACD, MACD signal & histogram |
| Volume | Volume / 20-day average volume ratio |
| Volatility | 20-day realized volatility |

## Requirements

- Python ≥ 3.10
- See [`requirements.txt`](requirements.txt) for Python package dependencies.

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/Fil126/SP500_Prediction.git
cd SP500_Prediction

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the prediction script
python sp500_prediction.py
```

The script will:
1. Download S&P 500 data from Yahoo Finance (2010 – 2024).
2. Compute technical features.
3. Train a Random Forest classifier on the first 80 % of the data.
4. Evaluate on the remaining 20 % and print a classification report.
5. Save diagnostic plots to `sp500_results.png`.

## Output example

```
[1/5] Downloading ^GSPC data (2010-01-01 – 2024-12-31) …
    Downloaded 3,773 rows.
[2/5] Engineering features …
    Feature set: 3,572 rows × 47 columns.
[3/5] Training Random Forest classifier …
[4/5] Evaluating model …

── Classification Report ─────────────────────────────
              precision    recall  f1-score   support
        Down       0.49      0.47      0.48       336
          Up       0.55      0.57      0.56       379
    accuracy                           0.52       715
   macro avg       0.52      0.52      0.52       715
weighted avg       0.52      0.52      0.52       715

[5/5] Generating plots …
    Plots saved to 'sp500_results.png'.
```

## Project structure

```
SP500_Prediction/
├── sp500_prediction.py   # main script
├── requirements.txt      # Python dependencies
└── README.md
```

## License

This project is released for educational purposes.