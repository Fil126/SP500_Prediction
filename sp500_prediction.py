"""
SP500 Prediction
================
Downloads historical S&P 500 data via yfinance, engineers technical features,
trains a Random Forest classifier to predict the next-day direction
(up / down), evaluates the model, and plots results.

Usage
-----
    python sp500_prediction.py
"""

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKER = "^GSPC"          # S&P 500 index
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
PREDICTION_HORIZON = 1    # days ahead to predict
TRAIN_RATIO = 0.80        # fraction of data used for training
RANDOM_STATE = 42
EPSILON = 1e-9           # small constant to avoid division by zero

# ---------------------------------------------------------------------------
# 1. Download data
# ---------------------------------------------------------------------------

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for *ticker* between *start* and *end*."""
    print(f"[1/5] Downloading {ticker} data ({start} – {end}) …")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        sys.exit("ERROR: No data downloaded. Check your internet connection or ticker symbol.")
    # Flatten MultiIndex columns produced by yfinance ≥ 0.2
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    print(f"    Downloaded {len(df):,} rows.")
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as predictive features."""
    print("[2/5] Engineering features …")
    close = df["Close"]

    # Returns
    df["Return_1d"] = close.pct_change(1)
    df["Return_5d"] = close.pct_change(5)
    df["Return_20d"] = close.pct_change(20)

    # Moving averages
    for window in (5, 10, 20, 50, 200):
        df[f"SMA_{window}"] = close.rolling(window).mean()
        df[f"EMA_{window}"] = close.ewm(span=window, adjust=False).mean()

    # Ratios of price to moving averages
    for window in (5, 20, 50, 200):
        df[f"Price_to_SMA_{window}"] = close / df[f"SMA_{window}"]

    # Bollinger Bands (20-day)
    sma20 = df["SMA_20"]
    std20 = close.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma20
    df["BB_pct"] = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # RSI (14-day)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + EPSILON)
    df["RSI_14"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Volume features
    df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_ratio"] = df["Volume"] / (df["Volume_SMA_20"] + EPSILON)

    # Volatility (realized, 20-day)
    df["Volatility_20d"] = df["Return_1d"].rolling(20).std()

    # Target: 1 if price goes up the next PREDICTION_HORIZON day(s), else 0
    df["Target"] = (close.shift(-PREDICTION_HORIZON) > close).astype(int)

    df.dropna(inplace=True)
    print(f"    Feature set: {len(df):,} rows × {len(df.columns)} columns.")
    return df


# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame, feature_cols: list[str]):
    """Chronological train/test split (no shuffling)."""
    split_idx = int(len(df) * TRAIN_RATIO)
    X = df[feature_cols].values
    y = df["Target"].values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, df.index[split_idx:]


# ---------------------------------------------------------------------------
# 4. Model training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train a Random Forest and print evaluation metrics."""
    print("[3/5] Training Random Forest classifier …")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("[4/5] Evaluating model …")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n── Classification Report ─────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))
    return model, y_pred, y_prob


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def plot_results(df, feature_cols, model, y_test, y_pred, y_prob, test_index):
    """Generate and save diagnostic plots."""
    print("[5/5] Generating plots …")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("S&P 500 Prediction – Random Forest", fontsize=15, fontweight="bold")

    # -- Price history & train/test split
    ax = axes[0, 0]
    close = df["Close"]
    split_date = test_index[0]
    ax.plot(close[:split_date], color="steelblue", linewidth=0.8, label="Train")
    ax.plot(close[split_date:], color="darkorange", linewidth=0.8, label="Test")
    ax.axvline(split_date, color="black", linestyle="--", linewidth=1)
    ax.set_title("S&P 500 Close Price")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- Confusion matrix
    ax = axes[0, 1]
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (Test Set)")

    # -- Predicted probability distribution
    ax = axes[1, 0]
    ax.hist(y_prob[y_test == 0], bins=30, alpha=0.6, color="crimson", label="Actual Down")
    ax.hist(y_prob[y_test == 1], bins=30, alpha=0.6, color="forestgreen", label="Actual Up")
    ax.axvline(0.5, color="black", linestyle="--")
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("P(Up)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- Feature importances (top 15)
    ax = axes[1, 1]
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top15 = importances.nlargest(15).sort_values()
    top15.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importances (Top 15)")
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = "sp500_results.png"
    plt.savefig(output_path, dpi=120)
    print(f"    Plots saved to '{output_path}'.")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df_raw = download_data(TICKER, START_DATE, END_DATE)
    df = add_features(df_raw.copy())

    feature_cols = [
        c for c in df.columns
        if c not in ("Open", "High", "Low", "Close", "Adj Close", "Volume", "Target")
    ]

    X_train, X_test, y_train, y_test, scaler, test_index = split_data(df, feature_cols)
    model, y_pred, y_prob = train_and_evaluate(X_train, X_test, y_train, y_test)
    plot_results(df, feature_cols, model, y_test, y_pred, y_prob, test_index)
    print("\nDone!")


if __name__ == "__main__":
    main()
