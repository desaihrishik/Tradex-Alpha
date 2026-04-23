"""
Build labeled dataset for NVDA using daily OHLCV data.

- Loads backend/data/nvda_daily_5y.csv
- Adds:
    * trend & return features
    * volume features
    * RSI
    * candlestick patterns (detailed)
    * structural swing patterns (double top/bottom, H&S, triangles, wedges, flags)
- Defines BUY / HOLD / SELL target based on next-day return
- Saves processed dataset to backend/data/nvda_daily_dataset.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .pattern_detection import add_candlestick_patterns, add_structural_patterns


ROOT_DIR = Path(__file__).resolve().parents[1]  # backend/
DATA_DIR = ROOT_DIR / "data"
RAW_CSV = DATA_DIR / "nvda_daily_5y.csv"
OUT_CSV = DATA_DIR / "nvda_daily_dataset.csv"


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val


def build_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # Runtime market history may include source/provider metadata.
    # Keep the training frame numeric so sklearn never sees provider strings.
    for optional_col in ("Source", "source"):
        if optional_col in df.columns:
            df = df.drop(columns=[optional_col])

    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw data: {missing}")

    if "Dividends" not in df.columns:
        df["Dividends"] = 0.0
    if "Stock Splits" not in df.columns:
        df["Stock Splits"] = 0.0

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["return_1d"] = df["Close"].pct_change()
    df["return_3d"] = df["Close"].pct_change(3)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    df["ema_5"] = ema(df["Close"], span=5)
    df["ema_12"] = ema(df["Close"], span=12)
    df["ema_26"] = ema(df["Close"], span=26)

    df["ema_5_dist"] = (df["Close"] - df["ema_5"]) / df["ema_5"]
    df["ema_12_dist"] = (df["Close"] - df["ema_12"]) / df["ema_12"]
    df["ema_26_dist"] = (df["Close"] - df["ema_26"]) / df["ema_26"]

    df["vol_change_1d"] = df["Volume"].pct_change()
    df["vol_ma_5"] = df["Volume"].rolling(5).mean()
    df["vol_vs_ma5"] = np.where(df["vol_ma_5"] > 0, df["Volume"] / df["vol_ma_5"], 1.0)

    df["rsi_14"] = rsi(df["Close"], period=14)
    df = add_candlestick_patterns(df)
    df = add_structural_patterns(df, lookback=60)

    df = df.dropna().reset_index(drop=True)
    return df


def build_dataset():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_CSV}. Run download_data.py first.")

    raw_df = pd.read_csv(RAW_CSV)
    df = build_feature_frame(raw_df)

    # ==== Target: BUY / HOLD / SELL based on next-day return ====
    future_return = df["Close"].shift(-1) / df["Close"] - 1.0
    df["future_return_1d"] = future_return

    buy_thresh = 0.005   # +0.5%
    sell_thresh = -0.005 # -0.5%

    conditions = [
        future_return > buy_thresh,
        future_return < sell_thresh
    ]
    choices = [1, -1]
    df["signal"] = np.select(conditions, choices, default=0)  # 1=BUY, -1=SELL, 0=HOLD

    # Drop rows with NaNs caused by indicators
    df = df.dropna().reset_index(drop=True)

    # Collect feature columns programmatically
    # Exclude only Date, future_return_1d, signal
    feature_cols = [
        c for c in df.columns
        if c not in ("Date", "future_return_1d", "signal")
    ]

    output_cols = ["Date"] + feature_cols + ["future_return_1d", "signal"]
    dataset = df[output_cols].copy()

    dataset.to_csv(OUT_CSV, index=False)

    print(f"Saved processed dataset to: {OUT_CSV}")
    print("\nSample rows:")
    print(dataset.head())

    print("\nClass distribution (signal):")
    print(dataset["signal"].value_counts().sort_index())
    print("\nLegend: -1=SELL, 0=HOLD, 1=BUY")

    print("\nNumber of feature columns:", len(feature_cols))


if __name__ == "__main__":
    build_dataset()
