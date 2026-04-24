"""
Signal engine for NVDA.

- Loads latest processed row from nvda_daily_dataset.csv
- Loads trained RandomForest model
- Computes BUY / HOLD / SELL with confidence
- Detects active candlestick patterns for the latest day
- Suggests number of shares to trade for a given budget + risk profile

This is a pure Python helper. Later, we'll wrap this in a REST API
so the frontend can call it.
"""

from pathlib import Path
import json
from typing import Literal, Dict, Any

import numpy as np
import pandas as pd
from joblib import load

from src.build_dataset import build_feature_frame
from src.data.repositories.market_data import LocalMarketDataRepository
from src.data.repositories.sentiment_repository import SentimentSnapshotRepository


ROOT_DIR = Path(__file__).resolve().parents[1]   # backend/
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

DATASET_CSV = DATA_DIR / "nvda_daily_dataset.csv"
MODEL_PATH = MODEL_DIR / "nvda_rf_signal.pkl"
PRIMARY_META_PATH = MODEL_DIR / "nvda_rf_signal_metadata.json"
LEGACY_META_PATH = MODEL_DIR / "nvda_rf_metadata.json"

RiskProfile = Literal["low", "medium", "high"]

BULLISH_PATTERNS = {
    "hammer",
    "bullish_engulfing",
    "morning_star",
    "three_white_soldiers",
    "piercing",
    "bullish_harami",
    "struct_double_bottom",
    "struct_inverse_head_shoulders",
    "struct_bull_flag",
    "struct_ascending_triangle",
    "struct_falling_wedge",
}

BEARISH_PATTERNS = {
    "hanging_man",
    "shooting_star",
    "bearish_engulfing",
    "dark_cloud_cover",
    "bearish_harami",
    "evening_star",
    "three_black_crows",
    "struct_double_top",
    "struct_head_shoulders",
    "struct_bear_flag",
    "struct_descending_triangle",
    "struct_rising_wedge",
}


def load_model_and_data():
    metadata_path = PRIMARY_META_PATH if PRIMARY_META_PATH.exists() else LEGACY_META_PATH

    if not MODEL_PATH.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Model or metadata not found. Run train_model.py first."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    raw_label_mapping: Dict[str, str] = meta["label_mapping"]
    # convert key strings back to ints: {"-1": "SELL"} → {-1: "SELL"}
    label_mapping = {int(k): v for k, v in raw_label_mapping.items()}

    clf = load(MODEL_PATH)
    if hasattr(clf, "n_jobs"):
        clf.n_jobs = 1

    repository = LocalMarketDataRepository()
    try:
        raw_df = repository.get_raw_market_history(ticker="NVDA", limit=500)
        df = build_feature_frame(raw_df)
    except Exception:
        if not DATASET_CSV.exists():
            raise FileNotFoundError(
                f"Dataset not found at {DATASET_CSV}. Run build_dataset.py first."
            )
        df = pd.read_csv(DATASET_CSV, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df, clf, feature_cols, label_mapping


def compute_position_size(
    budget: float,
    risk_profile: RiskProfile,
    confidence: float,
    price: float,
) -> int:
    """
    Very simple position sizing rule:
    - Allocate a fraction of budget based on risk_profile
    - Scale by model confidence
    - Convert to integer number of shares (floor)
    - Ensure we can still take a tiny position if there is a BUY signal
    """

    # Basic sanity check
    if budget <= 0 or price <= 0:
        return 0

    # Base risk fraction by profile
    if risk_profile == "low":
        base_frac = 0.25
    elif risk_profile == "medium":
        base_frac = 0.5
    else:  # "high"
        base_frac = 0.9

    # Clamp confidence between 0 and 1
    confidence = max(0.0, min(1.0, confidence))

    # Scale by confidence (0–1)
    capital_to_use = budget * base_frac * confidence

    # Raw integer share count
    shares = int(capital_to_use // price)

    # --- MINIMUM POSITION RULE (this is the new bit) ---
    # If this rounds to 0 shares but we still have a meaningful fraction
    # of a share's price, allow 1 share as a tiny test position.
    # Example: capital_to_use is at least 20% of a share.
    if shares <= 0:
        min_cap_threshold = 0.20  # 20% of price
        if capital_to_use / price >= min_cap_threshold:
            shares = 1
        else:
            shares = 0
    # ---------------------------------------------------

    if shares < 0:
        shares = 0
    return shares


def summarize_pattern_bias(active_patterns: list[str]) -> str:
    bullish = [pattern for pattern in active_patterns if pattern in BULLISH_PATTERNS]
    bearish = [pattern for pattern in active_patterns if pattern in BEARISH_PATTERNS]

    if bullish and not bearish:
        return "bullish"
    if bearish and not bullish:
        return "bearish"
    if bullish or bearish:
        return "mixed"
    return "neutral"


def apply_risk_decision_policy(
    *,
    probas_by_label: Dict[str, float],
    risk_profile: RiskProfile,
    pattern_bias: str = "neutral",
    trend_strength: float = 0.0,
    sentiment_strength: float = 0.0,
) -> tuple[str, str | None]:
    buy = float(probas_by_label.get("BUY", 0.0))
    hold = float(probas_by_label.get("HOLD", 0.0))
    sell = float(probas_by_label.get("SELL", 0.0))

    base_label = max(probas_by_label, key=probas_by_label.get)

    if risk_profile == "low":
        if base_label == "BUY" and (buy < 0.45 or buy - max(hold, sell) < 0.06):
            return "HOLD", "Low-risk policy softened a borderline BUY into HOLD."
        if base_label == "SELL" and (sell < 0.45 or sell - max(hold, buy) < 0.06):
            return "HOLD", "Low-risk policy softened a borderline SELL into HOLD."
        return base_label, None

    if risk_profile == "high":
        bullish_support = pattern_bias == "bullish" or trend_strength > 0.15 or sentiment_strength > 0.10
        bearish_support = pattern_bias == "bearish" or trend_strength < -0.15 or sentiment_strength < -0.10

        if base_label == "SELL" and (sell - buy) < 0.04 and buy >= 0.30 and bullish_support:
            if buy >= hold - 0.02:
                return "BUY", "High-risk policy allowed a contrarian BUY because the bearish edge was weak and upside support was present."
            return "HOLD", "High-risk policy rejected a weak SELL because bullish support was present."

        if base_label == "BUY" and (buy - sell) < 0.04 and sell >= 0.30 and bearish_support:
            return "HOLD", "High-risk policy treated the weak BUY as HOLD because downside evidence was still meaningful."

        if base_label == "HOLD":
            if buy >= 0.33 and buy >= sell and bullish_support:
                return "BUY", "High-risk policy promoted HOLD to BUY because upside evidence was acceptable."
            if sell >= 0.33 and sell >= buy and bearish_support:
                return "SELL", "High-risk policy promoted HOLD to SELL because downside evidence was acceptable."

    return base_label, None


def compute_sentiment_strength(label: str, score: float) -> float:
    if label == "bullish":
        return max(0.0, min(0.5, score * 0.5))
    if label == "bearish":
        return max(-0.5, min(0.0, score * 0.5))
    return 0.0


def build_decision_text(
    *,
    action: str,
    top_margin: float,
    pattern_bias: str,
    active_patterns: list[str],
    policy_note: str | None = None,
) -> str:
    if action == "BUY":
        if top_margin < 0.05:
            decision_text = "Model narrowly prefers BUY, but the edge is weak and signals remain mixed."
        elif pattern_bias == "bearish":
            decision_text = "Model leans BUY, but bearish pattern signals mean this setup has conflicting evidence."
        else:
            decision_text = "Model suggests BUY based on current trend and indicators."
    elif action == "SELL":
        if top_margin < 0.05:
            decision_text = "Model narrowly prefers SELL, but the edge is weak and signals remain mixed."
        elif pattern_bias == "bullish":
            decision_text = "Model leans SELL, but bullish pattern signals mean this setup has conflicting evidence."
        else:
            decision_text = "Model suggests SELL; downside risk is elevated."
    else:
        decision_text = "Model suggests HOLD; no strong edge detected."

    if policy_note:
        decision_text += f" {policy_note}"
    if active_patterns:
        decision_text += f" Detected patterns: {', '.join(active_patterns)}."
    return decision_text




def get_latest_recommendation(
    budget: float,
    risk_profile: RiskProfile = "medium",
) -> Dict[str, Any]:
    """
    Core function: return recommendation for the latest NVDA candle.

    Returns a dict with:
    - date
    - latest_close
    - action ("BUY"/"SELL"/"HOLD")
    - signal_value (-1/0/1)
    - confidence (0-1)
    - probas per class
    - suggested_shares
    - capital_used
    - patterns (list of detected patterns for that day)
    - explanation (short text)
    """

    df, clf, feature_cols, label_mapping = load_model_and_data()

    # Latest row (most recent trading day)
    latest = df.iloc[-1]
    date = latest["Date"]
    price = float(latest["Close"])

    # Prepare feature vector in the same column order used for training
    X_latest = latest[feature_cols].values.reshape(1, -1)

    # Predict probabilities and class
    proba = clf.predict_proba(X_latest)[0]  # shape: (n_classes,)
    classes = clf.classes_                  # e.g. [-1, 0, 1]

    # Map probabilities to labels for more detailed display
    probas_by_label = {
        label_mapping.get(int(cls), str(int(cls))): float(p)
        for cls, p in zip(classes, proba)
    }

    # Detect which candlestick patterns are active
    pattern_cols = [c for c in df.columns if c.startswith("pattern_")]
    active_patterns = [
        c.replace("pattern_", "")
        for c in pattern_cols
        if int(latest[c]) == 1
    ]

    sorted_proba = sorted((float(p) for p in proba), reverse=True)
    top_margin = sorted_proba[0] - sorted_proba[1] if len(sorted_proba) > 1 else sorted_proba[0]
    pattern_bias = summarize_pattern_bias(active_patterns)
    sentiment_snapshot = SentimentSnapshotRepository().get_latest_snapshot("NVDA")
    sentiment_label = sentiment_snapshot.label
    sentiment_score = float(sentiment_snapshot.score)
    sentiment_strength = compute_sentiment_strength(sentiment_label, sentiment_score)
    best_label, policy_note = apply_risk_decision_policy(
        probas_by_label=probas_by_label,
        risk_profile=risk_profile,
        pattern_bias=pattern_bias,
        trend_strength=float(latest.get("ema_5_dist", 0.0)),
        sentiment_strength=sentiment_strength,
    )
    reverse_label_mapping = {label: value for value, label in label_mapping.items()}
    best_class = reverse_label_mapping.get(best_label, 0)
    best_conf = float(probas_by_label.get(best_label, 0.0))

    # Compute position size
    suggested_shares = 0
    if best_label == "BUY":
        suggested_shares = compute_position_size(
            budget=budget,
            risk_profile=risk_profile,
            confidence=best_conf,
            price=price,
        )
    # For SELL, this engine does not know user's holdings.
    # Frontend can decide to sell up to current position.

    capital_used = suggested_shares * price

    decision_text = build_decision_text(
        action=best_label,
        top_margin=top_margin,
        pattern_bias=pattern_bias,
        active_patterns=active_patterns,
        policy_note=policy_note,
    )

    formatted_date = pd.to_datetime(date).strftime("%Y-%m-%d")

    result = {
        "date": formatted_date,
        "latest_close": price,
        "action": best_label,
        "signal_value": best_class,
        "confidence": best_conf,
        "probas": probas_by_label,
        "suggested_shares": int(suggested_shares),
        "capital_used": capital_used,
        "patterns": active_patterns,
        "risk_profile": risk_profile,
        "budget": budget,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "sentiment_strength": sentiment_strength,
        # For debugging / future frontend: we can expose features too
        "features": {col: float(latest[col]) for col in feature_cols},
        "explanation": decision_text,
    }

    return result


if __name__ == "__main__":
    # Quick manual test
    example_budget = 1000.0
    example_risk: RiskProfile = "medium"

    rec = get_latest_recommendation(
        budget=example_budget,
        risk_profile=example_risk,
    )

    print("=== NVDA Recommendation ===")
    for k, v in rec.items():
        print(f"{k}: {v}")
