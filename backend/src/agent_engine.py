"""
Agentic Swing Trading Engine for NVDA
-------------------------------------

This module builds a FULL agentic decision system that:
- uses RandomForest model predictions (BUY / HOLD / SELL)
- integrates candlestick & structural patterns
- integrates sentiment (Finnhub)
- calculates adaptive holding horizon (dynamic 1 to 40 days)
- performs Monte Carlo forecasting
- returns BUY / SELL / HOLD reasoning in human language
- computes potential profit, loss, CAGR, conviction score
- supports optional entry_price parameter for SELL analysis

Designed to be used by FastAPI or any backend module.

----------------------------------------
MAIN PUBLIC FUNCTION:
    get_agentic_recommendation(
        budget: float,
        risk_profile: Literal["low","medium","high"],
        entry_price: Optional[float] = None,
    )
----------------------------------------
"""

from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime, timedelta

from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Existing project imports
from src.signal_engine import (
    apply_risk_decision_policy,
    compute_position_size,
    load_model_and_data,
)
from src.data.repositories.sentiment_repository import SentimentSnapshotRepository


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

MAX_HORIZON = 40  # dynamic horizon upper limit for swing trading
MIN_HORIZON = 3   # never suggest ultra-short-term unless confidence is huge

# For Monte Carlo simulation
N_MONTE_CARLO_PATHS = 2000
FORECAST_DAYS = MAX_HORIZON


# -------------------------------------------------------------------
# BASIC HELPERS
# -------------------------------------------------------------------

def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def pct(a: float) -> str:
    """Format percentage 0.2334 -> '23.34%'"""
    return f"{a*100:.2f}%"


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def volatility(series: np.ndarray) -> float:
    """Annualized volatility."""
    if len(series) < 2:
        return 0.0
    return np.std(series) * np.sqrt(252)


# -------------------------------------------------------------------
# PATTERN STRENGTH WEIGHTING
# -------------------------------------------------------------------

PATTERN_WEIGHTS = {
    # Candlestick
    "hammer": 0.10,
    "hanging_man": -0.10,
    "bullish_engulfing": 0.20,
    "bearish_engulfing": -0.20,
    "morning_star": 0.25,
    "evening_star": -0.25,
    "three_white_soldiers": 0.35,
    "three_black_crows": -0.35,

    # Structural
    "struct_double_bottom": 0.40,
    "struct_double_top": -0.40,
    "struct_inverse_head_shoulders": 0.45,
    "struct_head_shoulders": -0.45,
    "struct_bull_flag": 0.30,
    "struct_bear_flag": -0.30,
    "struct_ascending_triangle": 0.20,
    "struct_descending_triangle": -0.20,
    "struct_rising_wedge": -0.20,
    "struct_falling_wedge": 0.20,
    "struct_sym_triangle": 0.05,
}


def compute_pattern_strength(patterns: List[str]) -> float:
    """Convert detected patterns into a combined strength score."""
    score = 0.0
    for p in patterns:
        base = PATTERN_WEIGHTS.get(p, 0)
        score += base
    return clamp(score, -1.0, 1.0)


# -------------------------------------------------------------------
# SENTIMENT -> NUMERIC IMPACT
# -------------------------------------------------------------------

def sentiment_strength(sentiment_label: str, score: float) -> float:
    """
    Maps sentiment to numeric impact:
    """
    if sentiment_label == "bullish":
        return clamp(score * 0.5, 0, 0.50)
    if sentiment_label == "bearish":
        return clamp(score * -0.5, -0.50, 0)
    return 0.0  # neutral


# -------------------------------------------------------------------
# TREND METRICS FROM LATEST CANDLE
# -------------------------------------------------------------------

def compute_trend_strength(latest_row: pd.Series) -> float:
    """
    Combine simple trend rules:
    - EMA alignment
    - recent returns
    """
    score = 0

    # EMA trend
    if latest_row["ema_5"] > latest_row["ema_12"] > latest_row["ema_26"]:
        score += 0.4
    if latest_row["ema_5"] < latest_row["ema_12"] < latest_row["ema_26"]:
        score -= 0.4

    # short-term returns
    if latest_row["return_5d"] > 0:
        score += 0.2
    if latest_row["return_5d"] < 0:
        score -= 0.2

    return clamp(score, -1, 1)


# -------------------------------------------------------------------
# DYNAMIC HORIZON MODEL (0–40 DAYS)
# -------------------------------------------------------------------

def compute_dynamic_horizon(
    signal_value: int,
    model_confidence: float,
    pattern_strength: float,
    sentiment_strength_value: float,
    risk_profile: str,
) -> int:
    """
    Predict holding duration adaptively.
    All inputs matter now — fully dynamic.
    """

    # Start from baseline (BUY = medium holding, HOLD shorter, SELL 0–small)
    if signal_value == 1:     # BUY
        base = 18
    elif signal_value == 0:   # HOLD
        base = 10
    else:                     # SELL
        base = 5

    # Influence by model confidence
    base += (model_confidence - 0.5) * 35  # ±17 days

    # Influence by patterns
    base += pattern_strength * 20          # ±20 days

    # Influence by sentiment
    base += sentiment_strength_value * 20  # ±10 days

    # Influence by risk
    if risk_profile == "low":
        base -= 4
    elif risk_profile == "high":
        base += 4

    horizon = int(clamp(base, MIN_HORIZON, MAX_HORIZON))
    return horizon


# -------------------------------------------------------------------
# MONTE CARLO FORECASTING
# -------------------------------------------------------------------

def monte_carlo_forecast(
    last_price: float,
    log_returns: np.ndarray,
    days: int,
    paths: int = N_MONTE_CARLO_PATHS,
) -> Dict[str, float]:
    """
    Simulate many price paths using last year's log returns distribution.
    """
    if len(log_returns) < 2:
        return {
            "p10": last_price,
            "p50": last_price,
            "p90": last_price,
        }

    mu = np.mean(log_returns)
    sigma = np.std(log_returns)

    results = np.zeros(paths)

    for i in range(paths):
        path = np.random.normal(mu, sigma, days)
        price_path = last_price * np.exp(np.cumsum(path))
        results[i] = price_path[-1]

    p10 = float(np.percentile(results, 10))
    p50 = float(np.percentile(results, 50))
    p90 = float(np.percentile(results, 90))

    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,
    }


def _next_trading_dates(last_date: str, days: int) -> list[str]:
    current = pd.to_datetime(last_date).to_pydatetime()
    current = current.replace(hour=0, minute=0, second=0, microsecond=0)
    dates: list[str] = []
    while len(dates) < days:
        current += timedelta(days=1)
        if current.weekday() >= 5:
            continue
        dates.append(current.strftime("%Y-%m-%d"))
    return dates


def logistic_trend_forecast(
    prices: np.ndarray,
    *,
    last_date: str,
    horizon_days: int,
    lags: int = 5,
) -> Dict[str, Any]:
    """
    Fit a lightweight logistic regression on recent return momentum and
    roll the predicted direction forward into a single expected trend line.
    """
    if len(prices) < lags + 8:
        return {
            "dates": _next_trading_dates(last_date, horizon_days),
            "values": [],
            "direction_probability": 0.5,
            "direction": "sideways",
        }

    log_returns = np.diff(np.log(prices))
    if len(log_returns) <= lags + 1:
        return {
            "dates": _next_trading_dates(last_date, horizon_days),
            "values": [],
            "direction_probability": 0.5,
            "direction": "sideways",
        }

    X: list[list[float]] = []
    y: list[int] = []
    y_reg: list[float] = []
    for idx in range(lags, len(log_returns)):
        X.append(log_returns[idx - lags:idx].tolist())
        y.append(1 if log_returns[idx] > 0 else 0)
        y_reg.append(float(log_returns[idx]))

    if len(set(y)) < 2:
        return {
            "dates": _next_trading_dates(last_date, horizon_days),
            "values": [],
            "direction_probability": 0.5,
            "direction": "sideways",
        }

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(np.asarray(X), np.asarray(y))
    reg_model = LinearRegression().fit(np.asarray(X), np.asarray(y_reg))

    recent_window = log_returns[-lags:].tolist()
    prob_up = float(model.predict_proba([recent_window])[0][1])
    recent_sigma = float(np.std(log_returns[-min(len(log_returns), 30):]))
    recent_sigma = max(recent_sigma, 0.0025)

    trend_window = min(len(prices), 20)
    x_fit = np.arange(trend_window).reshape(-1, 1)
    y_fit = prices[-trend_window:]
    trend_model = LinearRegression().fit(x_fit, y_fit)
    trend_slope_pct = float(trend_model.coef_[0] / max(float(y_fit[-1]), 1e-6))
    recent_return_5 = float(np.mean(log_returns[-min(len(log_returns), 5):]))
    recent_return_10 = float(np.mean(log_returns[-min(len(log_returns), 10):]))
    blended_drift = (recent_return_5 * 0.6) + (recent_return_10 * 0.4)
    base_drift = (trend_slope_pct * 0.9) + (blended_drift * 0.1)

    current_price = float(prices[-1])
    values = [round(current_price, 2)]
    working_window = recent_window.copy()
    for step in range(1, horizon_days + 1):
        step_prob = float(model.predict_proba([working_window])[0][1])
        direction_bias = (step_prob - 0.5) * 2.0
        logistic_return = float(reg_model.predict([working_window])[0])
        momentum_bias = base_drift * 0.9
        volatility_bias = recent_sigma * direction_bias * 0.75
        decay = max(0.60, 1.0 - (step - 1) * 0.08)
        expected_return = (
            (logistic_return * 0.45)
            + (momentum_bias * 0.40)
            + (volatility_bias * 0.15)
        ) * decay
        expected_return = float(np.clip(expected_return, -0.018, 0.018))
        current_price = max(0.01, current_price * float(np.exp(expected_return)))
        current_price = round(current_price, 2)
        values.append(current_price)
        working_window = (working_window[1:] + [expected_return]) if len(working_window) >= lags else working_window + [expected_return]

    direction = "bullish" if prob_up >= 0.55 else "bearish" if prob_up <= 0.45 else "sideways"
    future_dates = _next_trading_dates(last_date, horizon_days)
    return {
        "dates": future_dates,
        "values": values[1:],
        "direction_probability": prob_up,
        "direction": direction,
    }

# ===================================================================
# PnL & CAPITAL ANALYSIS
# ===================================================================

def compute_sell_analysis(
    entry_price: float,
    current_price: float,
    horizon_forecast: Dict[str, float],
) -> Dict[str, Any]:
    """
    When SELL action is recommended and user provides entry price,
    compute:
    - saved profit (if any)
    - drawdown risk
    - distribution of forecasted outcomes
    """
    p10 = horizon_forecast["p10"]
    p50 = horizon_forecast["p50"]
    p90 = horizon_forecast["p90"]

    current_return = safe_div(current_price - entry_price, entry_price)

    # If user holds, forecast outcomes:
    downside_if_hold = safe_div(p10 - current_price, current_price)
    midpoint_future = safe_div(p50 - current_price, current_price)
    upside_if_hold = safe_div(p90 - current_price, current_price)

    return {
        "current_return": current_return,
        "potential_downside_if_hold": downside_if_hold,
        "expected_future_ret": midpoint_future,
        "best_case_ret": upside_if_hold,
    }


# ===================================================================
# Human-like Explanation Builders
# ===================================================================

def explain_buy(
    horizon: int,
    confidence: float,
    pat_strength: float,
    sent_label: str,
    sent_score: float,
    forecast: Dict[str, float],
) -> str:
    """
    Builds natural language BUY explanation.
    """
    p50 = forecast["p50"]
    p10 = forecast["p10"]
    p90 = forecast["p90"]

    explanation = (
        f"The model is detecting a BUY opportunity driven by improving trend structure "
        f"and a confidence level of {pct(confidence)}. "
    )

    if confidence < 0.40:
        explanation += (
            "This is a low-conviction signal, so upside expectations should be treated cautiously. "
        )

    # Patterns
    if pat_strength > 0.25:
        explanation += (
            "Strong bullish structural/candlestick patterns add conviction, "
            "suggesting buyers are gaining control. "
        )
    elif pat_strength > 0:
        explanation += (
            "Mild bullish patterns support the upside. "
        )

    # Sentiment
    if sent_label == "bullish":
        explanation += (
            f"Market sentiment is also positive ({pct(sent_score)} bullish), providing additional confirmation. "
        )
    elif sent_label == "bearish":
        explanation += (
            "Some bearish sentiment is present, but the model believes the technical structure outweighs it. "
        )

    explanation += (
        f"Projected median price after ~{horizon} days is around ${p50:.2f}, with a best-case scenario near ${p90:.2f} "
        f"and a downside floor around ${p10:.2f}. "
        f"This makes it a reasonable swing opportunity over the next {horizon} days."
    )

    return explanation


def explain_hold(
    horizon: int,
    confidence: float,
    pat_strength: float,
    sent_label: str,
    sent_score: float,
    forecast: Dict[str, float],
) -> str:
    """
    HOLD explanation.
    """
    p10 = forecast["p10"]
    p50 = forecast["p50"]
    p90 = forecast["p90"]

    explanation = (
        "The model suggests HOLD because signals are mixed and no strong directional edge exists right now. "
    )

    # Patterns
    if abs(pat_strength) < 0.1:
        explanation += "Pattern structure is neutral. "
    elif pat_strength > 0:
        explanation += "There is some bullish pattern support, but not enough for a confident entry. "
    else:
        explanation += "Some bearish structures appear but lack strong confirmation. "

    # Sentiment
    if sent_label == "bullish":
        explanation += "Market sentiment is mildly supportive. "
    elif sent_label == "bearish":
        explanation += "Market sentiment is weighted to the downside, reinforcing caution. "

    explanation += (
        f"Median projected move over the next {horizon} days is modest (${p50:.2f} expected). "
        f"Thus, maintaining your current position is the most statistically grounded choice."
    )

    return explanation


def explain_sell(
    entry_price: Optional[float],
    current_price: float,
    horizon: int,
    confidence: float,
    pat_strength: float,
    sentiment_label: str,
    forecast: Dict[str, float],
    sell_analysis: Optional[Dict[str, Any]],
) -> str:
    """
    Human-like SELL explanation.
    """

    p10 = forecast["p10"]
    p50 = forecast["p50"]
    p90 = forecast["p90"]

    if confidence < 0.40:
        explanation = (
            "The model currently leans SELL, but the edge is weak and should be treated as a cautious de-risking signal. "
        )
    else:
        explanation = (
            "The model recommends SELL based on weakening trend conditions and reduced upside probability. "
        )

    if pat_strength > 0.10:
        explanation += (
            "Some bullish pattern support is still present, so the technical picture is mixed rather than strongly bearish. "
        )
    elif pat_strength < -0.10:
        explanation += (
            "Bearish pattern pressure is also present, which supports the defensive stance. "
        )
    else:
        explanation += (
            "Pattern structure is mixed, so the sell case is not being driven by a single dominant bearish formation. "
        )

    # Sentiment
    if sentiment_label == "bearish":
        explanation += "Market sentiment is also bearish, increasing downside risk. "
    else:
        explanation += "Sentiment does not strongly confirm the sell case. "

    # If user gave entry price — compute actual saved profits or risks
    if entry_price is not None and sell_analysis is not None:
        curr_ret = sell_analysis["current_return"]
        exp_mid = sell_analysis["expected_future_ret"]
        exp_down = sell_analysis["potential_downside_if_hold"]

        if curr_ret > 0:
            explanation += (
                f"If sold now, you would lock in a return of {pct(curr_ret)}. "
            )
        elif curr_ret < 0:
            explanation += (
                f"You are currently at a loss of {pct(curr_ret)}, but the model anticipates further downside. "
            )

        explanation += (
            f"If you continue holding, the median expected performance is {pct(exp_mid)}, "
            f"with a downside risk reaching {pct(exp_down)} over the next {horizon} days. "
        )

    else:
        explanation += (
            f"Expected median future price over {horizon} days is only ${p50:.2f}, "
            f"while worst-case projections drop toward ${p10:.2f}. "
        )

    explanation += "Reducing exposure is statistically favorable in this setup."

    return explanation


# ===================================================================
# MAIN AGENT DECISION FUNCTION
# ===================================================================

def get_agentic_recommendation(
    budget: float,
    risk_profile: Literal["low", "medium", "high"],
    entry_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Full agent decision pipeline:
    1) Load data + model
    2) Evaluate latest row
    3) Pattern strength
    4) Sentiment integration
    5) Trend assessment
    6) Dynamic horizon (1..40 days)
    7) Monte Carlo forecasting
    8) Action-specific explanation
    9) PnL for SELL if entry_price provided
    """
    # -------------------------------------------------------
    # Load model & data
    # -------------------------------------------------------
    df, clf, feature_cols, label_map = load_model_and_data()

    latest = df.iloc[-1]
    price = float(latest["Close"])
    date = latest["Date"]

    # Prepare features in correct order
    X_latest = latest[feature_cols].values.reshape(1, -1)

    # Model probabilities
    probas = clf.predict_proba(X_latest)[0]
    classes = clf.classes_

    best_idx = int(np.argmax(probas))
    best_class = int(classes[best_idx])   # -1, 0, 1
    best_label = label_map.get((best_class), str(best_class))
    best_conf = float(probas[best_idx])
    probas_by_label = {
        label_map.get(int(c), str(int(c))): float(p)
        for c, p in zip(classes, probas)
    }

    # -------------------------------------------------------
    # PATTERNS
    # -------------------------------------------------------
    pattern_cols = [c for c in df.columns if c.startswith("pattern_")]
    active_patterns = [
        c.replace("pattern_", "")
        for c in pattern_cols
        if int(latest[c]) == 1
    ]
    pat_strength = compute_pattern_strength(active_patterns)

    # -------------------------------------------------------
    # SENTIMENT
    # -------------------------------------------------------
    sentiment_snapshot = SentimentSnapshotRepository().get_latest_snapshot("NVDA")
    sent_str = sentiment_strength(sentiment_snapshot.label, sentiment_snapshot.score)

    # -------------------------------------------------------
    # TREND
    # -------------------------------------------------------
    trend_str = compute_trend_strength(latest)

    adjusted_label, policy_note = apply_risk_decision_policy(
        probas_by_label=probas_by_label,
        risk_profile=risk_profile,
        pattern_bias="bullish" if pat_strength > 0.1 else "bearish" if pat_strength < -0.1 else "mixed" if abs(pat_strength) > 0 else "neutral",
        trend_strength=trend_str,
        sentiment_strength=sent_str,
    )
    reverse_label_map = {label: value for value, label in label_map.items()}
    best_label = adjusted_label
    best_class = reverse_label_map.get(best_label, 0)
    best_conf = float(probas_by_label.get(best_label, 0.0))

    # -------------------------------------------------------
    # DYNAMIC HORIZON
    # -------------------------------------------------------
    horizon = compute_dynamic_horizon(
        signal_value=best_class,
        model_confidence=best_conf,
        pattern_strength=pat_strength,
        sentiment_strength_value=sent_str,
        risk_profile=risk_profile,
    )

    # -------------------------------------------------------
    # MONTE CARLO FORECAST
    # -------------------------------------------------------
    # use last 1 year of log returns
    prices = df["Close"].tail(252).values
    returns = np.diff(np.log(prices))

    forecast = monte_carlo_forecast(
        last_price=price,
        log_returns=returns,
        days=horizon,
    )
    trend_forecast = logistic_trend_forecast(
        df["Close"].tail(252).values,
        last_date=pd.to_datetime(date).strftime("%Y-%m-%d"),
        horizon_days=min(horizon, 5),
    )

    # -------------------------------------------------------
    # POSITION SIZE
    # -------------------------------------------------------
    shares = 0
    if best_class == 1:   # BUY
        shares = compute_position_size(
            budget=budget,
            risk_profile=risk_profile,
            confidence=best_conf,
            price=price,
        )

    capital_used = shares * price

    # -------------------------------------------------------
    # SELL ANALYSIS (optional)
    # -------------------------------------------------------
    sell_info = None
    if best_class == -1 and entry_price is not None:
        sell_info = compute_sell_analysis(
            entry_price=entry_price,
            current_price=price,
            horizon_forecast=forecast,
        )

    # -------------------------------------------------------
    # ACTION EXPLANATION
    # -------------------------------------------------------
    if best_class == 1:
        explanation = explain_buy(
            horizon=horizon,
            confidence=best_conf,
            pat_strength=pat_strength,
            sent_label=sentiment_snapshot.label,
            sent_score=sentiment_snapshot.score,
            forecast=forecast,
        )
    elif best_class == 0:
        explanation = explain_hold(
            horizon=horizon,
            confidence=best_conf,
            pat_strength=pat_strength,
            sent_label=sentiment_snapshot.label,
            sent_score=sentiment_snapshot.score,
            forecast=forecast,
        )
    else:
        explanation = explain_sell(
            entry_price=entry_price,
            current_price=price,
            horizon=horizon,
            confidence=best_conf,
            pat_strength=pat_strength,
            sentiment_label=sentiment_snapshot.label,
            forecast=forecast,
            sell_analysis=sell_info,
        )
    if policy_note:
        explanation = f"{policy_note} {explanation}"

    # -------------------------------------------------------
    # FINAL OUTPUT
    # -------------------------------------------------------
    formatted_date = pd.to_datetime(date).strftime("%Y-%m-%d")

    out = {
        "date": formatted_date,
        "latest_close": price,
        "action": best_label,
        "signal_value": best_class,
        "confidence": best_conf,
       "probas": probas_by_label,
        "patterns": active_patterns,
        "pattern_strength": pat_strength,

        "trend_strength": trend_str,

        "sentiment_label": sentiment_snapshot.label,
        "sentiment_score": sentiment_snapshot.score,
        "sentiment_strength": sent_str,

        "horizon_days": horizon,
        "forecast": forecast,
        "trend_forecast": trend_forecast,

        "suggested_shares": shares,
        "capital_used": capital_used,

        "sell_analysis": sell_info,

        "entry_price_used": entry_price,
        "risk_profile": risk_profile,
        "budget": budget,

        "explanation": explanation,
    }

    return out

