from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.data.repositories.market_data import LocalMarketDataRepository

RiskProfile = Literal["low", "medium", "high"]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _stoch_rsi(rsi_series: pd.Series, period: int = 14) -> pd.Series:
    rsi_min = rsi_series.rolling(period).min()
    rsi_max = rsi_series.rolling(period).max()
    return ((rsi_series - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)).fillna(0.5)


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0.0
    minus_dm[minus_dm < plus_dm] = 0.0

    tr_components = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().fillna(20)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr_components = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    return tr.rolling(period).mean().bfill().fillna(0.0)


def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    raw_flow = typical * df["Volume"]
    direction = typical.diff()
    positive = raw_flow.where(direction > 0, 0.0).rolling(period).sum()
    negative = raw_flow.where(direction < 0, 0.0).rolling(period).sum().abs()
    ratio = positive / negative.replace(0, np.nan)
    return (100 - (100 / (1 + ratio))).fillna(50.0)


def _regime_label(*, adx: float, hist_vol: float, close: float, ema50: float, ema200: float) -> str:
    high_vol = hist_vol >= 0.55
    low_vol = hist_vol <= 0.25
    bullish = close > ema50 > ema200
    bearish = close < ema50 < ema200
    if high_vol and adx >= 30:
        return "High Volatility Panic"
    if bullish and adx >= 25:
        return "Bullish Trend"
    if bearish and adx >= 25:
        return "Bearish"
    if adx >= 22 and not high_vol:
        return "Momentum Breakout"
    if low_vol and adx < 18:
        return "Sideways"
    return "Mean Reversion Zone"


def _risk_pct_from_profile(risk_profile: RiskProfile) -> float:
    if risk_profile == "low":
        return 0.01
    if risk_profile == "high":
        return 0.025
    return 0.015


def _compute_relative_strength(
    repo: LocalMarketDataRepository,
    symbol_df: pd.DataFrame,
    *,
    benchmarks: tuple[str, ...] = ("QQQ", "SOXX", "SPY"),
) -> dict[str, Any]:
    symbol_close = symbol_df[["Date", "Close"]].copy()
    symbol_close["Date"] = pd.to_datetime(symbol_close["Date"], utc=True).dt.tz_convert(None)
    symbol_close = symbol_close.sort_values("Date").tail(260)

    result: dict[str, Any] = {"available": False, "benchmarks": []}
    for benchmark in benchmarks:
        try:
            benchmark_raw = repo._fetch_live_daily_history(benchmark, days=420)  # noqa: SLF001
            if benchmark_raw.empty:
                continue
            bench_close = benchmark_raw[["Date", "Close"]].copy()
            bench_close["Date"] = pd.to_datetime(bench_close["Date"], utc=True).dt.tz_convert(None)
            aligned = symbol_close.merge(bench_close, on="Date", suffixes=("_symbol", "_bench")).dropna()
            if len(aligned) < 80:
                continue
            symbol_ret_20 = _to_float(aligned["Close_symbol"].pct_change(20).iloc[-1])
            bench_ret_20 = _to_float(aligned["Close_bench"].pct_change(20).iloc[-1])
            rel_alpha = symbol_ret_20 - bench_ret_20
            score = int(round(_clamp(50 + rel_alpha * 250, 0, 100)))
            result["benchmarks"].append(
                {
                    "ticker": benchmark,
                    "symbol_return_20d": symbol_ret_20,
                    "benchmark_return_20d": bench_ret_20,
                    "relative_alpha_20d": rel_alpha,
                    "outperforming": rel_alpha > 0,
                    "score": score,
                }
            )
        except Exception:
            continue

    result["available"] = len(result["benchmarks"]) > 0
    if result["available"]:
        avg_score = np.mean([item["score"] for item in result["benchmarks"]])
        result["composite_score"] = float(round(avg_score, 2))
    else:
        result["composite_score"] = None
    return result


def compute_quant_insights(
    *,
    market_data: LocalMarketDataRepository,
    symbol: str,
    budget: float,
    risk_profile: RiskProfile,
    model_confidence: float,
    entry_price: float | None = None,
) -> dict[str, Any]:
    raw = market_data.get_raw_market_history(ticker=symbol, limit=520).copy()
    if raw.empty:
        return {"available": False}

    df = raw.sort_values("Date").reset_index(drop=True)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)
    if len(df) < 120:
        return {"available": False}

    close = df["Close"]
    volume = df["Volume"]
    returns = close.pct_change().fillna(0)
    log_returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma100 = close.rolling(100).mean()

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    rsi14 = _rsi(close, 14)
    stoch_rsi14 = _stoch_rsi(rsi14, 14)
    adx14 = _adx(df, 14)
    roc10 = close.pct_change(10)
    atr14 = _atr(df, 14)
    bb_mid = sma20
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    hist_vol20 = log_returns.rolling(20).std() * np.sqrt(252)
    realized_vol20 = np.sqrt((returns.pow(2)).rolling(20).sum() * (252 / 20))

    vwap20 = (close * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    rvol20 = volume / volume.rolling(20).mean().replace(0, np.nan)
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    ad_line = (((close - df["Low"]) - (df["High"] - close)) / (df["High"] - df["Low"]).replace(0, np.nan)).fillna(0) * volume
    ad_line = ad_line.cumsum()
    mfi14 = _mfi(df, 14)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    latest_close = _to_float(latest["Close"])
    latest_volume = _to_float(latest["Volume"])
    latest_return = _to_float(returns.iloc[-1])
    latest_rsi = _to_float(rsi14.iloc[-1], 50.0)
    latest_macd_hist = _to_float(macd_hist.iloc[-1])
    latest_adx = _to_float(adx14.iloc[-1], 20.0)
    latest_roc = _to_float(roc10.iloc[-1])
    latest_rvol = _to_float(rvol20.iloc[-1], 1.0)
    latest_atr = _to_float(atr14.iloc[-1])
    latest_atr_pct = latest_atr / latest_close if latest_close else 0.0
    latest_hist_vol = _to_float(hist_vol20.iloc[-1], 0.25)
    latest_realized_vol = _to_float(realized_vol20.iloc[-1], 0.25)

    volume_strength = _clamp(latest_rvol / 1.8, 0.0, 1.0)
    macd_strength = _clamp(abs(latest_macd_hist) / max(latest_close * 0.01, 1e-6), 0.0, 1.0)
    rsi_component = _clamp(latest_rsi / 100.0, 0.0, 1.0)
    adx_component = _clamp(latest_adx / 50.0, 0.0, 1.0)
    roc_component = _clamp((latest_roc + 0.2) / 0.4, 0.0, 1.0)
    composite_momentum = (
        0.3 * rsi_component
        + 0.25 * macd_strength
        + 0.2 * adx_component
        + 0.15 * roc_component
        + 0.1 * volume_strength
    )
    momentum_score = float(round(composite_momentum * 100, 1))

    regime = _regime_label(
        adx=latest_adx,
        hist_vol=latest_hist_vol,
        close=latest_close,
        ema50=_to_float(ema50.iloc[-1], latest_close),
        ema200=_to_float(ema200.iloc[-1], latest_close),
    )
    vol_regime = "HIGH VOL" if latest_hist_vol >= 0.55 else "LOW VOL" if latest_hist_vol <= 0.25 else "NORMAL VOL"

    lookback_returns = returns.tail(252)
    rf_daily = 0.04 / 252.0
    ret_std = float(lookback_returns.std())
    downside_std = float(lookback_returns[lookback_returns < 0].std()) if (lookback_returns < 0).any() else 0.0
    sharpe = ((lookback_returns.mean() - rf_daily) / ret_std) * np.sqrt(252) if ret_std > 0 else 0.0
    sortino = ((lookback_returns.mean() - rf_daily) / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0
    equity = (1 + lookback_returns).cumprod()
    running_peak = equity.cummax()
    drawdown = (equity / running_peak) - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    wins = lookback_returns[lookback_returns > 0]
    losses = lookback_returns[lookback_returns < 0]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    win_rate = float((lookback_returns > 0).mean()) if not lookback_returns.empty else 0.0
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    expected_return = win_rate * avg_win + (1 - win_rate) * avg_loss
    expected_loss = abs((1 - win_rate) * avg_loss)
    kelly = win_rate - ((1 - win_rate) / risk_reward) if risk_reward > 0 else 0.0
    var95 = float(np.quantile(lookback_returns, 0.05)) if not lookback_returns.empty else 0.0
    cvar95 = float(lookback_returns[lookback_returns <= var95].mean()) if (lookback_returns <= var95).any() else var95

    base_risk_pct = _risk_pct_from_profile(risk_profile)
    confidence_scale = _clamp(0.55 + _clamp(model_confidence, 0.0, 1.0) * 0.75, 0.4, 1.3)
    effective_risk_pct = _clamp(base_risk_pct * confidence_scale, 0.005, 0.035)
    stop_distance = max(latest_atr * 1.2, latest_close * 0.012)
    stop_loss = latest_close - stop_distance
    target = latest_close + stop_distance * 2.2
    trailing_stop_pct = _clamp(latest_atr_pct * 1.4, 0.008, 0.05)
    shares = int(max(0, (budget * effective_risk_pct) // max(stop_distance, 1e-6)))
    capital_at_risk = shares * stop_distance

    previous_high = _to_float(df["High"].iloc[-2], latest_close)
    previous_low = _to_float(df["Low"].iloc[-2], latest_close)
    previous_close = _to_float(df["Close"].iloc[-2], latest_close)
    pivot = (previous_high + previous_low + previous_close) / 3.0
    support1 = (2 * pivot) - previous_high
    resistance1 = (2 * pivot) - previous_low
    prev_high_20 = _to_float(df["High"].rolling(20).max().shift(1).iloc[-1], latest_close)
    prev_low_20 = _to_float(df["Low"].rolling(20).min().shift(1).iloc[-1], latest_close)
    breakout = "bullish_breakout" if latest_close > prev_high_20 * 1.002 else "bearish_breakdown" if latest_close < prev_low_20 * 0.998 else "inside_range"
    gap_pct = (float(latest["Open"]) - previous_close) / previous_close if previous_close else 0.0
    nearest_resistance = min([level for level in [resistance1, prev_high_20] if level >= latest_close], default=resistance1)
    nearest_support = max([level for level in [support1, prev_low_20] if level <= latest_close], default=support1)

    horizon_days = 12
    mean_horizon = float(lookback_returns.tail(120).mean()) * horizon_days
    std_horizon = float(lookback_returns.tail(120).std()) * np.sqrt(horizon_days)
    if std_horizon > 0:
        sim = np.random.normal(mean_horizon, std_horizon, size=3000)
        end_prices = latest_close * (1 + sim)
        prob_target = float((end_prices >= target).mean())
        prob_stop = float((end_prices <= stop_loss).mean())
    else:
        prob_target = 0.5
        prob_stop = 0.5

    relative_strength = _compute_relative_strength(market_data, df, benchmarks=("QQQ", "SOXX", "SPY"))

    ema_alignment = "bullish" if _to_float(ema9.iloc[-1]) > _to_float(ema21.iloc[-1]) > _to_float(ema50.iloc[-1]) > _to_float(ema200.iloc[-1]) else "bearish" if _to_float(ema9.iloc[-1]) < _to_float(ema21.iloc[-1]) < _to_float(ema50.iloc[-1]) < _to_float(ema200.iloc[-1]) else "mixed"
    sma_alignment = "bullish" if _to_float(sma20.iloc[-1]) > _to_float(sma50.iloc[-1]) > _to_float(sma100.iloc[-1]) else "bearish" if _to_float(sma20.iloc[-1]) < _to_float(sma50.iloc[-1]) < _to_float(sma100.iloc[-1]) else "mixed"

    return {
        "available": True,
        "as_of": pd.to_datetime(df.iloc[-1]["Date"]).strftime("%Y-%m-%d"),
        "market_regime": {
            "label": regime,
            "volatility_regime": vol_regime,
        },
        "momentum": {
            "score": momentum_score,
            "ema_alignment": ema_alignment,
            "sma_alignment": sma_alignment,
            "rsi_14": latest_rsi,
            "stoch_rsi": float(round(_to_float(stoch_rsi14.iloc[-1]) * 100, 2)),
            "macd_hist": float(round(latest_macd_hist, 4)),
            "adx_14": float(round(latest_adx, 2)),
            "roc_10": float(round(latest_roc, 4)),
        },
        "volatility": {
            "atr_14": float(round(latest_atr, 3)),
            "atr_pct": float(round(latest_atr_pct, 4)),
            "bollinger": {
                "upper": float(round(_to_float(bb_upper.iloc[-1]), 2)),
                "mid": float(round(_to_float(bb_mid.iloc[-1]), 2)),
                "lower": float(round(_to_float(bb_lower.iloc[-1]), 2)),
            },
            "historical_vol_20": float(round(latest_hist_vol, 4)),
            "realized_vol_20": float(round(latest_realized_vol, 4)),
            "implied_vol": None,
            "regime_hint": "Use tighter stops and faster exits" if vol_regime == "HIGH VOL" else "Allow wider swing hold windows" if vol_regime == "LOW VOL" else "Balanced stop and hold profile",
        },
        "volume_intelligence": {
            "rvol_20": float(round(latest_rvol, 2)),
            "volume_spike": bool(latest_rvol >= 1.7),
            "obv_trend": "up" if _to_float(obv.diff().tail(5).mean()) > 0 else "down",
            "vwap_20": float(round(_to_float(vwap20.iloc[-1], latest_close), 2)),
            "money_flow_index": float(round(_to_float(mfi14.iloc[-1], 50.0), 2)),
            "acc_dist_slope_10": float(round(_to_float(ad_line.diff().tail(10).mean()), 2)),
            "volume_last": float(round(latest_volume, 0)),
        },
        "risk_metrics": {
            "sharpe": float(round(sharpe, 3)),
            "sortino": float(round(sortino, 3)),
            "max_drawdown": float(round(max_drawdown, 4)),
            "win_rate": float(round(win_rate, 4)),
            "risk_reward_ratio": float(round(risk_reward, 3)),
            "expected_return": float(round(expected_return, 4)),
            "expected_loss": float(round(expected_loss, 4)),
            "kelly_fraction": float(round(kelly, 4)),
            "var_95": float(round(var95, 4)),
            "cvar_95": float(round(cvar95, 4)),
        },
        "position_sizing": {
            "risk_pct": float(round(effective_risk_pct, 4)),
            "suggested_shares": int(shares),
            "capital_at_risk": float(round(capital_at_risk, 2)),
            "stop_loss": float(round(stop_loss, 2)),
            "trailing_stop_pct": float(round(trailing_stop_pct, 4)),
            "profit_target": float(round(target, 2)),
            "stop_loss_distance": float(round(stop_distance, 3)),
            "confidence_scaling": float(round(confidence_scale, 3)),
            "entry_price_used": _to_float(entry_price, latest_close) if entry_price is not None else None,
            "probability_target_hit_12d": float(round(prob_target, 4)),
            "probability_stop_hit_12d": float(round(prob_stop, 4)),
        },
        "support_resistance": {
            "pivot": float(round(pivot, 2)),
            "support_1": float(round(support1, 2)),
            "resistance_1": float(round(resistance1, 2)),
            "previous_high_20": float(round(prev_high_20, 2)),
            "previous_low_20": float(round(prev_low_20, 2)),
            "nearest_support": float(round(nearest_support, 2)),
            "nearest_resistance": float(round(nearest_resistance, 2)),
            "breakout_signal": breakout,
            "gap_pct": float(round(gap_pct, 4)),
        },
        "relative_strength": relative_strength,
        "fundamental_overlay": {
            "earnings_risk": "unavailable_in_current_feed",
            "analyst_target": None,
            "institutional_holdings_trend": None,
            "insider_flow": None,
            "revenue_growth": None,
            "eps_surprise": None,
        },
        "summary": {
            "momentum_state": "strong" if momentum_score >= 65 else "neutral" if momentum_score >= 45 else "weak",
            "risk_state": "contained" if max_drawdown > -0.2 and abs(var95) < 0.03 else "elevated",
            "last_daily_move": float(round(latest_return, 4)),
        },
    }
