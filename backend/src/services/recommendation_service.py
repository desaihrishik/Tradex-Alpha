from __future__ import annotations

from src.agent_engine import get_agentic_recommendation
from src.data.repositories.analysis_results import AnalysisResultsRepository
from src.data.repositories.market_data import LocalMarketDataRepository
from src.data.repositories.sentiment_repository import SentimentSnapshotRepository
from src.orchestration.graph import run_analysis_pipeline
from src.quant_engine import compute_quant_insights
from src.signal_engine import compute_position_size, get_latest_recommendation
import pandas as pd


class RecommendationService:
    def __init__(self) -> None:
        self.market_data = LocalMarketDataRepository()
        self.sentiment_repo = SentimentSnapshotRepository()
        self.analysis_results = AnalysisResultsRepository()

    def _refresh_live_inputs(self, symbol: str = "NVDA") -> None:
        self.market_data.sync_recent_market_data(ticker=symbol)
        self.sentiment_repo.sync_sentiment_snapshot(symbol=symbol)

    def get_candles(self, limit: int = 120) -> dict[str, list[dict[str, object]]]:
        try:
            self._refresh_live_inputs("NVDA")
        except Exception:
            pass
        candles = self.market_data.get_candles(limit=limit, ticker="NVDA")
        return {
            "candles": [
                {
                    "date": candle.date,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "patterns": candle.patterns,
                    "signal": candle.signal,
                }
                for candle in candles
            ]
        }

    def get_current_trend(self, symbol: str = "NVDA", *, refresh: bool = True) -> dict[str, object]:
        if refresh:
            try:
                self._refresh_live_inputs(symbol)
            except Exception:
                pass
        snapshot = self.market_data.get_latest_market_snapshot(ticker=symbol, refresh=False)
        sentiment = self.sentiment_repo.get_latest_snapshot(symbol=symbol)
        if snapshot:
            snapshot["sentiment"] = {
                "label": sentiment.label,
                "score": sentiment.score,
                "article_count": sentiment.article_count,
                "ts": sentiment.ts,
            }
            return snapshot

        candles = self.market_data.get_candles(limit=2, ticker=symbol)
        latest = candles[-1]
        previous = candles[-2] if len(candles) > 1 else latest
        change = latest.close - previous.close
        change_pct = (change / previous.close) if previous.close else 0.0
        trend_label = "sideways"
        if change_pct > 0.01:
            trend_label = "bullish"
        elif change_pct < -0.01:
            trend_label = "bearish"
        return {
            "symbol": symbol,
            "as_of": latest.date,
            "latest_close": latest.close,
            "previous_close": previous.close,
            "change": change,
            "change_pct": change_pct,
            "trend_label": trend_label,
            "trend_strength": latest.signal,
            "patterns": latest.patterns,
            "volume": latest.volume,
            "recommendation": None,
            "sentiment": {
                "label": sentiment.label,
                "score": sentiment.score,
                "article_count": sentiment.article_count,
                "ts": sentiment.ts,
            },
        }

    def get_sentiment_details(self, symbol: str = "NVDA", *, refresh: bool = False) -> dict[str, object]:
        if refresh:
            try:
                self.sentiment_repo.sync_sentiment_snapshot(symbol=symbol)
            except Exception:
                pass

        snapshot = self.sentiment_repo.get_latest_snapshot(symbol=symbol, refresh_if_stale=refresh)
        recent_snapshots = self.sentiment_repo.get_recent_snapshots(symbol=symbol, limit=2)
        previous_snapshot = recent_snapshots[1] if len(recent_snapshots) > 1 else None
        usefulness = [
            "Adjusts the recommendation confidence when news tone supports or contradicts the technical setup.",
            "Feeds the agentic horizon/risk reasoning so mixed news can shorten conviction and strong news can reinforce it.",
            "Helps explain whether the model is leaning with or against current headline tone.",
        ]

        score = float(snapshot.score)
        if score >= 0.6:
            interpretation = "Headline tone is strongly supportive."
        elif score >= 0.2:
            interpretation = "Headline tone is mildly supportive."
        elif score <= -0.6:
            interpretation = "Headline tone is strongly negative."
        elif score <= -0.2:
            interpretation = "Headline tone is mildly negative."
        else:
            interpretation = "Headline tone is broadly neutral or mixed."
        previous_score = float(previous_snapshot.score) if previous_snapshot else None
        score_change = score - previous_score if previous_score is not None else None
        score_change_pct = (
            (score_change / abs(previous_score))
            if previous_score not in (None, 0.0) and score_change is not None
            else None
        )

        return {
            "symbol": symbol,
            "as_of": snapshot.ts,
            "label": snapshot.label,
            "score": score,
            "article_count": snapshot.article_count,
            "interpretation": interpretation,
            "previous_score": previous_score,
            "previous_label": previous_snapshot.label if previous_snapshot else None,
            "previous_as_of": previous_snapshot.ts if previous_snapshot else None,
            "score_change": score_change,
            "score_change_pct": score_change_pct,
            "usefulness": usefulness,
        }

    def get_latest_signal(self, budget: float, risk: str) -> dict[str, object]:
        self._refresh_live_inputs("NVDA")
        return get_latest_recommendation(budget=budget, risk_profile=risk)  # type: ignore[arg-type]

    def validate_recommendation(
        self,
        *,
        latest_signal: dict[str, object],
        budget: float,
        risk: str,
        symbol: str = "NVDA",
    ) -> dict[str, object]:
        current_snapshot = self.get_current_trend(symbol=symbol, refresh=False)

        model_date = str(latest_signal.get("date") or "")
        market_date = str(current_snapshot.get("as_of") or "")
        model_price = float(latest_signal.get("latest_close") or 0.0)
        market_price = float(current_snapshot.get("latest_close") or 0.0)
        confidence = float(latest_signal.get("confidence") or 0.0)
        action = str(latest_signal.get("action") or "HOLD")

        date_gap_days = None
        if model_date and market_date:
            date_gap_days = int(
                abs(
                    (pd.to_datetime(market_date) - pd.to_datetime(model_date)).days
                )
            )

        price_diff = market_price - model_price
        price_diff_pct = (price_diff / model_price) if model_price else 0.0

        suggested_shares_at_market = 0
        capital_used_at_market = 0.0
        if action == "BUY" and market_price > 0:
            suggested_shares_at_market = compute_position_size(
                budget=budget,
                risk_profile=risk,  # type: ignore[arg-type]
                confidence=confidence,
                price=market_price,
            )
            capital_used_at_market = suggested_shares_at_market * market_price

        checks = []
        is_fresh = True
        if date_gap_days is not None and date_gap_days > 0:
            is_fresh = False
            checks.append(
                f"Model snapshot date ({model_date}) is {date_gap_days} day(s) behind latest market date ({market_date})."
            )
        if abs(price_diff_pct) > 0.01:
            is_fresh = False
            checks.append(
                f"Model price ${model_price:.2f} differs from latest market price ${market_price:.2f} by {(price_diff_pct * 100):.2f}%."
            )
        if int(latest_signal.get("suggested_shares") or 0) != suggested_shares_at_market:
            checks.append(
                f"Suggested shares at model price: {int(latest_signal.get('suggested_shares') or 0)}; at current market price: {suggested_shares_at_market}."
            )

        return {
            "is_fresh": is_fresh,
            "model_date": model_date,
            "market_date": market_date,
            "date_gap_days": date_gap_days,
            "model_price": model_price,
            "market_price": market_price,
            "price_diff": price_diff,
            "price_diff_pct": price_diff_pct,
            "displayed_suggested_shares": int(latest_signal.get("suggested_shares") or 0),
            "suggested_shares_at_market_price": suggested_shares_at_market,
            "displayed_capital_used": float(latest_signal.get("capital_used") or 0.0),
            "capital_used_at_market_price": capital_used_at_market,
            "checks": checks,
        }

    def get_agentic_signal(
        self,
        budget: float,
        risk: str,
        entry_price: float | None = None,
    ) -> dict[str, object]:
        self._refresh_live_inputs("NVDA")
        return get_agentic_recommendation(
            budget=budget,
            risk_profile=risk,  # type: ignore[arg-type]
            entry_price=entry_price,
        )

    def run_analysis(
        self,
        symbol: str,
        budget: float,
        risk: str,
        limit: int = 120,
        entry_price: float | None = None,
        persist: bool = True,
    ) -> dict[str, object]:
        self._refresh_live_inputs(symbol)
        result = run_analysis_pipeline(
            symbol=symbol,
            budget=budget,
            risk_profile=risk,
            limit=limit,
            entry_price=entry_price,
            service=self,
        )
        result["validation"] = self.validate_recommendation(
            latest_signal=result.get("latest_signal", {}),
            budget=budget,
            risk=risk,
            symbol=symbol,
        )
        result["quant_insights"] = compute_quant_insights(
            market_data=self.market_data,
            symbol=symbol,
            budget=budget,
            risk_profile=risk,  # type: ignore[arg-type]
            model_confidence=float((result.get("agentic_signal") or {}).get("confidence") or (result.get("latest_signal") or {}).get("confidence") or 0.0),
            entry_price=entry_price,
        )
        persistence_payload = {
            "forecast_run_id": None,
            "recommendation_id": None,
        }
        if persist:
            try:
                persisted = self.analysis_results.persist_analysis(
                    symbol=symbol,
                    budget=budget,
                    risk_profile=risk,
                    latest_signal=result.get("latest_signal", {}),
                    agentic_signal=result.get("agentic_signal", {}),
                    decision=result.get("decision", {}),
                    explanation=str(result.get("explanation", "")),
                )
                persistence_payload = {
                    "forecast_run_id": persisted.forecast_run_id,
                    "recommendation_id": persisted.recommendation_id,
                }
            except Exception as exc:
                result["persistence_warning"] = f"persist_analysis_failed: {exc}"

        result["persistence"] = persistence_payload
        return result
