from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.supabase_client import get_supabase_admin_client
from src.data.repositories.market_data import LocalMarketDataRepository


@dataclass(frozen=True)
class PersistedAnalysisRefs:
    forecast_run_id: str | None
    recommendation_id: str | None


class AnalysisResultsRepository:
    def __init__(self) -> None:
        self.market_data = LocalMarketDataRepository()

    def persist_analysis(
        self,
        *,
        symbol: str,
        budget: float,
        risk_profile: str,
        latest_signal: dict[str, Any],
        agentic_signal: dict[str, Any],
        decision: dict[str, Any],
        explanation: str,
    ) -> PersistedAnalysisRefs:
        client = get_supabase_admin_client()
        if client is None:
            return PersistedAnalysisRefs(forecast_run_id=None, recommendation_id=None)

        symbol_id = self.market_data.ensure_symbol(symbol)
        if symbol_id is None:
            return PersistedAnalysisRefs(forecast_run_id=None, recommendation_id=None)

        forecast_run_id = None
        forecast = agentic_signal.get("forecast")
        if isinstance(forecast, dict):
            forecast_payload = {
                "symbol_id": symbol_id,
                "model_id": None,
                "horizon_days": int(agentic_signal.get("horizon_days") or 0),
                "p10": forecast.get("p10"),
                "p50": forecast.get("p50"),
                "p90": forecast.get("p90"),
                "raw": {
                    "agentic_signal": agentic_signal,
                    "latest_signal": latest_signal,
                },
            }
            forecast_result = client.table("forecast_runs").insert(forecast_payload).execute()
            if forecast_result.data:
                forecast_run_id = forecast_result.data[0]["id"]

        recommendation_payload = {
            "symbol_id": symbol_id,
            "model_id": None,
            "forecast_run_id": forecast_run_id,
            "risk_profile": risk_profile,
            "budget": budget,
            "action": decision.get("action") or agentic_signal.get("action") or latest_signal.get("action") or "HOLD",
            "confidence": decision.get("confidence") or agentic_signal.get("confidence") or latest_signal.get("confidence") or 0,
            "suggested_amount": agentic_signal.get("capital_used") or latest_signal.get("capital_used"),
            "suggested_shares": agentic_signal.get("suggested_shares") or latest_signal.get("suggested_shares"),
            "suggested_duration_days": agentic_signal.get("horizon_days"),
            "current_trend": agentic_signal.get("sentiment_label"),
            "historical_trend": None,
            "sentiment_label": agentic_signal.get("sentiment_label"),
            "sentiment_score": agentic_signal.get("sentiment_score"),
            "explanation": explanation,
            "payload": {
                "decision": decision,
                "latest_signal": latest_signal,
                "agentic_signal": agentic_signal,
            },
        }
        recommendation_result = client.table("recommendations").insert(recommendation_payload).execute()
        recommendation_id = None
        if recommendation_result.data:
            recommendation_id = recommendation_result.data[0]["id"]

        return PersistedAnalysisRefs(
            forecast_run_id=forecast_run_id,
            recommendation_id=recommendation_id,
        )
