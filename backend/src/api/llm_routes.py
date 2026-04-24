from fastapi import APIRouter
from pydantic import BaseModel

from src.llm_client import ask_llm, ask_trade_question
from src.services.runtime_services import recommendation_service

router = APIRouter(prefix="/api/llm")


class ObserveRequest(BaseModel):
    setup: str


@router.post("/observe")
def observe_llm(data: ObserveRequest):
    analysis = ask_llm(
        f"You are a trading assistant. Analyze this setup:\n{data.setup}"
    )
    return {"analysis": analysis}


class TradeQuestionRequest(BaseModel):
    question: str
    symbol: str = "NVDA"
    budget: float = 1000.0
    risk: str = "medium"
    entry_price: float | None = None
    mode: str = "fast"


def _compact_trade_context(analysis: dict, current_trend: dict, sentiment: dict) -> dict:
    latest_signal = analysis.get("latest_signal") or {}
    agentic_signal = analysis.get("agentic_signal") or {}
    quant_insights = analysis.get("quant_insights") or {}

    return {
        "current_trend": {
            "as_of": current_trend.get("as_of"),
            "latest_close": current_trend.get("latest_close"),
            "change": current_trend.get("change"),
            "change_pct": current_trend.get("change_pct"),
            "trend_label": current_trend.get("trend_label"),
            "patterns": current_trend.get("patterns"),
            "volume": current_trend.get("volume"),
        },
        "latest_signal": {
            "date": latest_signal.get("date"),
            "action": latest_signal.get("action"),
            "confidence": latest_signal.get("confidence"),
            "patterns": latest_signal.get("patterns"),
            "explanation": latest_signal.get("explanation"),
        },
        "agentic_signal": {
            "action": agentic_signal.get("action"),
            "confidence": agentic_signal.get("confidence"),
            "horizon_days": agentic_signal.get("horizon_days"),
            "forecast": agentic_signal.get("forecast"),
            "sentiment_label": agentic_signal.get("sentiment_label"),
            "sentiment_score": agentic_signal.get("sentiment_score"),
            "explanation": agentic_signal.get("explanation"),
        },
        "decision": analysis.get("decision"),
        "validation": analysis.get("validation"),
        "quant_summary": {
            "as_of": quant_insights.get("as_of"),
            "market_regime": quant_insights.get("market_regime"),
            "momentum": (quant_insights.get("momentum") or {}).get("score"),
            "risk_state": (quant_insights.get("summary") or {}).get("risk_state"),
            "momentum_state": (quant_insights.get("summary") or {}).get("momentum_state"),
        },
        "sentiment": {
            "as_of": sentiment.get("as_of"),
            "label": sentiment.get("label"),
            "score": sentiment.get("score"),
            "interpretation": sentiment.get("interpretation"),
        },
    }


@router.post("/trade_question")
def trade_question(data: TradeQuestionRequest):
    analysis = recommendation_service.run_analysis(
        symbol=data.symbol,
        budget=data.budget,
        risk=data.risk,
        limit=120,
        entry_price=data.entry_price,
        persist=False,
    )
    current_trend = recommendation_service.get_current_trend(symbol=data.symbol, refresh=False)
    sentiment = recommendation_service.get_sentiment_details(symbol=data.symbol, refresh=False)

    compact_context = _compact_trade_context(analysis, current_trend, sentiment)
    full_context = {
        "symbol": data.symbol,
        "budget": data.budget,
        "risk": data.risk,
        "entry_price": data.entry_price,
        "analyze_response": analysis,
        "current_trend": current_trend,
        "sentiment": sentiment,
    }
    mode = (data.mode or "fast").strip().lower()
    context_payload = full_context if mode == "deep" else {
        "symbol": data.symbol,
        "budget": data.budget,
        "risk": data.risk,
        "entry_price": data.entry_price,
        **compact_context,
    }
    answer = ask_trade_question(
        question=data.question,
        context=context_payload,
        mode=mode,
    )

    return {
        "answer": answer,
        "question": data.question,
    }
