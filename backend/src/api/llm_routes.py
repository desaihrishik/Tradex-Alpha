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

    answer = ask_trade_question(
        question=data.question,
        context={
            "symbol": data.symbol,
            "budget": data.budget,
            "risk": data.risk,
            "entry_price": data.entry_price,
            "current_trend": current_trend,
            "latest_signal": analysis.get("latest_signal"),
            "agentic_signal": analysis.get("agentic_signal"),
            "decision": analysis.get("decision"),
            "validation": analysis.get("validation"),
            "quant_insights": analysis.get("quant_insights"),
            "sentiment": sentiment,
        },
    )

    return {
        "answer": answer,
        "question": data.question,
    }
