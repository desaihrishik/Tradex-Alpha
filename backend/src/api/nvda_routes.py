from fastapi import APIRouter

from src.services.recommendation_service import RecommendationService

router = APIRouter()
recommendation_service = RecommendationService()


@router.get("/api/ping")
def ping():
    return {"status": "ok"}


@router.get("/api/nvda/latest_signal")
def latest_signal(budget: float = 1000.0, risk: str = "medium"):
    return recommendation_service.get_latest_signal(budget=budget, risk=risk)


@router.get("/api/nvda/candles")
def get_candles(limit: int = 120):
    return recommendation_service.get_candles(limit=limit)


@router.get("/api/nvda/agentic_signal")
def agentic_signal(
    budget: float = 1000.0,
    risk: str = "medium",
    entry_price: float | None = None,
):
    return recommendation_service.get_agentic_signal(
        budget=budget,
        risk=risk,
        entry_price=entry_price,
    )


@router.get("/api/nvda/analyze")
def analyze_nvda(
    budget: float = 1000.0,
    risk: str = "medium",
    limit: int = 120,
    entry_price: float | None = None,
):
    return recommendation_service.run_analysis(
        symbol="NVDA",
        budget=budget,
        risk=risk,
        limit=limit,
        entry_price=entry_price,
    )
