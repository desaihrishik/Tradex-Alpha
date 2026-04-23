from pydantic import BaseModel
from fastapi import APIRouter, Header, HTTPException
from hmac import compare_digest

from src.core.config import get_settings
from src.data.repositories.market_data import LocalMarketDataRepository
from src.data.repositories.sentiment_repository import SentimentSnapshotRepository
from src.services.runtime_services import refresh_service

router = APIRouter(prefix="/api/admin", tags=["admin"])
settings = get_settings()


class AdminLoginRequest(BaseModel):
    password: str


def require_admin(x_admin_password: str | None = Header(default=None)) -> None:
    if not compare_digest(str(x_admin_password or ""), settings.admin_panel_password):
        raise HTTPException(status_code=401, detail="Invalid admin credentials.")


@router.post("/login")
def admin_login(payload: AdminLoginRequest):
    if not compare_digest(payload.password, settings.admin_panel_password):
        raise HTTPException(status_code=401, detail="Invalid admin credentials.")
    return {"ok": True}


@router.post("/sync_market_data")
def sync_market_data(symbol: str = "NVDA", x_admin_password: str | None = Header(default=None)):
    try:
        require_admin(x_admin_password)
        repository = LocalMarketDataRepository()
        candle_result = repository.sync_dataset_to_supabase(ticker=symbol)
        technical_result = repository.sync_technicals_to_supabase(ticker=symbol)
        return {
            "candles": candle_result,
            "technicals": technical_result,
        }
    except Exception as exc:  # pragma: no cover - admin helper path
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/sync_sentiment")
def sync_sentiment(symbol: str = "NVDA", x_admin_password: str | None = Header(default=None)):
    try:
        require_admin(x_admin_password)
        repository = SentimentSnapshotRepository()
        snapshot = repository.sync_sentiment_snapshot(symbol=symbol)
        return {
            "symbol": snapshot.symbol,
            "ts": snapshot.ts,
            "label": snapshot.label,
            "score": snapshot.score,
            "article_count": snapshot.article_count,
        }
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/refresh_all")
def refresh_all(
    symbol: str = "NVDA",
    include_model: bool = False,
    x_admin_password: str | None = Header(default=None),
):
    try:
        require_admin(x_admin_password)
        status = refresh_service.refresh_all(symbol=symbol)
        if include_model:
            refresh_service.retrain_model(symbol=symbol)
            status = refresh_service.get_status()
        return {"symbol": symbol, "status": status}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/retrain_model")
def retrain_model(symbol: str = "NVDA", x_admin_password: str | None = Header(default=None)):
    try:
        require_admin(x_admin_password)
        refresh_service.retrain_model(symbol=symbol)
        return {"symbol": symbol, "status": refresh_service.get_status()}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/status")
def admin_status(
    symbol: str = "NVDA",
    event_limit: int = 10,
    x_admin_password: str | None = Header(default=None),
):
    try:
        require_admin(x_admin_password)
        return refresh_service.get_admin_snapshot(symbol=symbol, event_limit=event_limit)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
