from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.admin_routes import router as admin_router
from src.api.llm_routes import router as llm_router
from src.core.config import get_settings
from src.services.runtime_services import recommendation_service, refresh_service

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    refresh_service.start()
    try:
        yield
    finally:
        refresh_service.stop()

app = FastAPI(
    title=settings.app_name,
    description="NVDA prediction engine with patterns and ML signals",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "NVDA Edge AI Server Running"}


@app.get("/api/ping")
def ping():
    return {"status": "ok"}


@app.get("/api/nvda/latest_signal")
def latest_signal(budget: float = 1000.0, risk: str = "medium"):
    return recommendation_service.get_latest_signal(budget=budget, risk=risk)


@app.get("/api/nvda/current_trend")
def current_trend():
    return recommendation_service.get_current_trend(symbol="NVDA")


@app.get("/api/nvda/sentiment")
def sentiment_details():
    return recommendation_service.get_sentiment_details(symbol="NVDA", refresh=False)


@app.get("/api/nvda/candles")
def get_candles(limit: int = 120):
    return recommendation_service.get_candles(limit=limit)


@app.get("/api/nvda/agentic_signal")
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


@app.get("/api/nvda/analyze")
def analyze_nvda(
    budget: float = 1000.0,
    risk: str = "medium",
    limit: int = 120,
    entry_price: float | None = None,
    persist: bool = True,
):
    return recommendation_service.run_analysis(
        symbol="NVDA",
        budget=budget,
        risk=risk,
        limit=limit,
        entry_price=entry_price,
        persist=persist,
    )


app.include_router(llm_router)
app.include_router(admin_router)
