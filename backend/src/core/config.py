from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    app_name: str
    root_dir: Path
    data_dir: Path
    models_dir: Path
    dataset_csv: Path
    alpaca_api_key: str | None
    alpaca_secret_key: str | None
    alpaca_base_url: str
    alpaca_data_url: str
    alpaca_stock_feed: str
    alpaca_fallback_to_iex: bool
    openai_api_key: str | None
    openai_model: str
    supabase_url: str | None
    supabase_anon_key: str | None
    supabase_service_role_key: str | None
    supabase_models_bucket: str
    supabase_model_metadata_bucket: str
    supabase_datasets_bucket: str
    langgraph_enabled: bool
    auto_refresh_enabled: bool
    market_refresh_interval_seconds: int
    sentiment_refresh_interval_seconds: int
    model_retrain_enabled: bool
    model_retrain_interval_seconds: int
    admin_panel_password: str


def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parents[2]
    load_dotenv(root_dir / ".env")
    return Settings(
        app_name="Edge AI Trading Backend",
        root_dir=root_dir,
        data_dir=root_dir / "data",
        models_dir=root_dir / "models",
        dataset_csv=root_dir / "data" / "nvda_daily_dataset.csv",
        alpaca_api_key=os.getenv("ALPACA_API_KEY"),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY"),
        alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        alpaca_data_url=os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets"),
        alpaca_stock_feed=os.getenv("ALPACA_STOCK_FEED", "sip"),
        alpaca_fallback_to_iex=os.getenv("ALPACA_FALLBACK_TO_IEX", "true").lower() == "true",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_anon_key=os.getenv("SUPABASE_ANON_KEY"),
        supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        supabase_models_bucket=os.getenv("SUPABASE_MODELS_BUCKET", "models"),
        supabase_model_metadata_bucket=os.getenv("SUPABASE_MODEL_METADATA_BUCKET", "model-metadata"),
        supabase_datasets_bucket=os.getenv("SUPABASE_DATASETS_BUCKET", "datasets"),
        langgraph_enabled=os.getenv("LANGGRAPH_ENABLED", "false").lower() == "true",
        auto_refresh_enabled=os.getenv("AUTO_REFRESH_ENABLED", "true").lower() == "true",
        market_refresh_interval_seconds=int(os.getenv("MARKET_REFRESH_INTERVAL_SECONDS", "900")),
        sentiment_refresh_interval_seconds=int(os.getenv("SENTIMENT_REFRESH_INTERVAL_SECONDS", "1200")),
        model_retrain_enabled=os.getenv("MODEL_RETRAIN_ENABLED", "true").lower() == "true",
        model_retrain_interval_seconds=int(os.getenv("MODEL_RETRAIN_INTERVAL_SECONDS", "14400")),
        admin_panel_password=os.getenv("ADMIN_PANEL_PASSWORD", "changeme-admin"),
    )
