from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from src.core.config import get_settings
from src.data.repositories.admin_status import AdminStatusRepository
from src.data.repositories.market_data import LocalMarketDataRepository
from src.data.repositories.sentiment_repository import SentimentSnapshotRepository
from src.services.model_training_service import ModelTrainingService


@dataclass
class RefreshStatus:
    market_last_run: str | None = None
    market_last_error: str | None = None
    sentiment_last_run: str | None = None
    sentiment_last_error: str | None = None
    model_last_run: str | None = None
    model_last_error: str | None = None
    model_version: str | None = None


class RefreshService:
    def __init__(self) -> None:
        self.service_name = "edge-refresh-service"
        self.settings = get_settings()
        self.admin_status_repo = AdminStatusRepository()
        self.market_data = LocalMarketDataRepository()
        self.sentiment_repo = SentimentSnapshotRepository()
        self.model_training = ModelTrainingService()
        self.status = RefreshStatus()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _persist_admin_status(self, *, symbol: str, event_type: str) -> None:
        status = self.get_status()
        self.admin_status_repo.upsert_runtime_status(
            service_name=self.service_name,
            symbol=symbol,
            status=status,
        )
        self.admin_status_repo.insert_event(
            service_name=self.service_name,
            symbol=symbol,
            event_type=event_type,
            status=status,
        )

    def refresh_market(self, symbol: str = "NVDA") -> None:
        try:
            self.market_data.sync_recent_market_data(ticker=symbol)
            self.status.market_last_run = self._now()
            self.status.market_last_error = None
        except Exception as exc:
            self.status.market_last_error = str(exc)
        self._persist_admin_status(symbol=symbol, event_type="market_refresh")

    def refresh_sentiment(self, symbol: str = "NVDA") -> None:
        try:
            self.sentiment_repo.sync_sentiment_snapshot(symbol=symbol)
            self.status.sentiment_last_run = self._now()
            self.status.sentiment_last_error = None
        except Exception as exc:
            self.status.sentiment_last_error = str(exc)
        self._persist_admin_status(symbol=symbol, event_type="sentiment_refresh")

    def refresh_all(self, symbol: str = "NVDA") -> dict[str, str | None]:
        self.refresh_market(symbol=symbol)
        self.refresh_sentiment(symbol=symbol)
        return self.get_status()

    def retrain_model(self, symbol: str = "NVDA") -> None:
        try:
            result = self.model_training.retrain_model(symbol=symbol)
            self.status.model_last_run = self._now()
            self.status.model_last_error = None
            self.status.model_version = result.version
        except Exception as exc:
            self.status.model_last_error = str(exc)
        self._persist_admin_status(symbol=symbol, event_type="model_retrain")

    def _loop(self) -> None:
        market_interval = max(60, self.settings.market_refresh_interval_seconds)
        sentiment_interval = max(300, self.settings.sentiment_refresh_interval_seconds)
        model_interval = max(1800, self.settings.model_retrain_interval_seconds)
        next_market = 0.0
        next_sentiment = 0.0
        next_model = 0.0

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now >= next_market:
                self.refresh_market()
                next_market = now + market_interval
            if now >= next_sentiment:
                self.refresh_sentiment()
                next_sentiment = now + sentiment_interval
            if self.settings.model_retrain_enabled and now >= next_model:
                self.retrain_model()
                next_model = now + model_interval
            self._stop_event.wait(timeout=5)

    def start(self) -> None:
        if not self.settings.auto_refresh_enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="edge-refresh-service", daemon=True)
        self._thread.start()
        self._persist_admin_status(symbol="NVDA", event_type="scheduler_start")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._persist_admin_status(symbol="NVDA", event_type="scheduler_stop")

    def get_status(self) -> dict[str, str | bool | None]:
        return {
            **self.status.__dict__.copy(),
            "auto_refresh_enabled": self.settings.auto_refresh_enabled,
            "model_retrain_enabled": self.settings.model_retrain_enabled,
            "scheduler_running": bool(self._thread and self._thread.is_alive()),
        }

    def get_admin_snapshot(self, *, symbol: str = "NVDA", event_limit: int = 10) -> dict[str, object]:
        return {
            "live_status": self.get_status(),
            "persisted_status": self.admin_status_repo.get_runtime_status(
                service_name=self.service_name,
                symbol=symbol,
            ),
            "recent_events": self.admin_status_repo.get_recent_events(
                service_name=self.service_name,
                symbol=symbol,
                limit=event_limit,
            ),
        }
