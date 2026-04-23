from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from src.core.config import get_settings


@dataclass(frozen=True)
class AlpacaNewsItem:
    source: str
    headline: str
    url: str
    summary: str
    datetime: str


@dataclass(frozen=True)
class AlpacaBarsResult:
    bars: list[dict[str, Any]]
    feed: str


class AlpacaClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def is_configured(self) -> bool:
        return bool(self.settings.alpaca_api_key and self.settings.alpaca_secret_key)

    def _headers(self) -> dict[str, str]:
        if not self.is_configured():
            raise RuntimeError("Alpaca credentials are not configured.")
        return {
            "APCA-API-KEY-ID": str(self.settings.alpaca_api_key),
            "APCA-API-SECRET-KEY": str(self.settings.alpaca_secret_key),
            "Accept": "application/json",
        }

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.settings.alpaca_data_url.rstrip('/')}/{path.lstrip('/')}"
        response = requests.get(url, headers=self._headers(), params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_daily_bars(self, symbol: str, *, days: int = 540, limit: int | None = None) -> AlpacaBarsResult:
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
        preferred_feed = (self.settings.alpaca_stock_feed or "sip").lower()
        feeds_to_try = [preferred_feed]
        if self.settings.alpaca_fallback_to_iex and preferred_feed != "iex":
            feeds_to_try.append("iex")

        last_error: Exception | None = None
        for feed in feeds_to_try:
            try:
                payload = self._get(
                    "/v2/stocks/bars",
                    params={
                        "symbols": symbol,
                        "timeframe": "1Day",
                        "start": start,
                        "adjustment": "raw",
                        "feed": feed,
                        "sort": "asc",
                        "limit": limit or min(days + 30, 10000),
                    },
                )
                bars = (payload.get("bars") or {}).get(symbol) or []
                if bars:
                    return AlpacaBarsResult(bars=list(bars), feed=feed)
            except Exception as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"No Alpaca bars returned for {symbol}.")

    def get_news(self, symbol: str, *, lookback_days: int = 5, limit: int = 50) -> list[AlpacaNewsItem]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        payload = self._get(
            "/v1beta1/news",
            params={
                "symbols": symbol,
                "start": start.strftime("%Y-%m-%dT00:00:00Z"),
                "end": end.strftime("%Y-%m-%dT23:59:59Z"),
                "sort": "desc",
                "limit": limit,
            },
        )
        items = payload.get("news") or payload.get("articles") or []
        return [
            AlpacaNewsItem(
                source=str(item.get("source", "")),
                headline=str(item.get("headline", "")),
                url=str(item.get("url", "")),
                summary=str(item.get("summary", "")),
                datetime=str(item.get("created_at") or item.get("updated_at") or item.get("timestamp") or ""),
            )
            for item in items
        ]
