from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.core.supabase_client import get_supabase_admin_client
from src.data.repositories.market_data import LocalMarketDataRepository
from src.sentiment_engine import SentimentResult, get_symbol_sentiment


@dataclass(frozen=True)
class SentimentSnapshot:
    symbol: str
    ts: str
    label: str
    score: float
    article_count: int
    raw: dict[str, Any]


class SentimentSnapshotRepository:
    def __init__(self) -> None:
        self.market_data = LocalMarketDataRepository()

    def _get_symbol_id(self, symbol: str) -> int | None:
        return self.market_data.ensure_symbol(symbol, name=symbol)

    def _map_result(self, *, symbol: str, ts: str, sentiment: SentimentResult) -> SentimentSnapshot:
        return SentimentSnapshot(
            symbol=symbol,
            ts=ts,
            label=sentiment.label,
            score=float(sentiment.score),
            article_count=int(sentiment.raw.get("article_count", len(sentiment.news))),
            raw={
                **sentiment.raw,
                "news": [
                    {
                        "source": item.source,
                        "headline": item.headline,
                        "url": item.url,
                        "datetime": item.datetime,
                    }
                    for item in sentiment.news
                ],
            },
        )

    def fetch_live_sentiment(self, symbol: str = "NVDA") -> SentimentSnapshot:
        sentiment = get_symbol_sentiment(symbol=symbol)
        ts = datetime.now(timezone.utc).isoformat()
        return self._map_result(symbol=symbol, ts=ts, sentiment=sentiment)

    def persist_snapshot(self, snapshot: SentimentSnapshot) -> SentimentSnapshot:
        client = get_supabase_admin_client()
        if client is None:
            return snapshot

        symbol_id = self._get_symbol_id(snapshot.symbol)
        if symbol_id is None:
            return snapshot

        payload = {
            "symbol_id": symbol_id,
            "ts": snapshot.ts,
            "sentiment_label": snapshot.label,
            "sentiment_score": snapshot.score,
            "article_count": snapshot.article_count,
            "raw": snapshot.raw,
        }
        client.table("sentiment_snapshots").insert(payload).execute()
        return snapshot

    def sync_sentiment_snapshot(self, symbol: str = "NVDA") -> SentimentSnapshot:
        snapshot = self.fetch_live_sentiment(symbol=symbol)
        return self.persist_snapshot(snapshot)

    def get_latest_snapshot(
        self,
        symbol: str = "NVDA",
        *,
        max_age_minutes: int = 30,
        refresh_if_stale: bool = True,
    ) -> SentimentSnapshot:
        client = get_supabase_admin_client()
        if client is not None:
            try:
                symbol_id = self._get_symbol_id(symbol)
                if symbol_id is not None:
                    result = (
                        client.table("sentiment_snapshots")
                        .select("ts, sentiment_label, sentiment_score, article_count, raw")
                        .eq("symbol_id", symbol_id)
                        .order("ts", desc=True)
                        .limit(1)
                        .execute()
                    )
                    if result.data:
                        row = result.data[0]
                        snapshot = SentimentSnapshot(
                            symbol=symbol,
                            ts=str(row["ts"]),
                            label=str(row["sentiment_label"]),
                            score=float(row["sentiment_score"]),
                            article_count=int(row.get("article_count") or 0),
                            raw=dict(row.get("raw") or {}),
                        )
                        snapshot_dt = pd.to_datetime(snapshot.ts, utc=True).to_pydatetime()
                        age = datetime.now(timezone.utc) - snapshot_dt
                        if age <= timedelta(minutes=max_age_minutes) or not refresh_if_stale:
                            return snapshot
            except Exception:
                pass

        if refresh_if_stale:
            return self.sync_sentiment_snapshot(symbol=symbol)

        return self.fetch_live_sentiment(symbol=symbol)

    def get_recent_snapshots(self, symbol: str = "NVDA", limit: int = 2) -> list[SentimentSnapshot]:
        client = get_supabase_admin_client()
        if client is None:
            latest = self.get_latest_snapshot(symbol=symbol, refresh_if_stale=False)
            return [latest]

        try:
            symbol_id = self._get_symbol_id(symbol)
            if symbol_id is None:
                return []

            result = (
                client.table("sentiment_snapshots")
                .select("ts, sentiment_label, sentiment_score, article_count, raw")
                .eq("symbol_id", symbol_id)
                .order("ts", desc=True)
                .limit(limit)
                .execute()
            )
            snapshots = []
            for row in result.data or []:
                snapshots.append(
                    SentimentSnapshot(
                        symbol=symbol,
                        ts=str(row["ts"]),
                        label=str(row["sentiment_label"]),
                        score=float(row["sentiment_score"]),
                        article_count=int(row.get("article_count") or 0),
                        raw=dict(row.get("raw") or {}),
                    )
                )
            return snapshots
        except Exception:
            return []
