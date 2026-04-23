"""
Live sentiment helper for equities.

- Uses Alpaca news as the primary source.
- Scores recent headlines with a lightweight keyword heuristic.
- Returns a normalized sentiment object that can be cached or persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.integrations.alpaca_client import AlpacaClient, AlpacaNewsItem


@dataclass
class SentimentNewsItem:
    source: str
    headline: str
    url: str
    datetime: str


@dataclass
class SentimentResult:
    score: float
    label: str
    raw: Dict[str, Any]
    news: List[SentimentNewsItem]


def _label_from_score(score: float) -> str:
    if score > 0.15:
        return "bullish"
    if score < -0.15:
        return "bearish"
    return "neutral"


def _compute_headline_sentiment(news_data: List[Dict[str, Any]]) -> float:
    if not news_data:
        return 0.0

    positive_words = [
        "beat",
        "beats",
        "record",
        "strong",
        "surge",
        "rally",
        "gain",
        "gains",
        "bullish",
        "upgrade",
        "upgraded",
        "buy",
        "outperform",
        "growth",
        "accelerate",
        "optimistic",
        "raised",
        "raise",
    ]
    negative_words = [
        "miss",
        "misses",
        "cut",
        "cuts",
        "downgrade",
        "downgraded",
        "sell",
        "underperform",
        "bearish",
        "loss",
        "losses",
        "decline",
        "drop",
        "drops",
        "plunge",
        "plunges",
        "slump",
        "weak",
        "lawsuit",
        "investigation",
        "regulator",
    ]

    score = 0
    hits = 0

    for item in news_data:
        text = ((item.get("headline") or "") + " " + (item.get("summary") or "")).lower()
        for word in positive_words:
            if word in text:
                score += 1
                hits += 1
        for word in negative_words:
            if word in text:
                score -= 1
                hits += 1

    if hits == 0:
        return 0.0

    raw = score / hits
    return max(-1.0, min(1.0, raw))


def get_symbol_sentiment(symbol: str = "NVDA", lookback_days: int = 5) -> SentimentResult:
    client = AlpacaClient()
    if not client.is_configured():
        return SentimentResult(
            score=0.0,
            label="neutral",
            raw={"reason": "Alpaca credentials not set", "symbol": symbol},
            news=[],
        )

    try:
        news_items_raw = client.get_news(symbol=symbol, lookback_days=lookback_days, limit=50)
        news_data = [
            {
                "headline": item.headline,
                "summary": item.summary,
                "source": item.source,
                "url": item.url,
                "datetime": item.datetime,
            }
            for item in news_items_raw
        ]

        news_items: List[SentimentNewsItem] = []
        for item in news_items_raw[:10]:
            news_items.append(
                SentimentNewsItem(
                    source=item.source,
                    headline=item.headline,
                    url=item.url,
                    datetime=item.datetime,
                )
            )

        score = _compute_headline_sentiment(news_data)
        label = _label_from_score(score)

        return SentimentResult(
            score=score,
            label=label,
            raw={
                "symbol": symbol,
                "article_count": len(news_data),
                "provider": "alpaca_news",
                "method": "headline_keyword_heuristic",
                "lookback_days": lookback_days,
            },
            news=news_items,
        )
    except Exception as exc:
        return SentimentResult(
            score=0.0,
            label="neutral",
            raw={"symbol": symbol, "error": str(exc)},
            news=[],
        )


def get_nvda_sentiment() -> SentimentResult:
    return get_symbol_sentiment(symbol="NVDA")
