from __future__ import annotations

import json

from src.data.repositories.market_data import LocalMarketDataRepository
from src.data.repositories.sentiment_repository import SentimentSnapshotRepository


def main() -> None:
    repository = LocalMarketDataRepository()
    candle_result = repository.sync_dataset_to_supabase(ticker="NVDA")
    technical_result = repository.sync_technicals_to_supabase(ticker="NVDA")
    sentiment_result = SentimentSnapshotRepository().sync_sentiment_snapshot(symbol="NVDA")
    print(json.dumps(
        {
            "candles": candle_result,
            "technicals": technical_result,
            "sentiment": {
                "symbol": sentiment_result.symbol,
                "label": sentiment_result.label,
                "score": sentiment_result.score,
                "article_count": sentiment_result.article_count,
                "ts": sentiment_result.ts,
            },
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
