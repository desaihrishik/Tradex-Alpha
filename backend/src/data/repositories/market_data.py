from __future__ import annotations

from dataclasses import dataclass
import time

import pandas as pd

from src.build_dataset import build_feature_frame
from src.core.config import get_settings
from src.core.supabase_client import get_supabase_admin_client
from src.integrations.alpaca_client import AlpacaClient


@dataclass(frozen=True)
class CandleRecord:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    patterns: list[str]
    signal: int
    source: str = "unknown"


class LocalMarketDataRepository:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.alpaca = AlpacaClient()

    @staticmethod
    def _normalize_trading_date(value: object) -> str:
        return pd.to_datetime(value).strftime("%Y-%m-%d")

    @staticmethod
    def _source_priority(source: str | None) -> int:
        normalized = str(source or "unknown").lower()
        priorities = {
            "alpaca_sip": 3,
            "alpaca_iex": 2,
            "alpaca_live": 1,
            "csv_seed": 0,
            "unknown": -1,
        }
        return priorities.get(normalized, 0)

    def _dedupe_market_rows(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        best_by_date: dict[str, dict[str, object]] = {}
        for row in rows:
            trading_date = self._normalize_trading_date(row["ts"])
            existing = best_by_date.get(trading_date)
            if existing is None:
                best_by_date[trading_date] = row
                continue

            current_priority = self._source_priority(row.get("source"))
            existing_priority = self._source_priority(existing.get("source"))
            current_ts = pd.to_datetime(row["ts"])
            existing_ts = pd.to_datetime(existing["ts"])

            if current_priority > existing_priority or (
                current_priority == existing_priority and current_ts > existing_ts
            ):
                best_by_date[trading_date] = row

        return sorted(best_by_date.values(), key=lambda row: pd.to_datetime(row["ts"]))

    def _fetch_live_daily_history(self, ticker: str, days: int = 540) -> pd.DataFrame:
        bars_result = self.alpaca.get_daily_bars(ticker, days=days)
        bars = bars_result.bars
        if not bars:
            raise RuntimeError(f"No Alpaca bars found for {ticker}.")

        frame = pd.DataFrame(
            {
                "Date": [pd.to_datetime(bar["t"], utc=True).tz_convert(None) for bar in bars],
                "Open": [float(bar["o"]) for bar in bars],
                "High": [float(bar["h"]) for bar in bars],
                "Low": [float(bar["l"]) for bar in bars],
                "Close": [float(bar["c"]) for bar in bars],
                "Volume": [float(bar["v"]) for bar in bars],
                "Dividends": [0.0 for _ in bars],
                "Stock Splits": [0.0 for _ in bars],
                "Source": [f"alpaca_{bars_result.feed}" for _ in bars],
            }
        )
        return frame.sort_values("Date").reset_index(drop=True)

    def _get_symbol_id(self, ticker: str) -> int | None:
        client = get_supabase_admin_client()
        if client is None:
            return None

        result = (
            client.table("symbols")
            .select("id")
            .eq("ticker", ticker)
            .limit(1)
            .execute()
        )
        if not result.data:
            return None
        return int(result.data[0]["id"])

    def ensure_symbol(self, ticker: str, name: str | None = None) -> int | None:
        client = get_supabase_admin_client()
        if client is None:
            return None

        existing_id = self._get_symbol_id(ticker)
        if existing_id is not None:
            return existing_id

        payload = {
            "ticker": ticker,
            "name": name or ticker,
            "is_active": True,
        }
        result = client.table("symbols").insert(payload).execute()
        if not result.data:
            return None
        return int(result.data[0]["id"])

    def sync_recent_market_data(
        self,
        *,
        ticker: str = "NVDA",
        timeframe: str = "1d",
        period: str = "18mo",
        source: str = "alpaca_live",
    ) -> dict[str, object]:
        client = get_supabase_admin_client()
        if client is None:
            raise RuntimeError("Supabase client is not configured or dependency is missing.")

        symbol_id = self.ensure_symbol(ticker=ticker, name=ticker)
        if symbol_id is None:
            raise RuntimeError(f"Unable to ensure symbol row for {ticker}.")

        history = self._fetch_live_daily_history(ticker=ticker)
        if history.empty:
            return {
                "symbol": ticker,
                "symbol_id": symbol_id,
                "timeframe": timeframe,
                "rows_upserted": 0,
            }

        records = []
        for _, row in history.iterrows():
            ts = pd.to_datetime(row["Date"]).normalize()
            records.append(
                {
                    "symbol_id": symbol_id,
                    "timeframe": timeframe,
                    "ts": ts.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                    "source": str(row.get("Source") or source),
                }
            )

        client.table("market_candles").upsert(
            records,
            on_conflict="symbol_id,timeframe,ts",
        ).execute()

        try:
            self.sync_recent_technicals_to_supabase(
                ticker=ticker,
                timeframe=timeframe,
                raw_history=history,
            )
        except Exception:
            pass

        return {
            "symbol": ticker,
            "symbol_id": symbol_id,
            "timeframe": timeframe,
            "rows_upserted": len(records),
        }

    def sync_recent_technicals_to_supabase(
        self,
        *,
        ticker: str = "NVDA",
        timeframe: str = "1d",
        raw_history: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        client = get_supabase_admin_client()
        if client is None:
            raise RuntimeError("Supabase client is not configured or dependency is missing.")

        symbol_id = self.ensure_symbol(ticker=ticker, name=ticker)
        if symbol_id is None:
            raise RuntimeError(f"Unable to ensure symbol row for {ticker}.")

        history = raw_history.copy() if raw_history is not None else self._fetch_live_daily_history(ticker=ticker)
        if history.empty:
            return {
                "symbol": ticker,
                "symbol_id": symbol_id,
                "timeframe": timeframe,
                "rows_upserted": 0,
            }

        if "Date" not in history.columns:
            history = history.reset_index()
        dates = pd.to_datetime(history["Date"], utc=True).dt.tz_convert(None)
        feature_input = pd.DataFrame(
            {
                "Date": dates,
                "Open": history["Open"].astype(float),
                "High": history["High"].astype(float),
                "Low": history["Low"].astype(float),
                "Close": history["Close"].astype(float),
                "Volume": history["Volume"].astype(float),
                "Dividends": history["Dividends"].astype(float) if "Dividends" in history.columns else 0.0,
                "Stock Splits": history["Stock Splits"].astype(float) if "Stock Splits" in history.columns else 0.0,
            }
        )

        featured = build_feature_frame(feature_input)
        pattern_cols = self._pattern_columns(featured)
        indicator_cols = self._indicator_columns(featured)

        records = []
        for _, row in featured.iterrows():
            patterns = [
                col.replace("pattern_", "")
                for col in pattern_cols
                if int(row[col]) == 1
            ]
            indicators = {}
            for col in indicator_cols:
                value = row[col]
                if pd.isna(value):
                    indicators[col] = None
                elif isinstance(value, (int, float, bool)):
                    indicators[col] = float(value) if not isinstance(value, bool) else value
                else:
                    indicators[col] = value

            trend_label = self._infer_trend_label(
                ema_5_dist=float(indicators.get("ema_5_dist") or 0.0),
                ema_12_dist=float(indicators.get("ema_12_dist") or 0.0),
                rsi_14=float(indicators.get("rsi_14") or 50.0),
            )

            records.append(
                {
                    "symbol_id": symbol_id,
                    "timeframe": timeframe,
                    "ts": pd.to_datetime(row["Date"]).isoformat(),
                    "indicators": indicators,
                    "patterns": patterns,
                    "trend_label": trend_label,
                    "trend_strength": float(indicators.get("ema_5_dist") or 0.0),
                }
            )

        chunk_size = 500
        upserted = 0
        for start in range(0, len(records), chunk_size):
            chunk = records[start:start + chunk_size]
            client.table("technical_snapshots").upsert(
                chunk,
                on_conflict="symbol_id,timeframe,ts",
            ).execute()
            upserted += len(chunk)

        return {
            "symbol": ticker,
            "symbol_id": symbol_id,
            "timeframe": timeframe,
            "rows_upserted": upserted,
        }

    def get_raw_market_history(self, ticker: str = "NVDA", limit: int = 400) -> pd.DataFrame:
        client = get_supabase_admin_client()
        if client is not None:
            try:
                symbol_id = self._get_symbol_id(ticker)
                if symbol_id is not None:
                    result = (
                        client.table("market_candles")
                        .select("ts, open, high, low, close, volume, source")
                        .eq("symbol_id", symbol_id)
                        .eq("timeframe", "1d")
                        .order("ts", desc=True)
                        .limit(limit)
                        .execute()
                    )
                    if result.data:
                        rows = self._dedupe_market_rows(result.data)
                        if limit:
                            rows = rows[-limit:]
                        df = pd.DataFrame(
                            {
                                "Date": [pd.to_datetime(row["ts"]) for row in rows],
                                "Open": [float(row["open"]) for row in rows],
                                "High": [float(row["high"]) for row in rows],
                                "Low": [float(row["low"]) for row in rows],
                                "Close": [float(row["close"]) for row in rows],
                                "Volume": [float(row["volume"]) for row in rows],
                                "Dividends": [0.0 for _ in rows],
                                "Stock Splits": [0.0 for _ in rows],
                                "Source": [str(row.get("source") or "unknown") for row in rows],
                            }
                        )
                        return df
            except Exception:
                pass

        try:
            history = self._fetch_live_daily_history(ticker=ticker, days=max(limit + 60, 540))
            if not history.empty:
                return history.tail(limit).reset_index(drop=True)
        except Exception:
            pass

        raw_csv = self.settings.data_dir / "nvda_daily_5y.csv"
        if raw_csv.exists():
            return pd.read_csv(raw_csv, parse_dates=["Date"])
        return self.load_dataset()[["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]].copy()

    def _get_supabase_candles(self, ticker: str, limit: int) -> list[CandleRecord]:
        client = get_supabase_admin_client()
        if client is None:
            return []

        try:
            symbol_id = self._get_symbol_id(ticker)
            if symbol_id is None:
                return []

            result = (
                client.table("market_candles")
                .select("ts, open, high, low, close, volume, source")
                .eq("symbol_id", symbol_id)
                .eq("timeframe", "1d")
                .order("ts", desc=True)
                .limit(limit)
                .execute()
            )
            if not result.data:
                return []

            market_rows = self._dedupe_market_rows(result.data)
            if limit:
                market_rows = market_rows[-limit:]

            technical_result = (
                client.table("technical_snapshots")
                .select("ts, patterns, indicators")
                .eq("symbol_id", symbol_id)
                .eq("timeframe", "1d")
                .order("ts", desc=True)
                .limit(limit)
                .execute()
            )
            technical_map = {
                pd.to_datetime(row["ts"]).strftime("%Y-%m-%d"): row
                for row in (technical_result.data or [])
            }

            candles = [
                CandleRecord(
                    date=self._normalize_trading_date(row["ts"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    patterns=list(technical_map.get(self._normalize_trading_date(row["ts"]), {}).get("patterns") or []),
                    signal=int(
                        (technical_map.get(self._normalize_trading_date(row["ts"]), {}).get("indicators") or {}).get("signal", 0)
                    ),
                    source=str(row.get("source") or "unknown"),
                )
                for row in market_rows
            ]
            return candles
        except Exception:
            return []

    def sync_dataset_to_supabase(
        self,
        *,
        ticker: str = "NVDA",
        timeframe: str = "1d",
        source: str = "csv_seed",
    ) -> dict[str, object]:
        client = get_supabase_admin_client()
        if client is None:
            raise RuntimeError("Supabase client is not configured or dependency is missing.")

        symbol_id = self.ensure_symbol(ticker=ticker, name=ticker)
        if symbol_id is None:
            raise RuntimeError(f"Unable to ensure symbol row for {ticker}.")

        df = self.load_dataset().sort_values("Date").reset_index(drop=True)
        records = [
            {
                "symbol_id": symbol_id,
                "timeframe": timeframe,
                "ts": pd.to_datetime(row["Date"]).isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
                "source": source,
            }
            for _, row in df.iterrows()
        ]

        chunk_size = 500
        inserted = 0
        for start in range(0, len(records), chunk_size):
            chunk = records[start:start + chunk_size]
            client.table("market_candles").upsert(
                chunk,
                on_conflict="symbol_id,timeframe,ts",
            ).execute()
            inserted += len(chunk)

        return {
            "symbol": ticker,
            "symbol_id": symbol_id,
            "timeframe": timeframe,
            "rows_upserted": inserted,
        }

    def sync_technicals_to_supabase(
        self,
        *,
        ticker: str = "NVDA",
        timeframe: str = "1d",
    ) -> dict[str, object]:
        client = get_supabase_admin_client()
        if client is None:
            raise RuntimeError("Supabase client is not configured or dependency is missing.")

        symbol_id = self.ensure_symbol(ticker=ticker, name=ticker)
        if symbol_id is None:
            raise RuntimeError(f"Unable to ensure symbol row for {ticker}.")

        df = self.load_dataset().sort_values("Date").reset_index(drop=True)
        pattern_cols = self._pattern_columns(df)
        indicator_cols = self._indicator_columns(df)

        records = []
        for _, row in df.iterrows():
            patterns = [
                col.replace("pattern_", "")
                for col in pattern_cols
                if int(row[col]) == 1
            ]
            indicators = {}
            for col in indicator_cols:
                value = row[col]
                if pd.isna(value):
                    indicators[col] = None
                elif isinstance(value, (int, float, bool)):
                    indicators[col] = float(value) if not isinstance(value, bool) else value
                else:
                    indicators[col] = value

            records.append(
                {
                    "symbol_id": symbol_id,
                    "timeframe": timeframe,
                    "ts": pd.to_datetime(row["Date"]).isoformat(),
                    "indicators": indicators,
                    "patterns": patterns,
                    "trend_label": None,
                    "trend_strength": indicators.get("c_direction"),
                }
            )

        chunk_size = 500
        upserted = 0
        for start in range(0, len(records), chunk_size):
            chunk = records[start:start + chunk_size]
            client.table("technical_snapshots").upsert(
                chunk,
                on_conflict="symbol_id,timeframe,ts",
            ).execute()
            upserted += len(chunk)

        return {
            "symbol": ticker,
            "symbol_id": symbol_id,
            "timeframe": timeframe,
            "rows_upserted": upserted,
        }

    def load_dataset(self) -> pd.DataFrame:
        dataset_path = self.settings.dataset_csv
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        return pd.read_csv(dataset_path, parse_dates=["Date"])

    def get_latest_market_snapshot(self, ticker: str = "NVDA", *, refresh: bool = True) -> dict[str, object]:
        client = get_supabase_admin_client()
        if client is not None:
            if refresh:
                try:
                    self.sync_recent_market_data(ticker=ticker)
                except Exception:
                    pass
            symbol_id = self._get_symbol_id(ticker)
            if symbol_id is not None:
                for attempt in range(2):
                    try:
                        candle_result = (
                            client.table("market_candles")
                            .select("ts, open, high, low, close, volume, source")
                            .eq("symbol_id", symbol_id)
                            .eq("timeframe", "1d")
                            .order("ts", desc=True)
                            .limit(2)
                            .execute()
                        )
                        technical_result = (
                            client.table("technical_snapshots")
                            .select("ts, indicators, patterns, trend_label, trend_strength")
                            .eq("symbol_id", symbol_id)
                            .eq("timeframe", "1d")
                            .order("ts", desc=True)
                            .limit(1)
                            .execute()
                        )
                        recommendation_result = (
                            client.table("recommendations")
                            .select("action, confidence, suggested_amount, suggested_shares, suggested_duration_days, sentiment_label, sentiment_score, explanation, ts")
                            .eq("symbol_id", symbol_id)
                            .order("ts", desc=True)
                            .limit(1)
                            .execute()
                        )

                        candles = candle_result.data or []
                        candles = list(reversed(self._dedupe_market_rows(candles)))
                        latest_technical = (technical_result.data or [None])[0]
                        latest_recommendation = (recommendation_result.data or [None])[0]
                        if candles:
                            latest = candles[0]
                            previous = candles[1] if len(candles) > 1 else None
                            latest_close = float(latest["close"])
                            previous_close = float(previous["close"]) if previous else latest_close
                            change = latest_close - previous_close
                            change_pct = (change / previous_close) if previous_close else 0.0
                            indicators = (latest_technical or {}).get("indicators") or {}
                            trend_strength = (latest_technical or {}).get("trend_strength")
                            inferred_trend = "sideways"
                            if change_pct > 0.01:
                                inferred_trend = "bullish"
                            elif change_pct < -0.01:
                                inferred_trend = "bearish"

                            return {
                                "symbol": ticker,
                                "as_of": pd.to_datetime(latest["ts"]).strftime("%Y-%m-%d"),
                                "latest_close": latest_close,
                                "previous_close": previous_close,
                                "change": change,
                                "change_pct": change_pct,
                                "trend_label": (latest_technical or {}).get("trend_label") or inferred_trend,
                                "trend_strength": trend_strength if trend_strength is not None else indicators.get("c_direction", 0),
                                "patterns": (latest_technical or {}).get("patterns") or [],
                                "volume": float(latest["volume"]),
                                "market_data_source": str(latest.get("source") or "unknown"),
                                "recommendation": latest_recommendation,
                            }
                    except Exception:
                        if attempt == 0:
                            time.sleep(0.25)
                            continue

        candles = self.get_raw_market_history(ticker=ticker, limit=2)
        if not candles.empty:
            candles = candles.sort_values("Date").reset_index(drop=True)
            latest = candles.iloc[-1]
            previous = candles.iloc[-2] if len(candles) > 1 else latest
            latest_close = float(latest["Close"])
            previous_close = float(previous["Close"]) if previous is not None else latest_close
            change = latest_close - previous_close
            change_pct = (change / previous_close) if previous_close else 0.0
            inferred_trend = "sideways"
            if change_pct > 0.01:
                inferred_trend = "bullish"
            elif change_pct < -0.01:
                inferred_trend = "bearish"
            return {
                "symbol": ticker,
                "as_of": pd.to_datetime(latest["Date"]).strftime("%Y-%m-%d"),
                "latest_close": latest_close,
                "previous_close": previous_close,
                "change": change,
                "change_pct": change_pct,
                "trend_label": inferred_trend,
                "trend_strength": 0,
                "patterns": [],
                "volume": float(latest["Volume"]),
                "market_data_source": str(latest.get("Source") or "unknown"),
                "recommendation": None,
            }
        return {}

    def get_candles(self, limit: int = 120, ticker: str = "NVDA") -> list[CandleRecord]:
        try:
            self.sync_recent_market_data(ticker=ticker)
        except Exception:
            pass

        supabase_candles = self._get_supabase_candles(ticker=ticker, limit=limit)
        if supabase_candles:
            return supabase_candles

        df = self.load_dataset()
        if limit:
            df = df.tail(int(limit))

        pattern_cols = [c for c in df.columns if c.startswith("pattern_")]
        candles: list[CandleRecord] = []
        for _, row in df.iterrows():
            patterns = [
                c.replace("pattern_", "")
                for c in pattern_cols
                if int(row[c]) == 1
            ]
            candles.append(
                CandleRecord(
                    date=pd.to_datetime(row["Date"]).strftime("%Y-%m-%d"),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    patterns=patterns,
                    signal=int(row["signal"]),
                    source=str(row.get("Source") or "csv_seed"),
                )
            )
        return candles
    @staticmethod
    def _pattern_columns(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c.startswith("pattern_")]

    @staticmethod
    def _indicator_columns(df: pd.DataFrame) -> list[str]:
        excluded = {
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "future_return_1d",
        }
        return [
            c for c in df.columns
            if c not in excluded and not c.startswith("pattern_")
        ]

    @staticmethod
    def _infer_trend_label(*, ema_5_dist: float, ema_12_dist: float, rsi_14: float) -> str:
        if ema_5_dist > 0.01 and ema_12_dist > 0.02 and rsi_14 >= 55:
            return "bullish"
        if ema_5_dist < -0.01 and ema_12_dist < -0.02 and rsi_14 <= 45:
            return "bearish"
        return "sideways"
