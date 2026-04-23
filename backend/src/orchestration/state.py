from __future__ import annotations

from typing import TypedDict


class AnalysisState(TypedDict, total=False):
    symbol: str
    budget: float
    risk_profile: str
    limit: int
    entry_price: float | None
    candles: list[dict[str, object]]
    latest_signal: dict[str, object]
    agentic_signal: dict[str, object]
    decision: dict[str, object]
    explanation: str
    errors: list[str]

