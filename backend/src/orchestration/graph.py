from __future__ import annotations

from typing import Any

from src.core.config import get_settings
from src.orchestration.nodes import (
    finalize_decision,
    load_agentic_signal,
    load_candles,
    load_latest_signal,
)
from src.orchestration.state import AnalysisState

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover
    END = START = StateGraph = None


def _run_sequential(state: AnalysisState, service: Any) -> AnalysisState:
    next_state = dict(state)
    for step in (load_candles, load_latest_signal, load_agentic_signal, finalize_decision):
        next_state.update(step(next_state, service))
    return next_state


def _build_langgraph(service: Any):
    graph = StateGraph(AnalysisState)
    graph.add_node("load_candles", lambda state: load_candles(state, service))
    graph.add_node("load_latest_signal", lambda state: load_latest_signal(state, service))
    graph.add_node("load_agentic_signal", lambda state: load_agentic_signal(state, service))
    graph.add_node("finalize_decision", lambda state: finalize_decision(state, service))
    graph.add_edge(START, "load_candles")
    graph.add_edge("load_candles", "load_latest_signal")
    graph.add_edge("load_latest_signal", "load_agentic_signal")
    graph.add_edge("load_agentic_signal", "finalize_decision")
    graph.add_edge("finalize_decision", END)
    return graph.compile()


def run_analysis_pipeline(
    *,
    symbol: str,
    budget: float,
    risk_profile: str,
    limit: int,
    entry_price: float | None,
    service: Any,
) -> dict[str, object]:
    initial_state: AnalysisState = {
        "symbol": symbol,
        "budget": budget,
        "risk_profile": risk_profile,
        "limit": limit,
        "entry_price": entry_price,
        "errors": [],
    }

    settings = get_settings()
    if settings.langgraph_enabled and StateGraph is not None:
        result = _build_langgraph(service).invoke(initial_state)
    else:
        result = _run_sequential(initial_state, service)

    return {
        "symbol": result.get("symbol", symbol),
        "candles": result.get("candles", []),
        "latest_signal": result.get("latest_signal", {}),
        "agentic_signal": result.get("agentic_signal", {}),
        "decision": result.get("decision", {}),
        "explanation": result.get("explanation", ""),
    }

