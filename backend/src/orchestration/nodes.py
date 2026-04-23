from __future__ import annotations

from src.orchestration.state import AnalysisState


def load_candles(state: AnalysisState, service: object) -> AnalysisState:
    candles = service.get_candles(limit=state.get("limit", 120))["candles"]
    return {"candles": candles}


def load_latest_signal(state: AnalysisState, service: object) -> AnalysisState:
    latest_signal = service.get_latest_signal(
        budget=state["budget"],
        risk=state["risk_profile"],
    )
    return {"latest_signal": latest_signal}


def load_agentic_signal(state: AnalysisState, service: object) -> AnalysisState:
    agentic_signal = service.get_agentic_signal(
        budget=state["budget"],
        risk=state["risk_profile"],
        entry_price=state.get("entry_price"),
    )
    return {"agentic_signal": agentic_signal}


def finalize_decision(state: AnalysisState, service: object) -> AnalysisState:
    del service
    agentic_signal = state.get("agentic_signal", {})
    latest_signal = state.get("latest_signal", {})
    decision = {
        "symbol": state.get("symbol", "NVDA"),
        "action": agentic_signal.get("action", latest_signal.get("action")),
        "confidence": agentic_signal.get("confidence", latest_signal.get("confidence")),
        "risk_profile": state.get("risk_profile"),
        "budget": state.get("budget"),
        "latest_signal": latest_signal,
        "agentic_signal": agentic_signal,
    }
    explanation = str(agentic_signal.get("explanation") or latest_signal.get("explanation") or "")
    return {"decision": decision, "explanation": explanation}

