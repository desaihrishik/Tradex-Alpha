from __future__ import annotations

import json

import requests

from src.core.config import get_settings

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def ask_llm(prompt: str, *, system: str | None = None) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        return "Error communicating with LLM: OPENAI_API_KEY is not configured."

    try:
        input_items = []
        if system:
            input_items.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system}],
                }
            )
        input_items.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        )

        response = requests.post(
            OPENAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.openai_model,
                "input": input_items,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("output_text"):
            return str(data["output_text"])

        output = data.get("output") or []
        for item in output:
            for content in item.get("content") or []:
                if content.get("type") == "output_text":
                    return str(content.get("text") or "<<No response>>")

        return "<<No response>>"
    except Exception as exc:
        return f"Error communicating with LLM: {exc}"


def ask_trade_question(*, question: str, context: dict) -> str:
    system = (
        "You are a careful trade-analysis assistant. "
        "Answer only from the provided market and recommendation context. "
        "Do not invent prices, patterns, dates, probabilities, or sentiment values. "
        "If the signal is mixed or weak, say that clearly. "
        "Keep the answer practical and directly focused on the user's trading question."
    )
    prompt = (
        f"Question: {question}\n\n"
        "Trading context JSON:\n"
        f"{json.dumps(context, indent=2, default=str)}\n\n"
        "Answer the question using only this context."
    )
    return ask_llm(prompt, system=system)
