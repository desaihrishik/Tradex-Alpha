from __future__ import annotations

from typing import Any

from src.core.supabase_client import get_supabase_admin_client


class AdminStatusRepository:
    def upsert_runtime_status(
        self,
        *,
        service_name: str,
        symbol: str,
        status: dict[str, Any],
    ) -> None:
        client = get_supabase_admin_client()
        if client is None:
            return

        payload = {
            "service_name": service_name,
            "symbol": symbol,
            "status": status,
        }
        client.table("admin_runtime_status").upsert(
            payload,
            on_conflict="service_name,symbol",
        ).execute()

    def insert_event(
        self,
        *,
        service_name: str,
        symbol: str,
        event_type: str,
        status: dict[str, Any],
    ) -> None:
        client = get_supabase_admin_client()
        if client is None:
            return

        payload = {
            "service_name": service_name,
            "symbol": symbol,
            "event_type": event_type,
            "status": status,
        }
        client.table("admin_status_events").insert(payload).execute()

    def get_runtime_status(
        self,
        *,
        service_name: str,
        symbol: str = "NVDA",
    ) -> dict[str, Any] | None:
        client = get_supabase_admin_client()
        if client is None:
            return None

        result = (
            client.table("admin_runtime_status")
            .select("service_name, symbol, status, updated_at")
            .eq("service_name", service_name)
            .eq("symbol", symbol)
            .limit(1)
            .execute()
        )
        if not result.data:
            return None

        row = result.data[0]
        return {
            "service_name": row.get("service_name"),
            "symbol": row.get("symbol"),
            "updated_at": row.get("updated_at"),
            "status": row.get("status") or {},
        }

    def get_recent_events(
        self,
        *,
        service_name: str,
        symbol: str = "NVDA",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        client = get_supabase_admin_client()
        if client is None:
            return []

        result = (
            client.table("admin_status_events")
            .select("event_type, status, created_at")
            .eq("service_name", service_name)
            .eq("symbol", symbol)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return list(result.data or [])
