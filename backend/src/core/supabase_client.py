from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.core.config import get_settings

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover
    Client = Any  # type: ignore[misc,assignment]
    create_client = None


@lru_cache(maxsize=1)
def get_supabase_admin_client() -> Client | None:
    settings = get_settings()
    if create_client is None:
        return None
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return None
    return create_client(settings.supabase_url, settings.supabase_service_role_key)

