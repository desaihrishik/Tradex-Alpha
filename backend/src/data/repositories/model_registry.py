from __future__ import annotations

from dataclasses import dataclass

from src.core.supabase_client import get_supabase_admin_client


@dataclass(frozen=True)
class ModelArtifactReference:
    model_name: str
    version: str
    storage_path: str
    metadata_path: str | None = None


class ModelRegistryRepository:
    def archive_active_models(self, model_name: str = "nvda_rf_signal") -> None:
        client = get_supabase_admin_client()
        if client is None:
            return

        (
            client.table("model_registry")
            .update({"status": "archived"})
            .eq("model_name", model_name)
            .eq("status", "active")
            .execute()
        )

    def register_model_version(
        self,
        *,
        model_name: str,
        version: str,
        storage_path: str,
        metadata_path: str | None,
        trained_at: str,
        metrics: dict[str, object],
        status: str = "active",
    ) -> None:
        client = get_supabase_admin_client()
        if client is None:
            return

        payload = {
            "model_name": model_name,
            "version": version,
            "storage_path": storage_path,
            "metadata_path": metadata_path,
            "framework": "sklearn",
            "status": status,
            "trained_at": trained_at,
            "metrics": metrics,
        }
        client.table("model_registry").insert(payload).execute()

    def get_active_model(self, model_name: str = "nvda_rf_signal") -> ModelArtifactReference | None:
        client = get_supabase_admin_client()
        if client is None:
            return None

        result = (
            client.table("model_registry")
            .select("model_name, version, storage_path, metadata_path")
            .eq("model_name", model_name)
            .eq("status", "active")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not result.data:
            return None

        row = result.data[0]
        return ModelArtifactReference(
            model_name=row["model_name"],
            version=row["version"],
            storage_path=row["storage_path"],
            metadata_path=row.get("metadata_path"),
        )
