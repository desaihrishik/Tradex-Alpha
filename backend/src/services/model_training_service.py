from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.build_dataset import build_feature_frame
from src.core.config import get_settings
from src.core.supabase_client import get_supabase_admin_client
from src.data.repositories.market_data import LocalMarketDataRepository
from src.data.repositories.model_registry import ModelRegistryRepository


@dataclass(frozen=True)
class ModelTrainingResult:
    model_name: str
    version: str
    trained_at: str
    train_rows: int
    test_rows: int
    dataset_rows: int
    local_model_path: str
    local_metadata_path: str
    storage_model_path: str
    storage_metadata_path: str | None
    metrics: dict[str, object]


class ModelTrainingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.market_data = LocalMarketDataRepository()
        self.model_registry = ModelRegistryRepository()
        self.model_name = "nvda_rf_signal"
        self.model_path = self.settings.models_dir / f"{self.model_name}.pkl"
        self.metadata_path = self.settings.models_dir / f"{self.model_name}_metadata.json"

    def retrain_model(self, symbol: str = "NVDA") -> ModelTrainingResult:
        self.market_data.sync_recent_market_data(ticker=symbol)
        dataset = self._build_training_dataset(symbol=symbol)
        model, metadata = self._train(dataset)

        trained_at = datetime.now(timezone.utc).isoformat()
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        metadata["trained_at"] = trained_at
        metadata["version"] = version
        metadata["symbol"] = symbol

        local_model_path, local_metadata_path = self._write_local_artifacts(
            model=model,
            metadata=metadata,
            version=version,
        )
        storage_model_path, storage_metadata_path = self._publish_artifacts(
            model_path=Path(local_model_path),
            metadata_path=Path(local_metadata_path),
            version=version,
        )

        self.model_registry.archive_active_models(self.model_name)
        self.model_registry.register_model_version(
            model_name=self.model_name,
            version=version,
            storage_path=storage_model_path,
            metadata_path=storage_metadata_path,
            trained_at=trained_at,
            metrics=metadata["metrics"],
            status="active",
        )

        split_idx = int(metadata["split_idx"])
        return ModelTrainingResult(
            model_name=self.model_name,
            version=version,
            trained_at=trained_at,
            train_rows=split_idx,
            test_rows=len(dataset) - split_idx,
            dataset_rows=len(dataset),
            local_model_path=local_model_path,
            local_metadata_path=local_metadata_path,
            storage_model_path=storage_model_path,
            storage_metadata_path=storage_metadata_path,
            metrics=metadata["metrics"],
        )

    def _build_training_dataset(self, *, symbol: str) -> pd.DataFrame:
        raw_history = self.market_data.get_raw_market_history(ticker=symbol, limit=1500)
        featured = build_feature_frame(raw_history)

        future_return = featured["Close"].shift(-1) / featured["Close"] - 1.0
        featured["future_return_1d"] = future_return

        conditions = [
            future_return > 0.005,
            future_return < -0.005,
        ]
        choices = [1, -1]
        featured["signal"] = np.select(conditions, choices, default=0)

        dataset = featured.dropna().reset_index(drop=True)
        if len(dataset) < 300:
            raise RuntimeError(f"Not enough rows to retrain model safely for {symbol}. Found {len(dataset)} rows.")
        return dataset

    def _train(self, dataset: pd.DataFrame) -> tuple[RandomForestClassifier, dict[str, object]]:
        feature_cols = [
            column for column in dataset.columns
            if column not in ("Date", "future_return_1d", "signal")
        ]
        X = dataset[feature_cols].values
        y = dataset["signal"].values

        test_size = 252 if len(dataset) > 600 else max(int(0.2 * len(dataset)), 50)
        split_idx = len(dataset) - test_size

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, digits=3, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, y_pred).tolist()

        metadata = {
            "feature_cols": feature_cols,
            "label_mapping": {"-1": "SELL", "0": "HOLD", "1": "BUY"},
            "class_order": [int(value) for value in model.classes_],
            "model_path": self.model_path.name,
            "split_idx": split_idx,
            "metrics": {
                "classification_report": report,
                "confusion_matrix": matrix,
                "dataset_rows": len(dataset),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            },
        }
        return model, metadata

    def _write_local_artifacts(
        self,
        *,
        model: RandomForestClassifier,
        metadata: dict[str, object],
        version: str,
    ) -> tuple[str, str]:
        self.settings.models_dir.mkdir(exist_ok=True)

        versioned_model_path = self.settings.models_dir / f"{self.model_name}_{version}.pkl"
        versioned_metadata_path = self.settings.models_dir / f"{self.model_name}_{version}_metadata.json"

        with NamedTemporaryFile(delete=False, suffix=".pkl", dir=self.settings.models_dir) as temp_model:
            temp_model_path = Path(temp_model.name)
        dump(model, temp_model_path)
        temp_model_path.replace(self.model_path)
        dump(model, versioned_model_path)

        metadata_text = json.dumps(metadata, indent=2)
        with NamedTemporaryFile(delete=False, suffix=".json", dir=self.settings.models_dir, mode="w", encoding="utf-8") as temp_meta:
            temp_meta.write(metadata_text)
            temp_meta_path = Path(temp_meta.name)
        temp_meta_path.replace(self.metadata_path)
        versioned_metadata_path.write_text(metadata_text, encoding="utf-8")

        return str(versioned_model_path), str(versioned_metadata_path)

    def _publish_artifacts(self, *, model_path: Path, metadata_path: Path, version: str) -> tuple[str, str | None]:
        client = get_supabase_admin_client()
        storage_model_path = str(model_path)
        storage_metadata_path: str | None = str(metadata_path)
        if client is None:
            return storage_model_path, storage_metadata_path

        model_storage_key = f"{self.model_name}/{version}/{model_path.name}"
        metadata_storage_key = f"{self.model_name}/{version}/{metadata_path.name}"

        try:
            with model_path.open("rb") as handle:
                client.storage.from_(self.settings.supabase_models_bucket).upload(
                    model_storage_key,
                    handle.read(),
                    {"content-type": "application/octet-stream", "upsert": "true"},
                )
            storage_model_path = model_storage_key
        except Exception:
            storage_model_path = str(model_path)

        try:
            with metadata_path.open("rb") as handle:
                client.storage.from_(self.settings.supabase_model_metadata_bucket).upload(
                    metadata_storage_key,
                    handle.read(),
                    {"content-type": "application/json", "upsert": "true"},
                )
            storage_metadata_path = metadata_storage_key
        except Exception:
            storage_metadata_path = str(metadata_path)

        return storage_model_path, storage_metadata_path
