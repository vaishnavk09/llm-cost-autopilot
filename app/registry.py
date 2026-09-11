from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.models import ModelConfig
from app.settings import abs_path, get_settings


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_models(config_path: str | None = None) -> list[ModelConfig]:
    settings = get_settings()
    path = abs_path(config_path or settings.models_config)
    data = _load_yaml(path)
    models: list[ModelConfig] = []
    for row in data.get("models", []):
        models.append(
            ModelConfig(
                id=row["id"],
                provider=row["provider"],
                model_id=row["model_id"],
                cost_per_input_token=float(row["cost_per_input_token"]),
                cost_per_output_token=float(row["cost_per_output_token"]),
                average_latency_ms=float(row["average_latency_ms"]),
                quality_tier=row["quality_tier"],
                enabled=bool(row.get("enabled", True)),
            )
        )
    return models


def load_routing(config_path: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    path = abs_path(config_path or settings.routing_config)
    return _load_yaml(path)


@lru_cache(maxsize=1)
def model_index() -> dict[str, ModelConfig]:
    return {m.id: m for m in load_models() if m.enabled}


def get_model(model_id: str) -> ModelConfig:
    idx = model_index()
    if model_id not in idx:
        raise KeyError(f"Unknown or disabled model: {model_id}")
    return idx[model_id]


def reload_registry() -> None:
    model_index.cache_clear()


def save_routing(data: dict[str, Any], config_path: str | None = None) -> None:
    settings = get_settings()
    path = abs_path(config_path or settings.routing_config)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    reload_registry()
