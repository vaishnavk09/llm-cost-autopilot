from __future__ import annotations

from app.models import LLMResponse, ModelConfig
from app.providers.base import Provider, ProviderError
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.huggingface import HuggingFaceProvider
from app.providers.mock import MockProvider
from app.providers.ollama import OllamaProvider
from app.settings import get_settings

_PROVIDERS: dict[str, Provider] | None = None


def get_providers() -> dict[str, Provider]:
    global _PROVIDERS
    if _PROVIDERS is None:
        _PROVIDERS = {
            "groq": GroqProvider(),
            "gemini": GeminiProvider(),
            "huggingface": HuggingFaceProvider(),
            "ollama": OllamaProvider(),
            "mock": MockProvider(),
        }
    return _PROVIDERS


def reset_providers() -> None:
    global _PROVIDERS
    _PROVIDERS = None


async def send_request(prompt: str, model_config: ModelConfig, timeout: float = 60.0) -> LLMResponse:
    """Unified entry point for every registered provider."""
    settings = get_settings()
    providers = get_providers()
    if settings.mock_llm:
        return await providers["mock"].complete(prompt, model_config, timeout=timeout)
    provider = providers.get(model_config.provider)
    if provider is None:
        raise ProviderError(f"No adapter for provider {model_config.provider}")
    return await provider.complete(prompt, model_config, timeout=timeout)


async def provider_health() -> dict[str, bool]:
    settings = get_settings()
    if settings.mock_llm:
        return {"mock": True, "groq": False, "gemini": False, "huggingface": False, "ollama": False}
    out: dict[str, bool] = {}
    for name, provider in get_providers().items():
        out[name] = await provider.healthy()
    return out
