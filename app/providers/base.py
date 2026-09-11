from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import LLMResponse, ModelConfig


class ProviderError(RuntimeError):
    pass


class Provider(ABC):
    name: str

    @abstractmethod
    async def complete(self, prompt: str, model: ModelConfig, timeout: float = 60.0) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def healthy(self) -> bool:
        raise NotImplementedError
