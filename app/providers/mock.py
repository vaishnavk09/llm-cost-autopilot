from __future__ import annotations

import hashlib
import time

from app.models import LLMResponse, ModelConfig
from app.providers.base import Provider


class MockProvider(Provider):
    """Deterministic offline provider for tests and demos without API keys."""

    name = "mock"

    async def complete(self, prompt: str, model: ModelConfig, timeout: float = 60.0) -> LLMResponse:
        started = time.perf_counter()
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        text = (
            f"[mock:{model.model_id}] {prompt[:180].strip()}\n"
            f"Summary: handled locally. id={digest}"
        )
        in_tok = max(1, len(prompt.split()))
        out_tok = max(1, len(text.split()))
        latency = (time.perf_counter() - started) * 1000
        return LLMResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=max(latency, 1.0),
            cost=model.shadow_cost(in_tok, out_tok),
            model_id=model.model_id,
            provider=self.name,
            registry_id=model.id,
        )

    async def healthy(self) -> bool:
        return True
