from __future__ import annotations

import time

import httpx

from app.models import LLMResponse, ModelConfig
from app.providers.base import Provider, ProviderError
from app.settings import get_settings


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, host: str | None = None) -> None:
        settings = get_settings()
        self.host = (host or settings.ollama_host).rstrip("/")

    async def complete(self, prompt: str, model: ModelConfig, timeout: float = 120.0) -> LLMResponse:
        started = time.perf_counter()
        payload = {
            "model": model.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self.host}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        text = ((data.get("message") or {}).get("content")) or data.get("response") or ""
        in_tok = int(data.get("prompt_eval_count") or _est_tokens(prompt))
        out_tok = int(data.get("eval_count") or _est_tokens(text))
        latency = (time.perf_counter() - started) * 1000
        return LLMResponse(
            text=text.strip(),
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency,
            cost=model.shadow_cost(in_tok, out_tok),
            model_id=model.model_id,
            provider=self.name,
            registry_id=model.id,
            raw=data,
        )

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.host}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False


def _est_tokens(text: str) -> int:
    return max(1, len(text.split()))
