from __future__ import annotations

import time

import httpx

from app.models import LLMResponse, ModelConfig
from app.providers.base import Provider, ProviderError
from app.settings import get_settings


class HuggingFaceProvider(Provider):
    """OpenAI-compatible Hugging Face Inference Router (free token, open-weight models)."""

    name = "huggingface"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.hf_token
        self.base_url = (base_url or settings.hf_base_url).rstrip("/")

    async def complete(self, prompt: str, model: ModelConfig, timeout: float = 60.0) -> LLMResponse:
        if not self.api_key:
            raise ProviderError("HF_TOKEN is not set")
        started = time.perf_counter()
        payload = {
            "model": model.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Hugging Face request failed: {exc}") from exc

        choice = (data.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content")) or ""
        usage = data.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or _est_tokens(prompt))
        out_tok = int(usage.get("completion_tokens") or _est_tokens(text))
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
        return bool(self.api_key)


def _est_tokens(text: str) -> int:
    return max(1, len(text.split()))
