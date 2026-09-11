from __future__ import annotations

import time

import httpx

from app.models import LLMResponse, ModelConfig
from app.providers.base import Provider, ProviderError
from app.settings import get_settings


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.base_url = (base_url or settings.gemini_base_url).rstrip("/")

    async def complete(self, prompt: str, model: ModelConfig, timeout: float = 60.0) -> LLMResponse:
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is not set")
        started = time.perf_counter()
        url = f"{self.base_url}/models/{model.model_id}:generateContent"
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, params={"key": self.api_key}, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini request failed: {exc}") from exc

        text = _extract_text(data)
        usage = data.get("usageMetadata") or {}
        in_tok = int(usage.get("promptTokenCount") or _est_tokens(prompt))
        out_tok = int(usage.get("candidatesTokenCount") or _est_tokens(text))
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


def _extract_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    return "".join(p.get("text", "") for p in parts)


def _est_tokens(text: str) -> int:
    return max(1, len(text.split()))
