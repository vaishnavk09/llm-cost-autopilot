from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import __version__
from app.client import provider_health
from app.db import fetch_stats, init_db
from app.pipeline import handle_completion, run_verification_job
from app.registry import load_models, load_routing, reload_registry, save_routing
from app.settings import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="LLM Cost Autopilot",
    version=__version__,
    description="Routes each prompt to the cheapest capable free/open-source model.",
    lifespan=lifespan,
)


class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class CompletionRequest(BaseModel):
    messages: list[ChatMessage]
    use_case: str | None = None
    verify: bool = True


class RoutingConfigUpdate(BaseModel):
    config: dict[str, Any] = Field(..., description="Full routing.yaml object")


async def _verify_later(request_id: int, payload: dict) -> None:
    payload = {**payload, "id": request_id, "verify_status": "pending"}
    await run_verification_job(payload)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "providers": await provider_health(), "mock": get_settings().mock_llm}


@app.post("/v1/completions")
async def completions(body: CompletionRequest, background: BackgroundTasks) -> dict:
    if not body.messages:
        raise HTTPException(400, "messages is required")
    try:
        result = await handle_completion(
            [m.model_dump() for m in body.messages],
            use_case=body.use_case,
            verify_async=body.verify,
        )
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc

    if body.verify and result.get("verify_async"):
        background.add_task(
            _verify_later,
            result["id"],
            {
                "prompt": result["prompt"],
                "use_case": result["use_case"],
                "response_text": result["text"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "latency_ms": result["latency_ms"],
                "cost": result["cost"],
                "routed_model": result["model"],
                "provider": result["provider"],
            },
        )

    return {
        "id": result["id"],
        "object": "chat.completion",
        "model": result["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }
        ],
        "usage": result["usage"],
        "routing": {
            "complexity_tier": result["complexity_tier"],
            "reason": result["reason"],
            "provider": result["provider"],
            "cost": result["cost"],
            "baseline_cost": result["baseline_cost"],
            "gpt4o_cost": result["gpt4o_cost"],
            "latency_ms": result["latency_ms"],
            "escalated": result["escalated"],
            "quality_score": result["quality_score"],
        },
    }


@app.get("/v1/models")
async def list_models() -> dict:
    models = [
        {
            "id": m.id,
            "provider": m.provider,
            "model_id": m.model_id,
            "quality_tier": m.quality_tier,
            "cost_per_input_token": m.cost_per_input_token,
            "cost_per_output_token": m.cost_per_output_token,
            "average_latency_ms": m.average_latency_ms,
        }
        for m in load_models()
        if m.enabled
    ]
    health = await provider_health()
    return {"models": models, "provider_health": health}


@app.get("/v1/stats")
async def stats() -> dict:
    return fetch_stats()


@app.get("/v1/routing-config")
async def get_routing() -> dict:
    return load_routing()


@app.put("/v1/routing-config")
async def put_routing(body: RoutingConfigUpdate) -> dict:
    save_routing(body.config)
    reload_registry()
    return {"ok": True, "config": load_routing()}
