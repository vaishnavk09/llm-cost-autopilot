from __future__ import annotations

from typing import Any

from app.client import provider_health, send_request
from app.models import ModelConfig, RoutingDecision
from app.providers.base import ProviderError
from app.registry import get_model, load_routing, model_index
from app.settings import get_settings


async def healthy_providers() -> dict[str, bool]:
    return await provider_health()


def _eligible(model: ModelConfig, health: dict[str, bool], mock: bool) -> bool:
    if mock:
        return True
    if model.provider == "mock":
        # Only use mock when no live backend is healthy.
        live = any(health.get(p) for p in ("groq", "gemini", "ollama"))
        return not live
    return bool(health.get(model.provider))


async def pick_model(tier: int, routing: dict[str, Any] | None = None) -> tuple[ModelConfig, str, list[str]]:
    settings = get_settings()
    routing = routing or load_routing()
    health = await healthy_providers()
    tried: list[str] = []
    preferred = (routing.get("tiers") or {}).get(tier) or (routing.get("tiers") or {}).get(str(tier)) or []
    for model_id in preferred:
        tried.append(model_id)
        try:
            model = get_model(model_id)
        except KeyError:
            continue
        if _eligible(model, health, settings.mock_llm):
            reason = (
                f"Tier {tier} mapped to {model.id} ({model.provider}/{model.model_id}); "
                f"quality_tier={model.quality_tier}"
            )
            return model, reason, tried

    # Last-resort: cheapest healthy model, then any mock.
    models = sorted(model_index().values(), key=lambda m: m.cost_per_input_token + m.cost_per_output_token)
    for model in models:
        tried.append(model.id)
        if _eligible(model, health, settings.mock_llm):
            return model, f"Fallback to {model.id} (no healthy preferred model for tier {tier})", tried
    raise ProviderError(f"No healthy model available for tier {tier}. Tried: {tried}")


async def pick_judge(routing: dict[str, Any] | None = None) -> ModelConfig:
    routing = routing or load_routing()
    settings = get_settings()
    health = await healthy_providers()
    for model_id in routing.get("judge_candidates") or []:
        try:
            model = get_model(model_id)
        except KeyError:
            continue
        if _eligible(model, health, settings.mock_llm):
            return model
    model, _, _ = await pick_model(3, routing)
    return model


def baseline_model(routing: dict[str, Any] | None = None) -> ModelConfig:
    routing = routing or load_routing()
    try:
        return get_model(routing.get("baseline_model") or "groq-llama-70b")
    except KeyError:
        models = list(model_index().values())
        return max(models, key=lambda m: m.cost_per_input_token + m.cost_per_output_token)


def hypothetical_gpt4o_cost(input_tokens: int, output_tokens: int, routing: dict[str, Any] | None = None) -> float:
    routing = routing or load_routing()
    prices = routing.get("hypothetical_gpt4o") or {}
    inp = float(prices.get("cost_per_input_token", 2.50))
    out = float(prices.get("cost_per_output_token", 10.00))
    return (input_tokens / 1_000_000) * inp + (output_tokens / 1_000_000) * out


async def route(prompt: str, tier: int, features: dict[str, float]) -> RoutingDecision:
    model, reason, tried = await pick_model(tier)
    return RoutingDecision(
        complexity_tier=tier,
        model=model,
        reason=reason,
        features=features,
        candidates_tried=tried,
    )


async def complete_with_failover(prompt: str, decision: RoutingDecision) -> tuple:
    errors: list[str] = []
    try:
        response = await send_request(prompt, decision.model)
        return response, decision, errors
    except ProviderError as exc:
        errors.append(f"{decision.model.id}: {exc}")

    routing = load_routing()
    preferred = (routing.get("tiers") or {}).get(decision.complexity_tier) or []
    health = await healthy_providers()
    settings = get_settings()
    for model_id in preferred:
        if model_id == decision.model.id:
            continue
        try:
            model = get_model(model_id)
        except KeyError:
            continue
        if not _eligible(model, health, settings.mock_llm):
            continue
        try:
            response = await send_request(prompt, model)
            decision.model = model
            decision.reason += f" | failed over to {model.id}"
            decision.candidates_tried.append(model.id)
            return response, decision, errors
        except ProviderError as exc:
            errors.append(f"{model.id}: {exc}")
    raise ProviderError("; ".join(errors) or "All candidates failed")
