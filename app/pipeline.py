from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.classifier import ComplexityClassifier
from app.db import add_event, insert_request, update_request
from app.features import extract_features, infer_use_case
from app.models import LLMResponse, RoutingDecision
from app.router import (
    baseline_model,
    complete_with_failover,
    hypothetical_gpt4o_cost,
    pick_judge,
    route,
)
from app.settings import abs_path, get_settings
from app.verifier import append_feedback, verify

_classifier: ComplexityClassifier | None = None


def get_classifier() -> ComplexityClassifier:
    global _classifier
    if _classifier is None:
        path = abs_path(get_settings().classifier_path)
        if path.exists():
            _classifier = ComplexityClassifier.load(path)
        else:
            raise FileNotFoundError(
                f"Classifier missing at {path}. Run: python scripts/train_classifier.py"
            )
    return _classifier


def reload_classifier() -> None:
    global _classifier
    _classifier = None


def messages_to_prompt(messages: list[dict]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts).strip()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


async def handle_completion(
    messages: list[dict],
    use_case: str | None = None,
    verify_async: bool = True,
) -> dict:
    prompt = messages_to_prompt(messages)
    clf = get_classifier()
    tier = clf.predict(prompt)
    features = extract_features(prompt)
    use_case = use_case or infer_use_case(prompt)
    decision = await route(prompt, tier, features)
    response, decision, errors = await complete_with_failover(prompt, decision)

    original_model = decision.model.id
    escalated = False
    original_cost = response.cost
    quality_meta: dict = {}

    settings = get_settings()
    remaining = settings.max_escalation_ms - response.latency_ms
    if remaining > 500:
        quality_meta = await _maybe_escalate(
            prompt, use_case, response, decision, remaining
        )
        if quality_meta.get("escalated"):
            escalated = True
            response = quality_meta["response"]
            decision = quality_meta["decision"]

    premium = baseline_model()
    baseline_cost = premium.shadow_cost(response.input_tokens, response.output_tokens)
    gpt4o_cost = hypothetical_gpt4o_cost(response.input_tokens, response.output_tokens)

    metadata = {
        "reason": decision.reason,
        "features": features,
        "candidates_tried": decision.candidates_tried,
        "errors": errors,
        "escalated": escalated,
        "original_model": original_model,
    }
    row_id = insert_request(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt_hash": prompt_hash(prompt),
            "prompt": prompt,
            "use_case": use_case,
            "complexity_tier": int(decision.complexity_tier),
            "routed_model": decision.model.id,
            "provider": decision.model.provider,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "cost": response.cost,
            "baseline_cost": baseline_cost,
            "gpt4o_cost": gpt4o_cost,
            "quality_score": quality_meta.get("quality_score"),
            "escalated": int(escalated),
            "original_model": original_model,
            "escalated_model": decision.model.id if escalated else None,
            "cost_delta": (response.cost - original_cost) if escalated else 0.0,
            "quality_gap": quality_meta.get("quality_gap"),
            "verify_status": "done" if quality_meta.get("verified") else "pending",
            "response_text": response.text,
            "judge_text": quality_meta.get("judge_text"),
            "metadata_json": json.dumps(metadata),
        }
    )
    add_event(row_id, "routed", {"tier": tier, "model": decision.model.id, "reason": decision.reason})
    if escalated:
        add_event(
            row_id,
            "escalated",
            {
                "original_model": original_model,
                "escalated_model": decision.model.id,
                "cost_delta": response.cost - original_cost,
                "quality_gap": quality_meta.get("quality_gap"),
            },
        )

    return {
        "id": row_id,
        "text": response.text,
        "model": decision.model.id,
        "provider": decision.model.provider,
        "complexity_tier": int(decision.complexity_tier),
        "reason": decision.reason,
        "cost": response.cost,
        "baseline_cost": baseline_cost,
        "gpt4o_cost": gpt4o_cost,
        "latency_ms": response.latency_ms,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "escalated": escalated,
        "quality_score": quality_meta.get("quality_score"),
        "prompt": prompt,
        "use_case": use_case,
        "verify_async": verify_async and not quality_meta.get("verified"),
        "usage": {
            "prompt_tokens": response.input_tokens,
            "completion_tokens": response.output_tokens,
            "total_tokens": response.input_tokens + response.output_tokens,
        },
    }


async def _maybe_escalate(
    prompt: str,
    use_case: str,
    response: LLMResponse,
    decision: RoutingDecision,
    remaining_ms: float,
) -> dict:
    if decision.model.quality_tier == "high":
        return {}
    judge = await pick_judge()
    if judge.id == decision.model.id:
        return {}
    result = await verify(prompt, use_case, response, judge)
    result["verified"] = True
    if not result["failed"]:
        return result
    try:
        better = await send_request(prompt, judge, timeout=max(5.0, remaining_ms / 1000))
    except Exception:
        append_feedback(prompt, 3)
        result["escalated"] = False
        return result
    new_decision = RoutingDecision(
        complexity_tier=3,
        model=judge,
        reason=decision.reason + f" | auto-escalated to {judge.id} after quality {result['quality_score']:.2f}",
        features=decision.features,
        candidates_tried=decision.candidates_tried + [judge.id],
    )
    append_feedback(prompt, 3)
    result.update({"escalated": True, "response": better, "decision": new_decision})
    return result


async def run_verification_job(row: dict) -> None:
    from app.models import LLMResponse as Resp

    if row.get("verify_status") not in (None, "pending", "verifying"):
        return
    judge = await pick_judge()
    cheap = Resp(
        text=row["response_text"] or "",
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        latency_ms=row["latency_ms"],
        cost=row["cost"],
        model_id=row["routed_model"],
        provider=row["provider"],
        registry_id=row["routed_model"],
    )
    result = await verify(row["prompt"], row["use_case"], cheap, judge)
    fields = {
        "quality_score": result["quality_score"],
        "quality_gap": result["quality_gap"],
        "judge_text": result["judge_text"],
        "verify_status": "failed" if result["failed"] else "ok",
    }
    update_request(row["id"], **fields)
    if result["failed"]:
        append_feedback(row["prompt"], 3)
        add_event(row["id"], "routing_failure", result)
    else:
        add_event(row["id"], "verified", {"quality_score": result["quality_score"]})
