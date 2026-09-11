from __future__ import annotations

import json
import re
from pathlib import Path

from app.client import send_request
from app.models import LLMResponse, ModelConfig
from app.registry import load_routing
from app.settings import ROOT, get_settings


def token_overlap(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-z0-9]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def parse_judge_score(text: str) -> float | None:
    match = re.search(r"([1-5](?:\.\d)?)\s*/\s*5", text)
    if match:
        return float(match.group(1))
    match = re.search(r"\b([1-5](?:\.\d)?)\b", text)
    if match:
        return float(match.group(1))
    return None


async def llm_judge(prompt: str, candidate: str, reference: str, judge_model: ModelConfig) -> float:
    rubric = (
        "Score the CANDIDATE answer against the REFERENCE for the USER PROMPT. "
        "Reply with a single number 1-5 and one sentence. 5 = equivalent quality.\n\n"
        f"USER PROMPT:\n{prompt[:4000]}\n\nREFERENCE:\n{reference[:4000]}\n\nCANDIDATE:\n{candidate[:4000]}"
    )
    try:
        judged = await send_request(rubric, judge_model, timeout=45.0)
        parsed = parse_judge_score(judged.text)
        if parsed is not None:
            return parsed / 5.0
    except Exception:
        pass
    return token_overlap(candidate, reference)


def score_outputs(use_case: str, prompt: str, cheap: str, judge: str) -> float:
    overlap = token_overlap(cheap, judge)
    if use_case == "classification":
        cheap_lab = cheap.strip().splitlines()[0][:80].lower()
        judge_lab = judge.strip().splitlines()[0][:80].lower()
        return 1.0 if cheap_lab == judge_lab or cheap_lab in judge_lab or judge_lab in cheap_lab else overlap
    if use_case == "extraction":
        return overlap
    return overlap


async def verify(
    prompt: str,
    use_case: str,
    cheap_response: LLMResponse,
    judge_model: ModelConfig,
) -> dict:
    routing = load_routing()
    quality_cfg = routing.get("quality") or {}
    use_cfg = (routing.get("use_cases") or {}).get(use_case) or {}
    threshold = float(use_cfg.get("fail_threshold") or quality_cfg.get("fail_threshold") or get_settings().quality_fail_threshold)

    judge_resp = await send_request(prompt, judge_model, timeout=60.0)
    agreement = score_outputs(use_case, prompt, cheap_response.text, judge_resp.text)
    if use_case == "summarization":
        judged = await llm_judge(prompt, cheap_response.text, judge_resp.text, judge_model)
        agreement = 0.4 * agreement + 0.6 * judged

    failed = agreement < threshold
    return {
        "quality_score": agreement,
        "threshold": threshold,
        "failed": failed,
        "judge_text": judge_resp.text,
        "judge_model": judge_model.id,
        "judge_cost": judge_resp.cost,
        "quality_gap": max(0.0, threshold - agreement),
    }


def append_feedback(prompt: str, true_tier: int, path: Path | None = None) -> None:
    dest = path or (ROOT / "data" / "feedback.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "tier": true_tier, "source": "routing_failure"}) + "\n")
