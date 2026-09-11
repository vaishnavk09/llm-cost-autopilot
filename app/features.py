from __future__ import annotations

import re
from typing import Any

INSTRUCTION_WORDS = (
    "analyze",
    "compare",
    "evaluate",
    "reason",
    "critique",
    "justify",
    "debate",
    "prove",
    "design",
    "synthesize",
    "trade-off",
    "nuanced",
    "step by step",
    "multi-step",
)
MODERATE_WORDS = (
    "summarize",
    "summary",
    "classify",
    "categorize",
    "outline",
    "extract",
    "label",
    "sentiment",
    "structured",
)
FORMAT_WORDS = ("json", "yaml", "xml", "table", "markdown", "csv", "schema")
CONSTRAINT_PATTERNS = (
    r"\bmust\b",
    r"\bshould\b",
    r"\bdo not\b",
    r"\bdon't\b",
    r"\balways\b",
    r"\bnever\b",
    r"\brequire",
    r"\bconstraint",
)
CONTEXT_MARKERS = (
    "context:",
    "passage:",
    "document:",
    "transcript:",
    "article:",
    "here is the text",
    "given the following",
)

FEATURE_NAMES = [
    "token_count",
    "char_count",
    "instruction_hits",
    "moderate_hits",
    "constraint_count",
    "has_context",
    "format_complexity",
    "question_marks",
    "line_count",
    "avg_word_len",
]


def extract_features(prompt: str) -> dict[str, float]:
    text = prompt.lower()
    words = re.findall(r"[a-z0-9']+", text)
    token_count = float(len(words))
    instruction_hits = float(sum(1 for w in INSTRUCTION_WORDS if w in text))
    moderate_hits = float(sum(1 for w in MODERATE_WORDS if w in text))
    constraint_count = float(sum(len(re.findall(p, text)) for p in CONSTRAINT_PATTERNS))
    has_context = 1.0 if any(m in text for m in CONTEXT_MARKERS) or token_count > 180 else 0.0
    format_complexity = float(sum(1 for w in FORMAT_WORDS if w in text))
    if "json" in text and ("schema" in text or "fields" in text):
        format_complexity += 1.0
    question_marks = float(text.count("?"))
    line_count = float(max(1, prompt.count("\n") + 1))
    avg_word_len = (sum(len(w) for w in words) / token_count) if token_count else 0.0
    return {
        "token_count": token_count,
        "char_count": float(len(prompt)),
        "instruction_hits": instruction_hits,
        "moderate_hits": moderate_hits,
        "constraint_count": constraint_count,
        "has_context": has_context,
        "format_complexity": format_complexity,
        "question_marks": question_marks,
        "line_count": line_count,
        "avg_word_len": avg_word_len,
    }


def feature_vector(prompt: str) -> list[float]:
    feats = extract_features(prompt)
    return [feats[name] for name in FEATURE_NAMES]


def infer_use_case(prompt: str) -> str:
    text = prompt.lower()
    if any(w in text for w in ("extract", "json", "fields", "key-value")):
        return "extraction"
    if any(w in text for w in ("summarize", "summary", "tl;dr")):
        return "summarization"
    if any(w in text for w in ("classify", "label", "category", "sentiment")):
        return "classification"
    return "general"


def features_as_json(prompt: str) -> dict[str, Any]:
    return extract_features(prompt)
