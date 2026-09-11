from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Literal


QualityTier = Literal["high", "medium", "low"]
ProviderName = Literal["groq", "gemini", "ollama", "mock"]


class ComplexityTier(IntEnum):
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3


@dataclass(frozen=True)
class ModelConfig:
    id: str
    provider: ProviderName
    model_id: str
    cost_per_input_token: float
    cost_per_output_token: float
    average_latency_ms: float
    quality_tier: QualityTier
    enabled: bool = True

    def shadow_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * self.cost_per_input_token + (
            output_tokens / 1_000_000
        ) * self.cost_per_output_token


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float
    model_id: str
    provider: str
    registry_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        return d


@dataclass
class RoutingDecision:
    complexity_tier: int
    model: ModelConfig
    reason: str
    features: dict[str, float]
    candidates_tried: list[str] = field(default_factory=list)
