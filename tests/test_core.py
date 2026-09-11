from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.client import send_request
from app.features import extract_features, feature_vector
from app.models import ModelConfig
from app.providers.groq import GroqProvider
from app.registry import load_models, load_routing
from app.verifier import token_overlap


@pytest.fixture
def mock_model() -> ModelConfig:
    return ModelConfig(
        id="mock-local",
        provider="mock",
        model_id="mock-v1",
        cost_per_input_token=0.01,
        cost_per_output_token=0.01,
        average_latency_ms=5,
        quality_tier="low",
    )


@pytest.mark.asyncio
async def test_send_request_mock(mock_model: ModelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM", "1")
    from app.settings import Settings

    monkeypatch.setattr("app.client.get_settings", lambda: Settings(mock_llm=True))
    resp = await send_request("hello world", mock_model)
    assert resp.provider == "mock"
    assert resp.input_tokens >= 1
    assert resp.cost == mock_model.shadow_cost(resp.input_tokens, resp.output_tokens)
    assert "mock" in resp.text


@pytest.mark.asyncio
async def test_groq_provider_http(respx_mock, mock_model: ModelConfig) -> None:
    model = ModelConfig(**{**mock_model.__dict__, "provider": "groq", "model_id": "llama-3.1-8b-instant", "id": "groq-llama-8b"})
    provider = GroqProvider(api_key="test-key", base_url="https://api.groq.com/openai/v1")
    respx_mock.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=__import__("httpx").Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )
    )
    resp = await provider.complete("ping", model)
    assert resp.text == "ok"
    assert resp.input_tokens == 4
    assert resp.cost == model.shadow_cost(4, 1)


def test_registry_has_expected_models() -> None:
    ids = {m.id for m in load_models()}
    assert "groq-llama-8b" in ids
    assert "groq-llama-70b" in ids
    assert "gemini-flash-lite" in ids
    routing = load_routing()
    assert 1 in routing["tiers"] or "1" in routing["tiers"]


def test_features_detect_complexity() -> None:
    simple = extract_features("Reformat this list as CSV: a b c")
    complex_ = extract_features(
        "Analyze the trade-offs and compare two designs. You must justify a nuanced multi-step recommendation."
    )
    assert complex_["instruction_hits"] > simple["instruction_hits"]
    assert len(feature_vector("hello")) == 10


def test_token_overlap() -> None:
    assert token_overlap("alpha beta", "alpha beta") == 1.0
    assert token_overlap("alpha", "omega") == 0.0


def test_api_completions_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    from app import db, pipeline, settings
    from app.classifier import ComplexityClassifier, train_from_rows
    from app.settings import Settings

    prompts = ["reformat list a b c"] * 10 + ["summarize this passage"] * 10 + [
        "analyze and compare trade-offs step by step you must justify"
    ] * 10
    labels = [1] * 10 + [2] * 10 + [3] * 10
    clf, _ = train_from_rows(prompts, labels, test_size=0.3)
    model_path = tmp_path / "clf.pkl"
    clf.save(model_path)

    s = Settings(mock_llm=True, database_path=str(tmp_path / "t.db"), classifier_path=str(model_path))
    monkeypatch.setattr(settings, "get_settings", lambda: s)
    monkeypatch.setattr(db, "get_settings", lambda: s)
    monkeypatch.setattr(pipeline, "get_settings", lambda: s)
    monkeypatch.setattr("app.client.get_settings", lambda: s)
    monkeypatch.setattr("app.router.get_settings", lambda: s)
    pipeline.reload_classifier()
    db.init_db()

    from app.api import app

    client = TestClient(app)
    res = client.post(
        "/v1/completions",
        json={"messages": [{"role": "user", "content": "Reformat this list: a b c"}], "verify": False},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["choices"][0]["message"]["content"]
    assert "complexity_tier" in body["routing"]
    models = client.get("/v1/models")
    assert models.status_code == 200
    stats = client.get("/v1/stats")
    assert stats.status_code == 200
    assert stats.json()["total_requests"] >= 1
