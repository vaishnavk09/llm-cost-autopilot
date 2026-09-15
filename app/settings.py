from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    gemini_api_key: str = ""
    hf_token: str = ""
    ollama_host: str = "http://localhost:11434"
    mock_llm: bool = False
    database_path: str = "data/app.db"
    classifier_path: str = "models/complexity_clf.pkl"
    routing_config: str = "config/routing.yaml"
    models_config: str = "config/models.yaml"
    max_escalation_ms: int = 8000
    quality_fail_threshold: float = 0.62
    log_level: str = "INFO"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    hf_base_url: str = "https://router.huggingface.co/v1"


def get_settings() -> Settings:
    return Settings()


def abs_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return ROOT / path
