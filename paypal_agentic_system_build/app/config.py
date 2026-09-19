"""Central configuration. Credentials are read here and nowhere else."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Source of API definitions and generated registry.
    collection_path: Path = PROJECT_ROOT / "data" / "PayPal APIs.postman_collection.json"
    registry_path: Path = PROJECT_ROOT / "data" / "tool_registry.json"
    embedding_cache_path: Path = PROJECT_ROOT / "data" / "tool_embeddings.npz"

    # Runtime API configuration. Never reaches the registry or a prompt.
    paypal_base_url: str = "https://api-m.sandbox.paypal.com"
    paypal_client_id: str = ""
    paypal_client_secret: str = ""

    # Reasoning model (Groq) and local embedding model.
    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    fast_llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.0
    llm_max_retries: int = 2
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def llm_available(self) -> bool:
        return bool(self.groq_api_key)

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
