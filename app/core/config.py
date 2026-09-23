"""Application configuration powered by pydantic-settings."""

import os
from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "agentic-api"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # OpenAI
    openai_api_key: SecretStr | None = None

    # LangSmith / LangChain tracing
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: SecretStr | None = None
    langchain_project: str = "agentic-api"

    # Conversation memory (LangGraph checkpointer)
    checkpoint_db_path: str = "data/checkpoints.sqlite"

    # API authentication (optional; enforced only when configured)
    agentic_api_key: SecretStr | None = None

    # Rate limiting (per client identity; 0 disables)
    rate_limit_per_minute: int = 60

    @field_validator(
        "openai_api_key", "langchain_api_key", "agentic_api_key", mode="before"
    )
    @classmethod
    def _blank_secret_to_none(cls, value: object) -> object:
        """Treat an empty-string secret (e.g. an unset GitHub Actions secret) as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


def configure_langchain_environment(settings: Settings) -> None:
    """Expose LangChain/LangSmith tracing settings as process environment variables.

    The `langchain-core` tracing machinery reads these values directly from
    `os.environ`, so they must be propagated there once at startup.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    if settings.langchain_api_key is not None:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key.get_secret_value()
