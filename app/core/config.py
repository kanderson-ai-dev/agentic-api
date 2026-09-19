"""Application configuration powered by pydantic-settings."""

import os
from functools import lru_cache

from pydantic import SecretStr
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
