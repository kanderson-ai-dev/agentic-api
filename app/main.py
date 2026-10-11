"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents.graph import build_graph, set_prompt_registry_provider
from app.api.health import router as health_router
from app.api.middleware import RequestIDMiddleware, SecurityHeadersMiddleware
from app.api.prompts import router as prompts_router
from app.api.routes import router as api_router
from app.core.config import configure_langchain_environment, get_settings
from app.core.logging import configure_logging
from app.core.metrics import setup_instrumentator
from app.core.ratelimit import RateLimiter
from app.core.tracing import configure_langfuse_environment, shutdown_tracing
from app.services.prompt_registry import PromptRegistry

settings = get_settings()
configure_langchain_environment(settings)
configure_logging(settings)
configure_langfuse_environment(settings)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle application startup and shutdown events.

    Opens a persistent SQLite checkpointer for the lifetime of the process
    and compiles the checkpointed agent graph into `app.state.agent_graph`,
    which is what the API routes use (as opposed to the module-level,
    non-persistent `app.agents.graph.agent_graph` used by direct unit tests).
    """
    logger.info(
        "Starting %s v%s (environment=%s)",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )
    db_path = Path(settings.checkpoint_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    app.state.prompt_registry = PromptRegistry(settings.prompts_dir)
    app.state.prompt_registry.load()
    set_prompt_registry_provider(lambda: app.state.prompt_registry)
    if settings.prompt_admin_open:
        logger.warning(
            "PROMPT_ADMIN_OPEN=true — /prompts write endpoints accept requests "
            "without an API key. Development mode only; do not deploy like this."
        )

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        app.state.agent_graph = build_graph(checkpointer=checkpointer)
        app.state.rate_limiter = RateLimiter()
        yield

    shutdown_tracing()
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)

app.include_router(api_router, prefix=settings.api_prefix)
app.include_router(prompts_router, prefix=settings.api_prefix)
app.include_router(health_router)

setup_instrumentator(app)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Return the service health status (kept for backward compatibility)."""
    return {
        "status": "ok",
        "environment": settings.environment,
        "version": settings.app_version,
    }
