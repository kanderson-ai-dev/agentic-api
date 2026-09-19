"""Liveness/readiness health check endpoints.

Kept outside the versioned `/api/v1` prefix, as is conventional for
infrastructure-level endpoints consumed by orchestrators/load balancers.
"""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Liveness probe: returns 200 as long as the process can handle requests."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness(request: Request) -> JSONResponse:
    """Readiness probe: checks required configuration and dependencies are present."""
    settings = get_settings()
    issues: list[str] = []

    if settings.openai_api_key is None:
        issues.append("OPENAI_API_KEY is not configured.")
    if settings.langchain_tracing_v2 and settings.langchain_api_key is None:
        issues.append("LANGCHAIN_API_KEY is not configured but tracing is enabled.")
    if getattr(request.app.state, "agent_graph", None) is None:
        issues.append("Agent graph/checkpointer has not been initialized.")

    if issues:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "issues": issues},
        )
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})
