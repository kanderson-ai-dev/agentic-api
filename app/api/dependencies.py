"""FastAPI dependencies shared across API routes."""

import logging
import secrets

from fastapi import Header, HTTPException, Request, status

from app.core.config import get_settings
from app.core.ratelimit import RateLimiter

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"


async def verify_api_key(x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER)) -> None:
    """Enforce the `X-API-Key` header only when `AGENTIC_API_KEY` is configured.

    Keeps the API open for local development / the README quick-start when
    no key has been set, while requiring a valid key once one is configured
    (recommended for any non-local deployment).

    Raises:
        HTTPException: 401 if a key is configured and the request's header is
            missing or does not match.
    """
    settings = get_settings()
    if settings.agentic_api_key is None:
        return

    expected = settings.agentic_api_key.get_secret_value()
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        logger.warning("Rejected request with invalid or missing API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


_FALLBACK_LIMITER = RateLimiter()


async def enforce_rate_limit(request: Request) -> None:
    """Reject requests beyond `RATE_LIMIT_PER_MINUTE` per client identity.

    Keyed by the `X-API-Key` header when present, otherwise the client IP.
    Runs before auth so floods are rejected before any credential work; a
    limit of 0 disables enforcement (local/dev friendly).

    Raises:
        HTTPException: 429 once the configured per-minute budget is spent.
    """
    settings = get_settings()
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return

    limiter = getattr(request.app.state, "rate_limiter", None) or _FALLBACK_LIMITER
    client_key = request.headers.get(API_KEY_HEADER) or (
        request.client.host if request.client else "unknown"
    )
    if not await limiter.allow(client_key, limit):
        logger.warning("Rate limit exceeded for client.")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
        )
