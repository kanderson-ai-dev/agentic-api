"""FastAPI dependencies shared across API routes."""

import logging
import secrets

from fastapi import Header, HTTPException, status

from app.core.config import get_settings

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
