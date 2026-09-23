"""In-memory fixed-window rate limiter for the agent endpoints.

Deliberately dependency-free: a small sliding-window counter per client
identity (API key when provided, otherwise remote IP). Lives in
`app.state.rate_limiter` so tests can reset or replace it, and is sized for a
single-process deployment — a multi-replica setup would swap this for a
shared store (e.g. Redis) behind the same interface.
"""

import asyncio
import time
from collections import deque


class RateLimiter:
    """Sliding-window request counter keyed by client identity."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self._window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str, max_requests: int) -> bool:
        """Record a request for `key`; return False once `max_requests` in the window are used."""
        async with self._lock:
            now = time.monotonic()
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self._window_seconds:
                hits.popleft()
            if len(hits) >= max_requests:
                return False
            hits.append(now)
            return True

    def clear(self) -> None:
        """Drop all recorded hits (used by tests to reset state)."""
        self._hits.clear()
