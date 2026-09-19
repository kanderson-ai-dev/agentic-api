"""Prometheus metrics for the agentic workflow and HTTP layer."""

from fastapi import FastAPI
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

agent_blocked_requests_total = Counter(
    "agent_blocked_requests_total",
    "Total number of agent requests blocked by the input guardrail (OWASP LLM01).",
)

agent_tool_calls_total = Counter(
    "agent_tool_calls_total",
    "Total number of tool calls executed by the agent, labeled by tool name.",
    labelnames=("tool",),
)


def setup_instrumentator(app: FastAPI) -> None:
    """Instrument the FastAPI app with default HTTP metrics and expose `/metrics`."""
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
