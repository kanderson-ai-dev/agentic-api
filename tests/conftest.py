"""Shared pytest fixtures and helpers for the agentic-api test suite."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

settings = get_settings()

requires_openai_key = pytest.mark.skipif(
    settings.openai_api_key is None, reason="OPENAI_API_KEY not configured."
)
requires_langsmith_key = pytest.mark.skipif(
    settings.langchain_api_key is None, reason="LANGCHAIN_API_KEY not configured."
)


@pytest.fixture(scope="session")
def client():
    """A TestClient with the app's lifespan active (SQLite checkpointer, etc.).

    Using the `with` form is required so that `app.state.agent_graph` (set up
    in `app.main`'s lifespan) is actually initialized before requests hit the
    routes that depend on it.
    """
    with TestClient(app) as test_client:
        yield test_client


def initial_state(input_text: str) -> dict[str, object]:
    """Build a fresh initial AgentState payload for direct graph invocations."""
    return {
        "input_text": input_text,
        "messages": [],
        "plan": "",
        "tool_calls": [],
        "final_output": "",
        "errors": [],
        "blocked": False,
        "output_flagged": False,
    }
