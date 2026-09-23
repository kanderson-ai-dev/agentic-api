"""Shared pytest fixtures and helpers for the agentic-api test suite."""

import os
import re

import pytest
from fastapi.testclient import TestClient

from app.agents.graph import PlanResult, ToolCallPlan
from app.core.config import get_settings
from app.main import app

settings = get_settings()

# Set AGENTIC_LIVE_TESTS=1 (with the real API keys configured) to exercise the
# live OpenAI/LangSmith paths instead of the deterministic fakes. The optional
# `live-e2e` CI job does exactly this; the default suite runs fully offline.
LIVE_TESTS = os.environ.get("AGENTIC_LIVE_TESTS") == "1"

requires_openai_key = pytest.mark.skipif(
    settings.openai_api_key is None, reason="OPENAI_API_KEY not configured."
)
requires_langsmith_key = pytest.mark.skipif(
    settings.langchain_api_key is None, reason="LANGCHAIN_API_KEY not configured."
)
requires_live_tests = pytest.mark.skipif(
    not LIVE_TESTS, reason="AGENTIC_LIVE_TESTS=1 not set; running in offline mode."
)

_ARITHMETIC_RE = re.compile(r"(\d+)\s*(times|multiplied by|plus|\+|\*|x)\s*(\d+)", re.I)
_REPEAT_RE = re.compile(r"repeat this exactly:\s*(.+)", re.I | re.S)


def _fake_plan(messages: list) -> PlanResult:
    """Deterministic planner output for the fake LLM (no OpenAI call).

    Mirrors what `gpt-4o-mini` reliably decides for the suite's fixed inputs:
    arithmetic questions -> `calculator`, "repeat this exactly: ..." -> `echo`,
    anything else -> a plan-only answer with no tool calls.
    """
    text = str(getattr(messages[-1], "content", "")) if messages else ""
    match = _ARITHMETIC_RE.search(text)
    if match:
        left, op_word, right = match.group(1), match.group(2).lower(), match.group(3)
        op = "+" if op_word in ("plus", "+") else "*"
        expression = f"{left} {op} {right}"
        return PlanResult(
            plan=f"Calculate {expression}.",
            tool_calls=[ToolCallPlan(tool="calculator", input=expression)],
        )
    repeat = _REPEAT_RE.search(text)
    if repeat:
        return PlanResult(
            plan="Repeat the provided text verbatim.",
            tool_calls=[ToolCallPlan(tool="echo", input=repeat.group(1).strip())],
        )
    return PlanResult(plan="Respond to the user without calling tools.", tool_calls=[])


class _FakePlannerLLM:
    """Stand-in for `ChatOpenAI` whose structured output is deterministic."""

    def with_structured_output(self, schema: type) -> "_FakePlannerLLM":
        return self

    def invoke(self, messages: list) -> PlanResult:
        return _fake_plan(messages)


@pytest.fixture
def fake_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch `_build_llm` so planner-node tests never call the OpenAI API.

    Applied by default so CI stays green with zero secrets; when
    `AGENTIC_LIVE_TESTS=1` and a real `OPENAI_API_KEY` is configured the patch
    is skipped, exercising the real planner end to end.
    """
    if LIVE_TESTS and settings.openai_api_key is not None:
        return
    monkeypatch.setattr("app.agents.graph._build_llm", _FakePlannerLLM)


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
