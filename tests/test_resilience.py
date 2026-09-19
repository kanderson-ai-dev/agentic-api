"""Tests for retry/backoff resilience around the planner LLM call."""

from unittest.mock import MagicMock

import httpx2
import openai
import pytest

from app.agents.graph import PlanResult, _invoke_planner

_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def _rate_limit_error() -> openai.RateLimitError:
    response = httpx2.Response(429, request=_REQUEST)
    return openai.RateLimitError("rate limited", response=response, body=None)


class TestInvokePlannerRetry:
    def test_retries_on_transient_errors_then_succeeds(self) -> None:
        expected = PlanResult(plan="ok", tool_calls=[])
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            _rate_limit_error(),
            openai.APITimeoutError(request=_REQUEST),
            expected,
        ]

        result = _invoke_planner(mock_llm, [])

        assert result is expected
        assert mock_llm.invoke.call_count == 3

    def test_gives_up_after_max_attempts(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = _rate_limit_error()

        with pytest.raises(openai.RateLimitError):
            _invoke_planner(mock_llm, [])

        assert mock_llm.invoke.call_count == 3

    def test_does_not_retry_non_transient_errors(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ValueError("some unrelated failure")

        with pytest.raises(ValueError):
            _invoke_planner(mock_llm, [])

        assert mock_llm.invoke.call_count == 1
