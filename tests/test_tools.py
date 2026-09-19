"""Unit tests for the modular LangChain tools (app/agents/tools.py)."""

from unittest.mock import MagicMock, patch

import pytest

from app.agents.tools import calculator, echo, run_tool, web_search


class TestCalculator:
    def test_evaluates_basic_expression(self) -> None:
        assert calculator.invoke("2 + 2 * 3") == "8"

    def test_rejects_empty_expression(self) -> None:
        with pytest.raises(ValueError):
            calculator.invoke("")

    def test_rejects_malformed_expression(self) -> None:
        with pytest.raises(ValueError):
            calculator.invoke("2 + ")


class TestEcho:
    def test_repeats_text(self) -> None:
        assert echo.invoke("hello") == "hello"

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValueError):
            echo.invoke("")


class TestWebSearch:
    """`DDGS` network calls are mocked to keep the suite fast and deterministic."""

    def test_formats_results(self) -> None:
        fake_results = [
            {"title": "Python", "href": "https://python.org", "body": "Official site"},
        ]
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__.return_value.text.return_value = fake_results
        with patch("app.agents.tools.DDGS", return_value=mock_ddgs):
            output = web_search.invoke("python programming language")
        assert "Python" in output
        assert "https://python.org" in output

    def test_no_results_returns_placeholder(self) -> None:
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__.return_value.text.return_value = []
        with patch("app.agents.tools.DDGS", return_value=mock_ddgs):
            output = web_search.invoke("a query with no results")
        assert output == "No results found."

    def test_rejects_empty_query(self) -> None:
        with pytest.raises(ValueError):
            web_search.invoke("")

    def test_wraps_search_failures(self) -> None:
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__.return_value.text.side_effect = RuntimeError("network down")
        with patch("app.agents.tools.DDGS", return_value=mock_ddgs):
            with pytest.raises(ValueError):
                web_search.invoke("anything")


class TestRunTool:
    def test_none_returns_empty_string(self) -> None:
        assert run_tool("none", "irrelevant") == ""

    def test_unknown_tool_raises(self) -> None:
        with pytest.raises(ValueError):
            run_tool("not_a_real_tool", "irrelevant")

    def test_dispatches_to_calculator(self) -> None:
        assert run_tool("calculator", "3 * 3") == "9"
