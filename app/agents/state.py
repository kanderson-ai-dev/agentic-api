"""Shared TypedDict state types for the LangGraph agent workflow."""

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ToolCall(TypedDict):
    """A single planned (and optionally executed) tool invocation."""

    tool: Literal["calculator", "echo", "web_search", "none"]
    input: str
    output: str


class AgentState(TypedDict):
    """Shared state propagated across the graph nodes."""

    input_text: str
    messages: Annotated[list[AnyMessage], add_messages]
    plan: str
    tool_calls: list[ToolCall]
    final_output: str
    errors: list[str]
    blocked: bool
    output_flagged: bool
