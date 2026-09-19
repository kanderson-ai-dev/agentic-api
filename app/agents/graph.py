"""LangGraph agent workflow: guardrail -> planner -> execution."""

import logging
from typing import Literal

from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.agents.guardrails import guardrail_node
from app.agents.tools import run_tool
from app.core.config import get_settings
from app.core.prompts import (
    AGENT_BLOCKED_DEFAULT_REASON,
    AGENT_BLOCKED_OUTPUT_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class ToolCall(TypedDict):
    """A single planned (and optionally executed) tool invocation."""

    tool: Literal["calculator", "echo", "none"]
    input: str
    output: str


class ToolCallPlan(BaseModel):
    """Structured tool call requested by the planner."""

    tool: Literal["calculator", "echo", "none"] = Field(
        description="Name of the tool to invoke."
    )
    input: str = Field(description="Input passed to the tool.")


class PlanResult(BaseModel):
    """Structured output produced by the planner node."""

    plan: str = Field(description="Short natural language execution plan.")
    tool_calls: list[ToolCallPlan] = Field(
        default_factory=list, description="Tool calls required to fulfill the plan."
    )


class AgentState(TypedDict):
    """Shared state propagated across the graph nodes."""

    input_text: str
    plan: str
    tool_calls: list[ToolCall]
    final_output: str
    errors: list[str]
    blocked: bool


def route_after_guardrail(state: AgentState) -> Literal["planner", "error_output"]:
    """Send blocked requests straight to the error-output node, bypassing the LLM."""
    return "error_output" if state.get("blocked") else "planner"


def error_output_node(state: AgentState) -> dict[str, object]:
    """Produce a structured error response for requests blocked by guardrails."""
    reason = "; ".join(state.get("errors", [])) or AGENT_BLOCKED_DEFAULT_REASON
    return {
        "plan": "",
        "tool_calls": [],
        "final_output": AGENT_BLOCKED_OUTPUT_TEMPLATE.format(reason=reason),
    }


def _build_llm() -> ChatOpenAI:
    """Instantiate the planner LLM using project settings."""
    settings = get_settings()
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


def planner_node(state: AgentState) -> dict[str, object]:
    """Use an LLM to turn the user input into a plan and candidate tool calls."""
    try:
        llm = _build_llm().with_structured_output(PlanResult)
        result = llm.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                {"role": "user", "content": state["input_text"]},
            ]
        )
        assert isinstance(result, PlanResult)
        tool_calls: list[ToolCall] = [
            {"tool": call.tool, "input": call.input, "output": ""}
            for call in result.tool_calls
        ]
        return {"plan": result.plan, "tool_calls": tool_calls, "errors": []}
    except Exception as exc:  # noqa: BLE001 - surface any planning failure as agent state
        logger.exception("Planner node failed")
        return {
            "plan": "",
            "tool_calls": [],
            "errors": [f"planner_error: {exc}"],
        }


def execution_node(state: AgentState) -> dict[str, object]:
    """Execute each planned tool call and assemble the final output."""
    executed: list[ToolCall] = []
    errors = list(state.get("errors", []))

    for call in state.get("tool_calls", []):
        try:
            output = run_tool(call["tool"], call["input"])
        except Exception as exc:  # noqa: BLE001 - keep going on per-tool failures
            logger.exception("Tool execution failed for %s", call["tool"])
            output = ""
            errors.append(f"tool_error[{call['tool']}]: {exc}")
        executed.append({"tool": call["tool"], "input": call["input"], "output": output})

    if executed:
        final_output = "\n".join(
            f"{call['tool']}({call['input']!r}) -> {call['output']}"
            for call in executed
            if call["tool"] != "none"
        ) or state.get("plan", "")
    else:
        final_output = state.get("plan", "")

    return {"tool_calls": executed, "final_output": final_output, "errors": errors}


def build_graph() -> CompiledStateGraph:
    """Build and compile the agent graph: guardrail -> planner -> execution."""
    builder = StateGraph(AgentState)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("planner", planner_node)
    builder.add_node("execution", execution_node)
    builder.add_node("error_output", error_output_node)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"planner": "planner", "error_output": "error_output"},
    )
    builder.add_edge("planner", "execution")
    builder.add_edge("execution", END)
    builder.add_edge("error_output", END)
    return builder.compile()


agent_graph = build_graph()
