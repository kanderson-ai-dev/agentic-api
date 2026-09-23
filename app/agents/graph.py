"""LangGraph agent workflow: guardrail -> planner -> execution -> output guardrail."""

import logging
from typing import Literal

import openai
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.agents.guardrails import guardrail_node, output_guardrail_node
from app.agents.state import AgentState, ToolCall
from app.agents.tools import run_tool
from app.core.config import get_settings
from app.core.metrics import agent_blocked_requests_total, agent_tool_calls_total
from app.core.prompts import (
    AGENT_BLOCKED_DEFAULT_REASON,
    AGENT_BLOCKED_OUTPUT_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class ToolCallPlan(BaseModel):
    """Structured tool call requested by the planner."""

    tool: Literal["calculator", "echo", "web_search", "none"] = Field(
        description="Name of the tool to invoke."
    )
    input: str = Field(description="Input passed to the tool.")


class PlanResult(BaseModel):
    """Structured output produced by the planner node."""

    plan: str = Field(description="Short natural language execution plan.")
    tool_calls: list[ToolCallPlan] = Field(
        default_factory=list, description="Tool calls required to fulfill the plan."
    )


def route_after_guardrail(state: AgentState) -> Literal["planner", "error_output"]:
    """Send blocked requests straight to the error-output node, bypassing the LLM."""
    if state.get("blocked"):
        agent_blocked_requests_total.inc()
        return "error_output"
    return "planner"


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
    return ChatOpenAI(
        model="gpt-4o-mini", temperature=0, api_key=settings.openai_api_key
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _invoke_planner(llm: Runnable, messages: list[AnyMessage]) -> PlanResult:
    """Invoke the structured-output planner LLM, retrying on transient OpenAI errors."""
    result = llm.invoke(messages)
    assert isinstance(result, PlanResult)
    return result


def planner_node(state: AgentState) -> dict[str, object]:
    """Use an LLM to turn the user input (plus conversation history) into a plan."""
    try:
        llm = _build_llm().with_structured_output(PlanResult)
        history = state.get("messages", [])
        result = _invoke_planner(
            llm,
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                *history,
                HumanMessage(content=state["input_text"]),
            ],
        )
        tool_calls: list[ToolCall] = [
            {"tool": call.tool, "input": call.input, "output": ""}
            for call in result.tool_calls
        ]
        return {
            "plan": result.plan,
            "tool_calls": tool_calls,
            "errors": [],
            "messages": [
                HumanMessage(content=state["input_text"]),
                AIMessage(content=result.plan),
            ],
        }
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
        if call["tool"] != "none":
            agent_tool_calls_total.labels(tool=call["tool"]).inc()
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


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Build and compile the agent graph.

    Flow: guardrail (input, OWASP LLM01) -> planner (+ retry/backoff) ->
    execution (tools) -> output guardrail (OWASP LLM02) -> END, with blocked
    requests short-circuited from guardrail straight to error_output -> END.
    """
    builder = StateGraph(AgentState)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("planner", planner_node)
    builder.add_node("execution", execution_node)
    builder.add_node("output_guardrail", output_guardrail_node)
    builder.add_node("error_output", error_output_node)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"planner": "planner", "error_output": "error_output"},
    )
    builder.add_edge("planner", "execution")
    builder.add_edge("execution", "output_guardrail")
    builder.add_edge("output_guardrail", END)
    builder.add_edge("error_output", END)
    return builder.compile(checkpointer=checkpointer)


# Module-level graph without a checkpointer: used for direct unit/integration
# tests of the pure graph logic (no conversation persistence). The FastAPI
# app wires a separate, checkpointed instance into `app.state.agent_graph`
# at startup (see `app/main.py`) for the actual API request path.
agent_graph = build_graph()
