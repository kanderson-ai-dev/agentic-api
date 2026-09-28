"""REST API endpoints."""

import json
import logging
import uuid
from collections.abc import AsyncIterator, Iterable

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.graph import ToolCall
from app.agents.guardrails import screen_output
from app.api.dependencies import enforce_rate_limit, verify_api_key
from app.core.prompts import AGENT_EXECUTION_ERROR_MESSAGE, OUTPUT_REDACTED_MESSAGE

logger = logging.getLogger(__name__)

router = APIRouter()


class AgentRunRequest(BaseModel):
    """Payload for the agent run/stream endpoints."""

    input: str = Field(..., min_length=1, description="User input for the agent.")
    session_id: str | None = Field(
        default=None,
        description=(
            "Conversation/thread id used to persist and resume multi-turn memory. "
            "If omitted, a new session id is generated and returned in the response."
        ),
    )


class AgentRunResponse(BaseModel):
    """Structured result of an agent graph execution."""

    plan: str
    tool_calls: list[ToolCall]
    final_output: str
    errors: list[str]
    blocked: bool
    output_flagged: bool
    session_id: str


def _initial_state(input_text: str) -> dict[str, object]:
    """Build a fresh initial AgentState payload for a graph invocation."""
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


def _run_config(request: Request, session_id: str, tags: list[str]) -> dict[str, object]:
    """Build the LangGraph run config, correlating it with the request's `request_id`."""
    request_id = getattr(request.state, "request_id", None)
    return {
        "configurable": {"thread_id": session_id},
        "metadata": {"request_id": request_id},
        "tags": tags,
    }


def _serialize_update(update: dict[str, object]) -> dict[str, object]:
    """Make a partial node state update JSON-serializable for SSE streaming."""
    serialized: dict[str, object] = {}
    for key, value in update.items():
        if key == "messages" and isinstance(value, Iterable):
            serialized[key] = [
                {"type": getattr(m, "type", "unknown"), "content": getattr(m, "content", "")}
                for m in value
            ]
        else:
            serialized[key] = value
    return serialized


def _screen_serialized_update(update: dict[str, object]) -> dict[str, object]:
    """Apply the output guardrail to text fields inside a streamed state update.

    SSE emits each node's update as it happens — before the graph's
    `output_guardrail` node runs — so generated/untrusted fields (`plan`,
    `final_output`, tool payloads, message contents) get the same OWASP
    LLM02 screening here, with unsafe text replaced by a redaction marker.
    """
    screened = dict(update)
    for key, value in update.items():
        if isinstance(value, str) and not screen_output(value).is_safe:
            screened[key] = OUTPUT_REDACTED_MESSAGE
        elif key in ("tool_calls", "messages") and isinstance(value, list):
            screened[key] = [
                {
                    field: OUTPUT_REDACTED_MESSAGE
                    if isinstance(field_value, str) and not screen_output(field_value).is_safe
                    else field_value
                    for field, field_value in item.items()
                }
                if isinstance(item, dict)
                else item
                for item in value
            ]
    return screened


@router.post(
    "/agent/run",
    response_model=AgentRunResponse,
    status_code=status.HTTP_200_OK,
    tags=["agent"],
    dependencies=[Depends(enforce_rate_limit), Depends(verify_api_key)],
)
async def run_agent(payload: AgentRunRequest, request: Request) -> AgentRunResponse:
    """Execute the LangGraph agent workflow with the provided input."""
    graph = request.app.state.agent_graph
    session_id = payload.session_id or str(uuid.uuid4())
    config = _run_config(request, session_id, tags=["agentic-api"])
    try:
        result = await graph.ainvoke(_initial_state(payload.input), config=config)
    except Exception as exc:
        logger.exception("Agent graph execution failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=AGENT_EXECUTION_ERROR_MESSAGE,
        ) from exc
    return AgentRunResponse(
        plan=result["plan"],
        tool_calls=result["tool_calls"],
        final_output=result["final_output"],
        errors=result["errors"],
        blocked=result.get("blocked", False),
        output_flagged=result.get("output_flagged", False),
        session_id=session_id,
    )


@router.post(
    "/agent/stream",
    tags=["agent"],
    dependencies=[Depends(enforce_rate_limit), Depends(verify_api_key)],
)
async def stream_agent(payload: AgentRunRequest, request: Request) -> StreamingResponse:
    """Execute the agent workflow, streaming each node's state update as it happens (SSE)."""
    graph = request.app.state.agent_graph
    session_id = payload.session_id or str(uuid.uuid4())
    config = _run_config(request, session_id, tags=["agentic-api", "stream"])

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for update in graph.astream(
                _initial_state(payload.input), config=config, stream_mode="updates"
            ):
                for node_name, node_update in update.items():
                    data = json.dumps(
                        {
                            "node": node_name,
                            "update": _screen_serialized_update(
                                _serialize_update(node_update)
                            ),
                        }
                    )
                    yield f"data: {data}\n\n"
        except Exception:  # noqa: BLE001 - report streaming failure as an SSE event
            logger.exception("Agent graph streaming failed")
            yield f"event: error\ndata: {json.dumps({'detail': AGENT_EXECUTION_ERROR_MESSAGE})}\n\n"
            return
        yield f"event: done\ndata: {json.dumps({'session_id': session_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
