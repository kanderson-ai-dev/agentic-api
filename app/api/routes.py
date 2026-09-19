"""REST API endpoints."""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.graph import ToolCall, agent_graph

logger = logging.getLogger(__name__)

router = APIRouter()


class AgentRunRequest(BaseModel):
    """Payload for the agent run endpoint."""

    input: str = Field(..., min_length=1, description="User input for the agent.")


class AgentRunResponse(BaseModel):
    """Structured result of an agent graph execution."""

    plan: str
    tool_calls: list[ToolCall]
    final_output: str
    errors: list[str]
    blocked: bool


@router.post(
    "/agent/run",
    response_model=AgentRunResponse,
    status_code=status.HTTP_200_OK,
    tags=["agent"],
)
async def run_agent(payload: AgentRunRequest) -> AgentRunResponse:
    """Execute the LangGraph agent workflow with the provided input."""
    try:
        result = await agent_graph.ainvoke(
            {
                "input_text": payload.input,
                "plan": "",
                "tool_calls": [],
                "final_output": "",
                "errors": [],
                "blocked": False,
            }
        )
    except Exception as exc:
        logger.exception("Agent graph execution failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent execution failed.",
        ) from exc
    return AgentRunResponse(
        plan=result["plan"],
        tool_calls=result["tool_calls"],
        final_output=result["final_output"],
        errors=result["errors"],
        blocked=result.get("blocked", False),
    )
