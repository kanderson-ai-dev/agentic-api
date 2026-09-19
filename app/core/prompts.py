"""Centralized system prompts and instruction templates for the agent workflow.

Keeping prompt text here (rather than scattered across `app/agents/*.py`)
makes prompts easier to review, version, and iterate on independently of
the graph/node wiring that consumes them.
"""

# --- Planner node -----------------------------------------------------------

TOOL_SELECTION_GUIDELINES = (
    "Available tools:\n"
    "- calculator: evaluates a basic arithmetic expression (input must be a valid "
    "math expression, e.g. '2 + 2').\n"
    "- echo: simply repeats back a piece of text (input is the text to repeat).\n"
    "- none: use when no tool is required to answer the request.\n"
)

PLANNER_SYSTEM_PROMPT = (
    "You are the planning module of an autonomous agent. Given a user request, "
    "produce a short execution plan and, if needed, one or more tool calls.\n"
    f"{TOOL_SELECTION_GUIDELINES}"
)

# --- Guardrail node -----------------------------------------------------------

GUARDRAIL_EMPTY_INPUT_REASON = "empty input"

GUARDRAIL_INJECTION_REASON_TEMPLATE = "matched suspicious pattern: {pattern}"

GUARDRAIL_BLOCKED_ERROR_TEMPLATE = "guardrail_blocked: {reason}"

# --- Error output node -----------------------------------------------------------

AGENT_BLOCKED_DEFAULT_REASON = "Request blocked by guardrails."

AGENT_BLOCKED_OUTPUT_TEMPLATE = "Request blocked: {reason}"
