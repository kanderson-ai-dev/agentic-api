"""Guardrails: input sanitization/prompt-injection detection and output screening.

Mitigates:
- OWASP LLM01 (Prompt Injection): sanitizes user input and screens it against
  known jailbreak/injection patterns before it ever reaches the planning LLM.
- OWASP LLM02 (Insecure Output Handling): screens the LLM/tool-generated
  output before it is returned to the caller, catching system-prompt leaks
  and cases where the model reflects back an injected instruction.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass

from app.agents.state import AgentState
from app.core.prompts import (
    GUARDRAIL_BLOCKED_ERROR_TEMPLATE,
    GUARDRAIL_EMPTY_INPUT_REASON,
    GUARDRAIL_INJECTION_REASON_TEMPLATE,
    OUTPUT_BLOCKED_MESSAGE,
    OUTPUT_GUARDRAIL_ERROR_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_MAX_INPUT_LENGTH = 4000
_SYSTEM_PROMPT_LEAK_NGRAM_SIZE = 8

# Heuristic patterns commonly seen in prompt injection / jailbreak attempts.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.I),
    re.compile(r"forget\s+(everything|all)\s+(you|i)\s+(were|was)\s+told", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+\w+", re.I),
    re.compile(r"reveal\s+(your|the)\s+(system\s+)?prompt", re.I),
    re.compile(r"(show|print|leak)\s+(me\s+)?(your|the)\s+(system\s+)?prompt", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+(are|were)|a\s+jailbroken)", re.I),
    re.compile(r"do\s+anything\s+now", re.I),  # DAN-style jailbreaks
    re.compile(r"\bDAN\b"),
    re.compile(r"bypass\s+(your\s+)?(safety|content)\s+(polic\w*|filters?|guidelines?)", re.I),
    re.compile(r"<\|.*?\|>"),  # fake special tokens, e.g. <|im_start|>
    re.compile(r"^\s*(system|assistant)\s*:", re.I | re.M),  # fake role headers
    re.compile(r"###\s*(system|instruction)", re.I),
]


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of running the guardrail checks against a piece of user input."""

    is_safe: bool
    sanitized_input: str
    reason: str | None = None


def sanitize_input(raw_input: str) -> str:
    """Normalize input and strip non-printable/control characters.

    Helps neutralize obfuscation tricks (unicode homoglyphs, control chars,
    zero-width characters) sometimes used to smuggle prompt injections past
    naive keyword filters.
    """
    normalized = unicodedata.normalize("NFKC", raw_input)
    cleaned = "".join(
        ch
        for ch in normalized
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )
    return cleaned.strip()[:_MAX_INPUT_LENGTH]


def detect_prompt_injection(text: str) -> str | None:
    """Return a human-readable reason if the text matches a known injection pattern."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return GUARDRAIL_INJECTION_REASON_TEMPLATE.format(pattern=pattern.pattern)
    return None


def run_guardrails(raw_input: str) -> GuardrailResult:
    """Sanitize input and screen it for prompt injection attempts (OWASP LLM01)."""
    if not raw_input or not raw_input.strip():
        return GuardrailResult(
            is_safe=False, sanitized_input="", reason=GUARDRAIL_EMPTY_INPUT_REASON
        )

    sanitized = sanitize_input(raw_input)
    reason = detect_prompt_injection(sanitized)
    if reason:
        logger.warning("Prompt injection blocked: %s", reason)
        return GuardrailResult(is_safe=False, sanitized_input=sanitized, reason=reason)
    return GuardrailResult(is_safe=True, sanitized_input=sanitized, reason=None)


def guardrail_node(state: AgentState) -> dict[str, object]:
    """LangGraph node: sanitize `input_text` and flag prompt injection attempts.

    Designed to run as the first node of the agent graph. Downstream nodes
    should be skipped (routed to an error-output node) whenever `blocked`
    is `True`.
    """
    result = run_guardrails(str(state.get("input_text", "")))
    if not result.is_safe:
        return {
            "input_text": result.sanitized_input,
            "blocked": True,
            "errors": [GUARDRAIL_BLOCKED_ERROR_TEMPLATE.format(reason=result.reason)],
        }
    return {"input_text": result.sanitized_input, "blocked": False, "errors": []}


def _ngrams(text: str, size: int) -> set[str]:
    """Return the set of lowercase word n-grams of the given size found in `text`."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < size:
        return set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


_SYSTEM_PROMPT_NGRAMS = _ngrams(PLANNER_SYSTEM_PROMPT, _SYSTEM_PROMPT_LEAK_NGRAM_SIZE)


def detect_system_prompt_leak(text: str) -> str | None:
    """Return a reason if `text` reproduces a long enough fragment of the system prompt."""
    overlap = _ngrams(text, _SYSTEM_PROMPT_LEAK_NGRAM_SIZE) & _SYSTEM_PROMPT_NGRAMS
    if overlap:
        return f"system prompt leak detected (shared phrase: {next(iter(overlap))!r})"
    return None


def screen_output(text: str) -> GuardrailResult:
    """Screen LLM/tool-generated output for leakage or reflected injection (OWASP LLM02)."""
    if not text:
        return GuardrailResult(is_safe=True, sanitized_input=text)

    reason = detect_prompt_injection(text) or detect_system_prompt_leak(text)
    if reason:
        logger.warning("Output guardrail blocked: %s", reason)
        return GuardrailResult(is_safe=False, sanitized_input=text, reason=reason)
    return GuardrailResult(is_safe=True, sanitized_input=text)


def output_guardrail_node(state: AgentState) -> dict[str, object]:
    """LangGraph node: screen `final_output` before it is returned to the caller.

    Runs after the execution node. If the output leaks the system prompt or
    reflects a previously-injected instruction, it is replaced with a
    generic safe message and `output_flagged` is set to `True`.
    """
    result = screen_output(str(state.get("final_output", "")))
    if not result.is_safe:
        errors = list(state.get("errors", []))
        errors.append(OUTPUT_GUARDRAIL_ERROR_TEMPLATE.format(reason=result.reason))
        return {
            "final_output": OUTPUT_BLOCKED_MESSAGE,
            "output_flagged": True,
            "errors": errors,
        }
    return {"output_flagged": False}
