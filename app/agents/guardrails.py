"""Guardrails: input sanitization and prompt-injection detection.

Mitigates OWASP LLM01 (Prompt Injection) by sanitizing user input and
screening it against known jailbreak/injection patterns before it ever
reaches the planning LLM.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Mapping

from app.core.prompts import (
    GUARDRAIL_BLOCKED_ERROR_TEMPLATE,
    GUARDRAIL_EMPTY_INPUT_REASON,
    GUARDRAIL_INJECTION_REASON_TEMPLATE,
)

logger = logging.getLogger(__name__)

_MAX_INPUT_LENGTH = 4000

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


def guardrail_node(state: Mapping[str, object]) -> dict[str, object]:
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
