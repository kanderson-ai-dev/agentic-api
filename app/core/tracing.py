"""Optional Langfuse tracing for the agent graph.

Opt-in only: when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set,
`build_trace_callbacks` attaches a `langfuse.langchain.CallbackHandler` to each
LangGraph run; otherwise it returns an empty list and nothing leaves the
process. Any initialization or runtime failure degrades to "no tracing" — a
broken observability backend must never break a request.

Prompt-text protection: the planner's `SystemMessage` is an *input* to the LLM
call, so without masking the full template would land in every generation
span. `register_prompt_template` (called by the planner node when it resolves
a prompt) records each served template, and the client `mask` rewrites it to a
version/hash placeholder before export.
"""

import hashlib
import logging
import threading
from typing import TYPE_CHECKING, Any

from app.core.config import Settings

if TYPE_CHECKING:  # pragma: no cover - import only used for typing
    from langfuse import Langfuse

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_clients: dict[str, "Langfuse"] = {}
# template text -> masked replacement; templates the process has actually served.
_template_placeholders: dict[str, str] = {}


def langfuse_enabled(settings: Settings) -> bool:
    """True when both Langfuse credentials are configured."""
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def register_prompt_template(name: str, version: str, template: str) -> None:
    """Record a served prompt template so `mask` can replace it in trace data."""
    if not template:
        return
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
    with _lock:
        _template_placeholders[template] = (
            f"[prompt:{name}@{version} sha256:{digest}]"
        )


def _mask(*, data: Any, **_kwargs: Any) -> Any:
    """Replace every registered prompt template inside `data` with its placeholder."""
    if isinstance(data, str):
        # Longest templates first: a candidate that extends another template
        # must match whole before its substring prefix is replaced piecemeal.
        for template, placeholder in sorted(
            _template_placeholders.items(), key=lambda kv: -len(kv[0])
        ):
            if template in data:
                data = data.replace(template, placeholder)
        return data
    if isinstance(data, dict):
        return {key: _mask(data=value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_mask(data=item) for item in data]
    return data


def _get_or_create_client(settings: Settings) -> "Langfuse":
    """Return the cached client for this key, building it (with `mask`) once."""
    from langfuse import Langfuse

    assert settings.langfuse_public_key is not None
    assert settings.langfuse_secret_key is not None
    public_key = settings.langfuse_public_key
    with _lock:
        if public_key not in _clients:
            _clients[public_key] = Langfuse(
                public_key=public_key,
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                base_url=settings.langfuse_base_url,
                mask=_mask,
            )
        return _clients[public_key]


def configure_langfuse_environment(settings: Settings) -> None:
    """Eagerly initialize the Langfuse client at startup when credentials exist.

    Keys are passed explicitly (not exported to `os.environ`) so no code path
    can silently create an unmasked client. Failures degrade to "no tracing".
    """
    if not langfuse_enabled(settings):
        return
    try:
        _get_or_create_client(settings)
        logger.info(
            "Langfuse tracing enabled (base_url=%s)", settings.langfuse_base_url
        )
    except Exception:  # noqa: BLE001 - tracing must never break startup
        logger.warning("Langfuse tracing disabled: client initialization failed.")


def build_trace_callbacks(settings: Settings) -> list[object]:
    """Return a Langfuse `CallbackHandler` for a run, or `[]` when disabled.

    A fresh handler is created per request (cheap: it resolves the singleton
    client), so concurrent runs cannot share handler state.
    """
    if not langfuse_enabled(settings):
        return []
    try:
        from langfuse.langchain import CallbackHandler

        _get_or_create_client(settings)
        assert settings.langfuse_public_key is not None
        return [CallbackHandler(public_key=settings.langfuse_public_key)]
    except Exception:  # noqa: BLE001 - degrade to no tracing
        logger.warning("Langfuse tracing disabled for this request.")
        return []


def shutdown_tracing() -> None:
    """Flush and shut down every cached Langfuse client (lifespan teardown)."""
    with _lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        try:
            client.shutdown()
        except Exception:  # noqa: BLE001 - never block shutdown
            logger.warning("Langfuse shutdown failed; continuing.")


def _reset_for_tests() -> None:
    """Clear cached clients and registered templates (test isolation only)."""
    with _lock:
        _clients.clear()
        _template_placeholders.clear()
