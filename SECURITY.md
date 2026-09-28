# Security Model

This document maps the service's controls to the **OWASP Top 10** and the
**OWASP LLM Top 10**, with emphasis on the threats distinctive to this
architecture: **direct prompt injection against a paid LLM endpoint** and
**tool misuse by the planner** — both bounded *before* money is spent.

## Threat model

| Threat | Surface | Control |
|---|---|---|
| Direct prompt injection | `input` on `POST /api/v1/agent/run` and `/agent/stream` | `guardrail_node` is the graph's first node — NFKC normalization, control-character stripping, and injection-pattern screening run *before* any LLM call; blocked input burns zero tokens (`app/agents/guardrails.py`) |
| Indirect prompt injection | `web_search` snippets (untrusted web content) | Tool output never re-enters the planner's message context — it is assembled into `final_output` and screened by `output_guardrail_node`, treated strictly as *data* |
| System prompt leakage | `final_output` echoing `PLANNER_SYSTEM_PROMPT` or a reflected injected instruction | `detect_system_prompt_leak` (8-gram overlap against the system prompt) plus pattern re-screening; a flagged response withholds the whole generated surface — `final_output`, `plan`, `tool_calls` payloads, and streamed SSE fields are all redacted (OWASP LLM02/LLM07) |
| Code execution via tool input | `calculator(expression)` | `ast.parse` + operator allowlist — numeric literals and `+ - * / **` only; there is no `eval`/`exec` path (`app/agents/tools.py`) |
| Excessive agency | Planner tool selection | Structured output constrains `tool` to a `Literal` allowlist (`calculator`/`echo`/`web_search`/`none`); `TOOL_REGISTRY` is fixed at build time and the graph is a single pass — no autonomous loop |
| Credential / internals leakage | Config, logs, error bodies | `SecretStr` fields with blank→`None` normalization; client-visible errors are stable generic markers (`planner_error`, `tool_error[…]`, generic 401/500/SSE bodies) while exception detail is logged server-side only; secrets accepted only via headers/env — never query strings |
| Auth bypass / timing oracle | `/agent/*` endpoints | `X-API-Key` compared with `secrets.compare_digest` (constant time) when `AGENTIC_API_KEY` is configured; a single generic 401 |
| Rate / cost abuse | `/agent/*` endpoints | Sliding-window `RateLimiter` keyed by API key or client IP runs *before* auth — floods are rejected before credential work (`app/core/ratelimit.py`) |
| Secret exposure / supply chain via CI | GitHub Actions | The default test job injects zero secrets (fully offline suite); the `live-e2e` job runs only on manual/weekly triggers; `pip-audit` scans pinned dependencies and `gitleaks` scans git history |

## OWASP LLM Top 10 mapping

- **LLM01 Prompt Injection** — the input guardrail runs before any paid
  call; indirect injection is mitigated structurally: tool results are
  never re-fed to the model, only screened on the way out. Adversarial
  cases covered by tests (`tests/test_evaluations.py`,
  `tests/test_streaming.py`).
- **LLM02 Sensitive Information Disclosure** — `SecretStr` for every
  secret so keys cannot leak through reprs or logs; client-facing errors
  carry stable generic markers only (exception text is logged, not
  returned); output screening catches echoed prompts without quoting the
  matched fragment back to the client.
- **LLM03 Supply Chain** — pinned `requirements.txt`; CI installs from
  the pin file, gates on ruff + mypy + pytest, and runs a `pip-audit`
  vulnerability scan.
- **LLM05 Improper Output Handling** — `output_guardrail_node` gates
  `final_output` *and* redacts `plan`/`tool_calls` when a response is
  flagged; SSE streams node *metadata* live but releases generated
  content only after the graph completes (withheld entirely when flagged
  or failed mid-run); responses are rendered as plain data, never
  executed.
- **LLM06 Excessive Agency** — `Literal` tool allowlist via
  `with_structured_output`, a fixed tool registry, and a single
  non-looping graph pass; planner retries capped at 3 attempts.
- **LLM07 System Prompt Leakage** — 8-gram overlap detection against
  `PLANNER_SYSTEM_PROMPT`; no secret is ever interpolated into prompts.
- **LLM10 Unbounded Consumption** — 4000-character input cap, one LLM
  call per request, bounded retries, and per-client rate limiting before
  auth — spend per request is structurally bounded.

## OWASP Top 10 (classic)

- **A01 Broken Access Control** — `X-API-Key` enforcement on `/agent/*`
  when `AGENTIC_API_KEY` is configured; health and metrics stay open for
  orchestrators.
- **A02 Cryptographic Failures** — constant-time `secrets.compare_digest`
  for key checks; `SecretStr` for every held secret.
- **A03 Injection** — calculator evaluated through an AST allowlist (no
  `eval`/`exec`); input sanitization strips control characters and unicode
  obfuscation; Pydantic v2 validates request bodies.
- **A05 Security Misconfiguration** — security headers middleware
  (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` on every
  response, errors included); non-root Docker image with a `HEALTHCHECK`;
  `debug` defaults off; `/metrics` excluded from the OpenAPI schema;
  `.dockerignore` keeps `.env`, tests, and docs out of the image.
- **A07 Identification/Auth Failures** — one generic 401 ("Invalid or
  missing API key") — no oracle distinguishing absent vs. wrong keys.
- **A09 Logging Failures** — structured `structlog` output correlated by
  `X-Request-ID` end to end (HTTP logs ↔ LangSmith run metadata);
  guardrail blocks and tool failures are logged; Prometheus exposes
  `agent_blocked_requests_total` / `agent_tool_calls_total`.

## Known limitations

- The guardrails are heuristic (regex patterns + n-gram overlap) — a
  cheap first line of defense, not a proof of safety; a motivated
  attacker may still evade them.
- `RateLimiter` is in-memory and single-replica; scaling out requires a
  shared store (e.g. Redis) behind the same interface — documented in
  `app/core/ratelimit.py`.
- Auth is a single static API key — no per-client identity, rotation, or
  scopes. `session_id` additionally acts as an implicit bearer for
  conversation memory: anyone holding a session id can resume that
  thread's history. Treat session ids as sensitive; per-user session
  scoping would require real user authentication.
- `web_search` egresses model-chosen query text to DuckDuckGo at request
  time; there is no arbitrary-URL fetch surface, but queries do leave the
  perimeter. `SEARCH_PROVIDER=none` disables outbound search entirely for
  offline/egress-restricted deployments; a keyed provider (e.g. Tavily)
  with a data-processing agreement is the production-grade path.
