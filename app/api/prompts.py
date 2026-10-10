"""Prompt administration API (`/prompts`).

Exposes the prompt registry's lifecycle — list/inspect artifacts, diff
versions, create candidates, run the A/B evaluation gate, promote, roll back —
plus verification of the hash-chained audit log. Read endpoints follow the
existing optional `X-API-Key` behaviour; write endpoints additionally require
`require_prompt_admin` (503 unless `AGENTIC_API_KEY` is configured or
`PROMPT_ADMIN_OPEN=true`). The `actor` recorded in the audit chain is derived
server-side; error payloads stay generic (no paths, tracebacks, or regexes).
"""

import logging
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    status,
)
from pydantic import BaseModel, Field

from app.api.dependencies import (
    enforce_rate_limit,
    require_prompt_admin,
    verify_api_key,
)
from app.services.prompt_eval import (
    EvalReport,
    GateDecision,
    compare,
    default_dataset_path,
    load_dataset,
    offline_planner_factory,
    run_evaluation,
)
from app.services.prompt_registry import (
    MAX_TEMPLATE_CHARS,
    AuditVerification,
    PromptNotFoundError,
    PromptRegistry,
    PromptRegistryError,
    PromptVersion,
    RegistryConflictError,
    RegistryUnavailableError,
    RegistryValidationError,
    VersionRecord,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"
_VERSION_PATTERN = r"^v[0-9]{1,4}$"
_ADMIN_ACTOR = "api-key-holder"

# Reusable validated params (RegistryDep is declared after _get_registry below).
PromptName = Annotated[str, Path(pattern=_NAME_PATTERN)]
PromptVersionParam = Annotated[str, Path(pattern=_VERSION_PATTERN)]
FromVersion = Annotated[str, Query(alias="from", pattern=_VERSION_PATTERN)]
ToVersion = Annotated[str, Query(alias="to", pattern=_VERSION_PATTERN)]

ReadDeps = [Depends(enforce_rate_limit), Depends(verify_api_key)]
WriteDeps = [
    Depends(enforce_rate_limit),
    Depends(verify_api_key),
    Depends(require_prompt_admin),
]

# --- Request/response models ---------------------------------------------------


class CandidateRequest(BaseModel):
    """Body for creating a new candidate prompt version."""

    template: str = Field(
        min_length=1,
        max_length=MAX_TEMPLATE_CHARS,
        description="Full system-prompt template text.",
    )
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    description: str = ""
    notes: str = ""


class ABTestRequest(BaseModel):
    """Body for evaluating a candidate against the promoted version."""

    version: str = Field(
        pattern=_VERSION_PATTERN,
        description="Candidate version to evaluate (e.g. v2).",
    )


class PromoteRequest(BaseModel):
    """Body for promoting a candidate through the evaluation gate."""

    version: str = Field(pattern=_VERSION_PATTERN)
    reason: str = Field(min_length=1, description="Audit reason for the promotion.")
    override_reason: str | None = Field(
        default=None,
        min_length=10,
        max_length=500,
        description=(
            "Required when the gate reports an overridable regression; sealed "
            "into the audit chain."
        ),
    )


class RollbackRequest(BaseModel):
    """Body for rolling back to a previously registered version."""

    to_version: str = Field(pattern=_VERSION_PATTERN)
    reason: str = Field(min_length=1, description="Mandatory audit reason.")


class PromptDetail(BaseModel):
    """One prompt's versions with status, hash and creation timestamp."""

    name: str
    promoted: str | None
    versions: dict[str, VersionRecord]


class DiffResponse(BaseModel):
    """Unified diff between two prompt versions."""

    name: str
    from_version: str
    to_version: str
    diff: str


class ABTestResponse(BaseModel):
    """Gate decision plus both full evaluation reports."""

    decision: GateDecision
    candidate_report: EvalReport
    baseline_report: EvalReport


class PromoteResponse(BaseModel):
    name: str
    promoted: str
    decision: GateDecision


class RollbackResponse(BaseModel):
    name: str
    promoted: str


# --- Body aliases with OpenAPI examples ------------------------------------------

CandidateBody = Annotated[
    CandidateRequest,
    Body(
        openapi_examples={
            "new_candidate": {
                "summary": "Create a candidate",
                "value": {
                    "template": (
                        "You are the planning module of an autonomous agent."
                    ),
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                    "notes": "shorter system prompt, same behaviour",
                },
            }
        }
    ),
]
ABTestBody = Annotated[
    ABTestRequest,
    Body(
        openapi_examples={
            "evaluate_v2": {"summary": "A/B a candidate", "value": {"version": "v2"}}
        }
    ),
]
PromoteBody = Annotated[
    PromoteRequest,
    Body(
        openapi_examples={
            "promote_v2": {
                "summary": "Promote after a green gate",
                "value": {"version": "v2", "reason": "A/B gate passed"},
            },
            "promote_with_override": {
                "summary": "Promote over an overridable regression",
                "value": {
                    "version": "v2",
                    "reason": "scheduled rollout",
                    "override_reason": "accepted risk: tool-selection dip",
                },
            },
        }
    ),
]
RollbackBody = Annotated[
    RollbackRequest,
    Body(
        openapi_examples={
            "rollback_v1": {
                "summary": "Roll back to v1",
                "value": {"to_version": "v1", "reason": "regression in v2"},
            }
        }
    ),
]


# --- Helpers ---------------------------------------------------------------------


def _get_registry(request: Request) -> PromptRegistry:
    """Return the app-state registry created in the lifespan (503 if absent)."""
    registry = getattr(request.app.state, "prompt_registry", None)
    if not isinstance(registry, PromptRegistry):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="prompt registry unavailable.",
        )
    return registry


RegistryDep = Annotated[PromptRegistry, Depends(_get_registry)]


def _map_registry_error(exc: PromptRegistryError) -> HTTPException:
    """Translate service errors into generic HTTP failures (no internals)."""
    if isinstance(exc, PromptNotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, "unknown prompt or version.")
    if isinstance(exc, RegistryValidationError):
        return HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "invalid prompt identifier or content.",
        )
    if isinstance(exc, RegistryConflictError):
        return HTTPException(
            status.HTTP_409_CONFLICT, "conflict with current registry state."
        )
    if isinstance(exc, RegistryUnavailableError):
        return HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "prompt registry unavailable."
        )
    return HTTPException(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "prompt registry error."
    )


def _candidate_record(registry: PromptRegistry, name: str, version: str) -> None:
    """Require `version` to exist and carry `candidate` status."""
    record = registry.list_versions(name).get(version)
    if record is None:
        raise PromptNotFoundError(f"unknown version {name}/{version}")
    if record.status != "candidate":
        raise RegistryConflictError(
            f"{name}/{version} is {record.status}, not candidate."
        )


def _run_gate(
    registry: PromptRegistry, name: str, candidate_version: str
) -> tuple[GateDecision, EvalReport, EvalReport]:
    """Evaluate candidate vs. promoted and apply the promotion gate."""
    promoted = registry.list_prompts().get(name)
    if promoted is None:
        raise RegistryConflictError(f"prompt {name} has no promoted version.")
    dataset_path = default_dataset_path(registry.base_dir, name)
    if not dataset_path.is_file():
        raise RegistryUnavailableError("evaluation dataset unavailable.")
    dataset = load_dataset(dataset_path)
    candidate_report = run_evaluation(
        registry, name, candidate_version, offline_planner_factory, dataset_path
    )
    baseline_report = run_evaluation(
        registry, name, promoted, offline_planner_factory, dataset_path
    )
    return (
        compare(candidate_report, baseline_report, dataset.thresholds),
        candidate_report,
        baseline_report,
    )


# --- Read endpoints --------------------------------------------------------------


@router.get(
    "/prompts",
    tags=["prompts"],
    dependencies=ReadDeps,
    response_model=dict[str, str | None],
)
def list_prompts(registry: RegistryDep) -> dict[str, Any]:
    """List all prompt names with their promoted version."""
    return dict(registry.list_prompts())


@router.get(
    "/prompts/audit",
    tags=["prompts"],
    dependencies=ReadDeps,
    response_model=AuditVerification,
)
def verify_audit(registry: RegistryDep) -> AuditVerification:
    """Recompute the hash-chained audit log and report the first broken index."""
    return registry.verify_chain()


@router.get(
    "/prompts/{name}",
    tags=["prompts"],
    dependencies=ReadDeps,
    response_model=PromptDetail,
)
def get_prompt(name: PromptName, registry: RegistryDep) -> PromptDetail:
    """List versions of a prompt with status, content hash and created_at."""
    try:
        versions = registry.list_versions(name)
    except PromptRegistryError as exc:
        raise _map_registry_error(exc) from exc
    return PromptDetail(
        name=name, promoted=registry.list_prompts().get(name), versions=versions
    )


@router.get(
    "/prompts/{name}/versions/{version}",
    tags=["prompts"],
    dependencies=ReadDeps,
    response_model=PromptVersion,
)
def get_version(
    name: PromptName, version: PromptVersionParam, registry: RegistryDep
) -> PromptVersion:
    """Return the full immutable artifact (template included)."""
    try:
        artifact = registry.get_version(name, version)
    except PromptRegistryError as exc:
        raise _map_registry_error(exc) from exc
    if artifact is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "prompt artifact unavailable.",
        )
    return artifact


@router.get(
    "/prompts/{name}/diff",
    tags=["prompts"],
    dependencies=ReadDeps,
    response_model=DiffResponse,
)
def diff_versions(
    name: PromptName,
    from_version: FromVersion,
    to_version: ToVersion,
    registry: RegistryDep,
) -> DiffResponse:
    """Unified diff of metadata + template between two versions."""
    try:
        diff_text = registry.diff(name, from_version, to_version)
    except PromptRegistryError as exc:
        raise _map_registry_error(exc) from exc
    return DiffResponse(
        name=name, from_version=from_version, to_version=to_version, diff=diff_text
    )


# --- Write endpoints (prompt admin) ---------------------------------------------


@router.post(
    "/prompts/{name}/candidates",
    tags=["prompts"],
    dependencies=WriteDeps,
    response_model=PromptVersion,
    status_code=status.HTTP_201_CREATED,
)
def create_candidate(
    payload: CandidateBody, name: PromptName, registry: RegistryDep
) -> PromptVersion:
    """Create the next immutable `vN.yaml` as a `candidate`."""
    try:
        return registry.create_candidate(
            name,
            template=payload.template,
            model=payload.model,
            temperature=payload.temperature,
            author=_ADMIN_ACTOR,
            notes=payload.notes,
            description=payload.description,
        )
    except PromptRegistryError as exc:
        raise _map_registry_error(exc) from exc


@router.post(
    "/prompts/{name}/ab-test",
    tags=["prompts"],
    dependencies=WriteDeps,
    response_model=ABTestResponse,
)
def ab_test(
    payload: ABTestBody, name: PromptName, registry: RegistryDep
) -> ABTestResponse:
    """Evaluate a candidate vs. the promoted version on the gate dataset."""
    try:
        _candidate_record(registry, name, payload.version)
        decision, candidate_report, baseline_report = _run_gate(
            registry, name, payload.version
        )
    except PromptRegistryError as exc:
        raise _map_registry_error(exc) from exc
    return ABTestResponse(
        decision=decision,
        candidate_report=candidate_report,
        baseline_report=baseline_report,
    )


@router.post(
    "/prompts/{name}/promote",
    tags=["prompts"],
    dependencies=WriteDeps,
    response_model=PromoteResponse,
)
def promote(
    payload: PromoteBody, name: PromptName, registry: RegistryDep
) -> PromoteResponse:
    """Promote a candidate, gated on the server-side A/B evaluation.

    The gate is re-run here — client-supplied metrics are never trusted. A
    regression requires `override_reason` (sealed in the audit chain); a
    safety-metric regression is never overridable.
    """
    try:
        _candidate_record(registry, name, payload.version)
        decision, candidate_report, baseline_report = _run_gate(
            registry, name, payload.version
        )
    except PromptRegistryError as exc:
        raise _map_registry_error(exc) from exc
    if not decision.passed:
        if not decision.overridable or payload.override_reason is None:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                {
                    "detail": "promotion blocked by evaluation gate.",
                    "overridable": decision.overridable,
                    "regressions": decision.regressions,
                },
            )
    try:
        registry.promote(
            name,
            payload.version,
            actor=_ADMIN_ACTOR,
            reason=payload.reason,
            override_reason=payload.override_reason,
            metrics={
                "candidate_metrics": candidate_report.metrics,
                "baseline_metrics": baseline_report.metrics,
                "deltas": decision.deltas,
                "regressions": decision.regressions,
            },
        )
    except PromptRegistryError as exc:
        raise _map_registry_error(exc) from exc
    return PromoteResponse(name=name, promoted=payload.version, decision=decision)


@router.post(
    "/prompts/{name}/rollback",
    tags=["prompts"],
    dependencies=WriteDeps,
    response_model=RollbackResponse,
)
def rollback(
    payload: RollbackBody, name: PromptName, registry: RegistryDep
) -> RollbackResponse:
    """Roll back to a previously registered version; `reason` is mandatory."""
    try:
        registry.rollback(
            name, payload.to_version, actor=_ADMIN_ACTOR, reason=payload.reason
        )
    except PromptRegistryError as exc:
        raise _map_registry_error(exc) from exc
    return RollbackResponse(name=name, promoted=payload.to_version)
