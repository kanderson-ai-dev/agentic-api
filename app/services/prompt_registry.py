"""Versioned prompt registry with a SHA-256 hash-chained audit log.

Prompt artifacts live as immutable YAML files under ``prompts/<name>/vN.yaml``;
mutable lifecycle state (the promoted pointer, per-version status, and the
audit chain) lives in the committed ``prompts/registry.json`` so every change
is reviewable in a PR. All writes are atomic (temp file + ``os.replace``) and
serialised through a single lock. Read paths degrade gracefully: a missing or
corrupt registry yields ``None``/fallback behaviour instead of raising into
the agent graph.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.guardrails import detect_prompt_injection

logger = logging.getLogger(__name__)

MAX_TEMPLATE_CHARS = 8000
GENESIS_PREV_HASH = "0" * 64
REGISTRY_FILENAME = "registry.json"

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_VERSION_PATTERN = re.compile(r"^v[0-9]{1,4}$")
_VERSION_FILE_PATTERN = re.compile(r"^v([0-9]{1,4})\.yaml$")

# Candidate templates are screened for anything resembling a credential so a
# prompt artifact can never smuggle a key into the repo or the runtime.
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\blsv2_[A-Za-z0-9_]+"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+"),
]

# Fields hashed into each audit entry; the entry's own `hash` is excluded.
_AUDIT_HASH_FIELDS = (
    "index",
    "ts",
    "action",
    "name",
    "version",
    "actor",
    "reason",
    "metrics",
    "prev_hash",
)

PromptStatus = Literal["candidate", "promoted", "retired"]
AuditAction = Literal["create_candidate", "promote", "rollback"]


class PromptRegistryError(Exception):
    """Base class for prompt-registry failures."""


class RegistryValidationError(PromptRegistryError):
    """Malformed name/version or unsafe candidate content (maps to HTTP 422)."""


class PromptNotFoundError(PromptRegistryError):
    """Unknown prompt name or version (maps to HTTP 404)."""


class RegistryConflictError(PromptRegistryError):
    """Invalid lifecycle transition or immutable-file overwrite (HTTP 409)."""


class RegistryUnavailableError(PromptRegistryError):
    """The registry is corrupt/unreadable and refuses writes (HTTP 503)."""


def _require_valid_name(name: str) -> None:
    if not _NAME_PATTERN.fullmatch(name):
        raise RegistryValidationError(
            "Invalid prompt name: expected ^[a-z][a-z0-9_]{0,31}$."
        )


def _require_valid_version(version: str) -> None:
    if not _VERSION_PATTERN.fullmatch(version):
        raise RegistryValidationError(
            "Invalid prompt version: expected ^v[0-9]{1,4}$."
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256_text(blob)


class PromptVersion(BaseModel):
    """One immutable prompt artifact (``prompts/<name>/vN.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str = ""
    model: str
    temperature: float = 0.0
    created_at: str
    author: str = ""
    notes: str = ""
    template: str

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        _require_valid_name(value)
        return value

    @field_validator("version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        _require_valid_version(value)
        return value


class VersionRecord(BaseModel):
    """Mutable per-version state tracked in ``registry.json``."""

    model_config = ConfigDict(extra="forbid")

    status: PromptStatus
    hash: str
    created_at: str


class PromptState(BaseModel):
    """Lifecycle state for one prompt name."""

    model_config = ConfigDict(extra="forbid")

    promoted: str | None = None
    versions: dict[str, VersionRecord] = Field(default_factory=dict)


class AuditEntry(BaseModel):
    """One sealed entry in the hash-chained audit log."""

    model_config = ConfigDict(extra="forbid")

    index: int
    ts: str
    action: AuditAction
    name: str
    version: str
    actor: str
    reason: str = ""
    metrics: dict[str, Any] | None = None
    prev_hash: str
    hash: str


class RegistryFile(BaseModel):
    """On-disk schema of ``prompts/registry.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    prompts: dict[str, PromptState] = Field(default_factory=dict)
    audit: list[AuditEntry] = Field(default_factory=list)


class AuditVerification(BaseModel):
    """Result of recomputing the audit chain."""

    valid: bool
    entries: int
    audit_root: str | None
    broken_at: int | None


def _seal_entry(entry: AuditEntry) -> str:
    """Compute the entry's hash over the canonical JSON of its hashed fields."""
    payload = {field: getattr(entry, field) for field in _AUDIT_HASH_FIELDS}
    return _canonical_hash(payload)


def _artifact_diff_text(artifact: PromptVersion) -> str:
    """Render an artifact as comparable text: metadata lines then the template."""
    metadata = {
        "name": artifact.name,
        "version": artifact.version,
        "description": artifact.description,
        "model": artifact.model,
        "temperature": artifact.temperature,
        "created_at": artifact.created_at,
        "author": artifact.author,
        "notes": artifact.notes,
    }
    lines = [f"{key}: {value}" for key, value in metadata.items()]
    return "\n".join([*lines, "---", artifact.template])


class PromptRegistry:
    """Load, query, and mutate the prompt registry rooted at ``base_dir``.

    The registry file is parsed lazily on first access and cached in memory;
    every mutation rewrites ``registry.json`` atomically and invalidates the
    artifact cache. A missing file starts an empty (writable) registry, while
    a corrupt file marks the registry unavailable: reads fall back to ``None``
    and writes raise ``RegistryUnavailableError``.
    """

    def __init__(self, base_dir: str | Path = "prompts") -> None:
        self._base = Path(base_dir)
        self._registry_path = self._base / REGISTRY_FILENAME
        self._lock = threading.Lock()
        self._state: RegistryFile | None = None
        self._loaded = False
        self._artifact_cache: dict[tuple[str, str], PromptVersion | None] = {}

    # -- loading -----------------------------------------------------------

    def load(self) -> bool:
        """(Re)read ``registry.json`` from disk. Returns True when healthy."""
        try:
            raw = self._registry_path.read_text(encoding="utf-8")
            state = RegistryFile.model_validate(json.loads(raw))
            for name, prompt in state.prompts.items():
                _require_valid_name(name)
                for version in prompt.versions:
                    _require_valid_version(version)
                if prompt.promoted is not None:
                    _require_valid_version(prompt.promoted)
        except FileNotFoundError:
            logger.warning(
                "Prompt registry not found at %s; starting empty.", self._registry_path
            )
            self._state = RegistryFile()
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning(
                "Prompt registry at %s is unreadable (%s); falling back.",
                self._registry_path,
                exc,
            )
            self._state = None
        else:
            self._state = state
        self._loaded = True
        self._artifact_cache.clear()
        return self._state is not None

    def _ensure_loaded(self) -> RegistryFile | None:
        if not self._loaded:
            self.load()
        return self._state

    def _require_writable(self) -> RegistryFile:
        state = self._ensure_loaded()
        if state is None:
            raise RegistryUnavailableError(
                "Prompt registry is corrupt or unreadable; refusing to write."
            )
        return state

    # -- read paths --------------------------------------------------------

    def get_promoted(self, name: str) -> PromptVersion | None:
        """Return the promoted artifact for `name`, or None on any fallback path."""
        state = self._ensure_loaded()
        if state is None:
            return None
        prompt = state.prompts.get(name)
        if prompt is None or prompt.promoted is None:
            return None
        try:
            return self.get_version(name, prompt.promoted)
        except PromptNotFoundError:
            # Promoted pointer references an unregistered version (tampered
            # state): fall back rather than raising into the caller.
            logger.warning(
                "Promoted version %s of %s is not registered; falling back.",
                prompt.promoted,
                name,
            )
            return None

    def get_version(self, name: str, version: str) -> PromptVersion | None:
        """Return one artifact, or None when its file is missing/corrupt.

        Raises ``PromptNotFoundError`` for names/versions absent from the
        registry and ``RegistryValidationError`` for malformed identifiers.
        """
        _require_valid_name(name)
        _require_valid_version(version)
        state = self._ensure_loaded()
        if state is None:
            return None
        prompt = state.prompts.get(name)
        if prompt is None:
            raise PromptNotFoundError(f"Unknown prompt: {name}")
        record = prompt.versions.get(version)
        if record is None:
            raise PromptNotFoundError(f"Unknown version: {name}/{version}")
        if (name, version) not in self._artifact_cache:
            self._artifact_cache[(name, version)] = self._read_artifact(
                name, version, record
            )
        return self._artifact_cache[(name, version)]

    def _read_artifact(
        self, name: str, version: str, record: VersionRecord
    ) -> PromptVersion | None:
        path = self._version_path(name, version)
        try:
            data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
            artifact = PromptVersion.model_validate(data)
            if artifact.name != name or artifact.version != version:
                raise ValueError("artifact identity does not match its path")
            if _sha256_text(artifact.template) != record.hash:
                raise ValueError("artifact template hash mismatch")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            logger.warning(
                "Prompt artifact %s/%s is unreadable (%s); falling back.",
                name,
                version,
                exc,
            )
            return None
        return artifact

    def list_prompts(self) -> dict[str, str | None]:
        """Map every registered prompt name to its promoted version (or None)."""
        state = self._ensure_loaded()
        if state is None:
            return {}
        return {name: prompt.promoted for name, prompt in state.prompts.items()}

    def list_versions(self, name: str) -> dict[str, VersionRecord]:
        """Return the per-version records for `name` (404 on unknown names)."""
        _require_valid_name(name)
        state = self._ensure_loaded()
        if state is None:
            return {}
        prompt = state.prompts.get(name)
        if prompt is None:
            raise PromptNotFoundError(f"Unknown prompt: {name}")
        return dict(prompt.versions)

    def diff(self, name: str, from_version: str, to_version: str) -> str:
        """Unified diff of two artifacts (metadata + template); "" when identical."""
        source = self.get_version(name, from_version)
        target = self.get_version(name, to_version)
        if source is None or target is None:
            raise PromptNotFoundError(
                f"Cannot diff {name}: artifact unreadable or unknown."
            )
        diff_lines = difflib.unified_diff(
            _artifact_diff_text(source).splitlines(),
            _artifact_diff_text(target).splitlines(),
            fromfile=f"{name}/{from_version}",
            tofile=f"{name}/{to_version}",
            lineterm="",
        )
        return "\n".join(diff_lines)

    # -- write paths ---------------------------------------------------------

    def create_candidate(
        self,
        name: str,
        template: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        author: str = "",
        notes: str = "",
        description: str = "",
    ) -> PromptVersion:
        """Write the next ``vN.yaml`` immutably and register it as a candidate."""
        with self._lock:
            _require_valid_name(name)
            self._validate_template(template)
            state = self._require_writable()
            prompt = state.prompts.setdefault(name, PromptState())
            version = self._next_version(name, prompt)
            artifact = PromptVersion(
                name=name,
                version=version,
                description=description,
                model=model,
                temperature=temperature,
                created_at=_utc_now(),
                author=author,
                notes=notes,
                template=template,
            )
            self._write_version_file(name, version, artifact)
            prompt.versions[version] = VersionRecord(
                status="candidate",
                hash=_sha256_text(template),
                created_at=artifact.created_at,
            )
            self._append_audit(
                action="create_candidate",
                name=name,
                version=version,
                actor=author or "unknown",
                reason=notes,
                metrics={"template_sha256": _sha256_text(template)},
            )
            self._write_state()
            return artifact

    def promote(
        self,
        name: str,
        version: str,
        actor: str,
        reason: str,
        override_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Promote a candidate: retire the current promoted version, move pointer."""
        with self._lock:
            _require_valid_name(name)
            _require_valid_version(version)
            state = self._require_writable()
            prompt = state.prompts.get(name)
            record = prompt.versions.get(version) if prompt else None
            if prompt is None or record is None:
                raise PromptNotFoundError(f"Unknown version: {name}/{version}")
            if record.status != "candidate":
                raise RegistryConflictError(
                    f"Only a candidate can be promoted; {name}/{version} is "
                    f"{record.status}."
                )
            metrics_payload = dict(metrics or {})
            if override_reason is not None:
                metrics_payload["override_reason"] = override_reason
            self._set_promoted(prompt, version)
            self._append_audit(
                action="promote",
                name=name,
                version=version,
                actor=actor,
                reason=reason,
                metrics=metrics_payload or None,
            )
            self._write_state()

    def rollback(self, name: str, to_version: str, actor: str, reason: str) -> None:
        """Roll back to a previously registered version; `reason` is mandatory."""
        with self._lock:
            _require_valid_name(name)
            _require_valid_version(to_version)
            if not reason.strip():
                raise RegistryValidationError("rollback requires a non-empty reason.")
            state = self._require_writable()
            prompt = state.prompts.get(name)
            record = prompt.versions.get(to_version) if prompt else None
            if prompt is None or record is None:
                raise PromptNotFoundError(f"Unknown version: {name}/{to_version}")
            if to_version == prompt.promoted:
                raise RegistryConflictError(f"{name}/{to_version} is already promoted.")
            self._set_promoted(prompt, to_version)
            self._append_audit(
                action="rollback",
                name=name,
                version=to_version,
                actor=actor,
                reason=reason,
                metrics=None,
            )
            self._write_state()

    @staticmethod
    def _set_promoted(prompt: PromptState, version: str) -> None:
        """Retire the current promoted version and move the pointer (1 promoted)."""
        if prompt.promoted is not None and prompt.promoted in prompt.versions:
            prompt.versions[prompt.promoted].status = "retired"
        prompt.versions[version].status = "promoted"
        prompt.promoted = version

    # -- audit chain ---------------------------------------------------------

    def _append_audit(
        self,
        action: AuditAction,
        name: str,
        version: str,
        actor: str,
        reason: str,
        metrics: dict[str, Any] | None,
    ) -> None:
        """Append a sealed audit entry. Callers must hold ``self._lock``."""
        state = self._require_writable()
        entry = AuditEntry(
            index=len(state.audit),
            ts=_utc_now(),
            action=action,
            name=name,
            version=version,
            actor=actor,
            reason=reason,
            metrics=metrics,
            prev_hash=state.audit[-1].hash if state.audit else GENESIS_PREV_HASH,
            hash="",
        )
        entry.hash = _seal_entry(entry)
        state.audit.append(entry)

    def verify_chain(self) -> AuditVerification:
        """Recompute the whole chain; report the first broken index, if any."""
        state = self._ensure_loaded()
        if state is None:
            return AuditVerification(
                valid=False, entries=0, audit_root=None, broken_at=None
            )
        audit_root = state.audit[-1].hash if state.audit else None
        prev_hash = GENESIS_PREV_HASH
        for index, entry in enumerate(state.audit):
            if (
                entry.index != index
                or entry.prev_hash != prev_hash
                or entry.hash != _seal_entry(entry)
            ):
                return AuditVerification(
                    valid=False,
                    entries=len(state.audit),
                    audit_root=audit_root,
                    broken_at=index,
                )
            prev_hash = entry.hash
        return AuditVerification(
            valid=True,
            entries=len(state.audit),
            audit_root=audit_root,
            broken_at=None,
        )

    # -- internals -----------------------------------------------------------

    def _validate_template(self, template: str) -> None:
        if not template.strip():
            raise RegistryValidationError("template must not be empty.")
        if len(template) > MAX_TEMPLATE_CHARS:
            raise RegistryValidationError(
                f"template exceeds {MAX_TEMPLATE_CHARS} characters."
            )
        if detect_prompt_injection(template) is not None:
            logger.warning("Rejected candidate template matching an injection pattern.")
            raise RegistryValidationError("template rejected by safety policy.")
        if any(pattern.search(template) for pattern in _SECRET_PATTERNS):
            logger.warning("Rejected candidate template containing a secret-like token.")
            raise RegistryValidationError("template appears to contain a secret.")

    def _version_path(self, name: str, version: str) -> Path:
        """Resolve the artifact path, refusing anything outside ``base_dir``."""
        resolved_base = self._base.resolve()
        path = (self._base / name / f"{version}.yaml").resolve()
        if not path.is_relative_to(resolved_base):
            raise RegistryValidationError("artifact path escapes the prompts directory.")
        return path

    def _next_version(self, name: str, prompt: PromptState) -> str:
        """Allocate ``v<max+1>`` across registry records and files on disk."""
        numbers = [int(version[1:]) for version in prompt.versions]
        prompt_dir = self._base / name
        if prompt_dir.is_dir():
            for file in prompt_dir.iterdir():
                match = _VERSION_FILE_PATTERN.fullmatch(file.name)
                if match:
                    numbers.append(int(match.group(1)))
        return f"v{max(numbers, default=0) + 1}"

    def _write_version_file(
        self, name: str, version: str, artifact: PromptVersion
    ) -> None:
        """Write an artifact immutably; an existing file is a conflict."""
        path = self._version_path(name, version)
        if path.exists():
            raise RegistryConflictError(
                f"Prompt artifact {name}/{version} already exists and is immutable."
            )
        payload = yaml.safe_dump(
            artifact.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        )
        self._write_atomic(path, payload)

    def _write_state(self) -> None:
        """Atomically rewrite ``registry.json`` and drop the artifact cache."""
        state = self._require_writable()
        payload = json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True)
        self._write_atomic(self._registry_path, payload + "\n")
        self._artifact_cache.clear()

    def _write_atomic(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
