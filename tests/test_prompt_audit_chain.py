"""Tests for the hash-chained audit log inside the prompt registry."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.services.prompt_registry import (
    GENESIS_PREV_HASH,
    AuditEntry,
    PromptRegistry,
    RegistryFile,
)


def _bootstrap(registry: PromptRegistry) -> None:
    registry.create_candidate("planner", "Template v1.", author="tester")
    registry.promote("planner", "v1", actor="tester", reason="initial")


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    return tmp_path / "prompts"


@pytest.fixture
def registry(prompts_dir: Path) -> PromptRegistry:
    reg = PromptRegistry(prompts_dir)
    _bootstrap(reg)
    return reg


def _rewrite_registry(
    prompts_dir: Path, mutate: Callable[[dict[str, Any]], None]
) -> PromptRegistry:
    """Apply `mutate` to the on-disk registry and return a freshly loaded one."""
    path = prompts_dir / "registry.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    registry = PromptRegistry(prompts_dir)
    registry.load()
    return registry


class TestChainIntegrity:
    def test_genesis_entry_links_to_zero_hash(self, registry: PromptRegistry) -> None:
        state = RegistryFile.model_validate(
            json.loads((registry._registry_path).read_text(encoding="utf-8"))
        )
        genesis = state.audit[0]
        assert genesis.index == 0
        assert genesis.prev_hash == GENESIS_PREV_HASH
        assert genesis.action == "create_candidate"

    def test_chain_links_and_audit_root(self, registry: PromptRegistry) -> None:
        result = registry.verify_chain()
        assert result.valid is True
        assert result.entries == 2
        state = RegistryFile.model_validate(
            json.loads((registry._registry_path).read_text(encoding="utf-8"))
        )
        assert state.audit[1].prev_hash == state.audit[0].hash
        assert result.audit_root == state.audit[-1].hash

    def test_mutating_a_sealed_field_breaks_chain(
        self, registry: PromptRegistry, prompts_dir: Path
    ) -> None:
        def mutate(data: dict) -> None:
            data["audit"][1]["reason"] = "tampered reason"

        reloaded = _rewrite_registry(prompts_dir, mutate)
        result = reloaded.verify_chain()
        assert result.valid is False
        assert result.broken_at == 1

    def test_deleted_entry_is_detected(
        self, registry: PromptRegistry, prompts_dir: Path
    ) -> None:
        def mutate(data: dict) -> None:
            del data["audit"][0]

        reloaded = _rewrite_registry(prompts_dir, mutate)
        result = reloaded.verify_chain()
        assert result.valid is False
        assert result.broken_at == 0

    def test_reordered_entries_are_detected(
        self, registry: PromptRegistry, prompts_dir: Path
    ) -> None:
        registry.create_candidate("planner", "Template v2.", author="tester")

        def mutate(data: dict) -> None:
            data["audit"] = [data["audit"][1], data["audit"][0], data["audit"][2]]

        reloaded = _rewrite_registry(prompts_dir, mutate)
        result = reloaded.verify_chain()
        assert result.valid is False
        assert result.broken_at == 0

    def test_pointer_tampering_is_outside_chain_but_safe(
        self, registry: PromptRegistry, prompts_dir: Path
    ) -> None:
        """The chain seals audit entries, not the promoted pointer; a pointer
        aimed at an unregistered version still verifies but falls back."""

        def mutate(data: dict) -> None:
            data["prompts"]["planner"]["promoted"] = "v9"

        reloaded = _rewrite_registry(prompts_dir, mutate)
        assert reloaded.verify_chain().valid is True
        assert reloaded.get_promoted("planner") is None


class TestAuditReasons:
    def test_override_reason_is_sealed(self, registry: PromptRegistry) -> None:
        registry.create_candidate("planner", "Template v2.", author="tester")
        registry.promote(
            "planner",
            "v2",
            actor="tester",
            reason="promote despite regression",
            override_reason="manual override accepted by operator",
            metrics={"tool_selection_accuracy": -0.05},
        )
        entry = registry._require_writable().audit[-1]
        assert entry.action == "promote"
        assert entry.reason == "promote despite regression"
        assert entry.metrics is not None
        assert entry.metrics["override_reason"] == (
            "manual override accepted by operator"
        )
        assert registry.verify_chain().valid is True

    def test_rollback_entry_carries_reason(self, registry: PromptRegistry) -> None:
        registry.create_candidate("planner", "Template v2.", author="tester")
        registry.promote("planner", "v2", actor="tester", reason="promote v2")
        registry.rollback(
            "planner", "v1", actor="tester", reason="v2 regressed in production"
        )
        entry = registry._require_writable().audit[-1]
        assert isinstance(entry, AuditEntry)
        assert entry.action == "rollback"
        assert entry.version == "v1"
        assert entry.reason == "v2 regressed in production"
        assert registry.verify_chain().valid is True
