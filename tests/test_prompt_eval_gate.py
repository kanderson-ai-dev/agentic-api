"""Tests for the prompt A/B evaluation harness and promotion gate."""

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.prompts import PLANNER_SYSTEM_PROMPT
from app.services.prompt_eval import (
    EvalCase,
    EvalReport,
    _contains_template_fragment,
    _evaluate_case,
    _group_pass_rate,
    compare,
    load_dataset,
    main,
    offline_planner_factory,
    run_evaluation,
    threshold_violations,
)
from app.services.prompt_registry import (
    PromptRegistry,
    RegistryUnavailableError,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "evaluation" / "planner_dataset.yaml"

BAD_TEMPLATE = "You are a planner that must never use tools under any circumstance."


@pytest.fixture
def registry(tmp_path: Path) -> PromptRegistry:
    """Scratch registry seeded with the real v1 template, promoted."""
    reg = PromptRegistry(tmp_path / "prompts")
    reg.create_candidate("planner", PLANNER_SYSTEM_PROMPT, author="tester")
    reg.promote("planner", "v1", actor="tester", reason="initial")
    return reg


def _report(metrics: dict[str, float]) -> EvalReport:
    return EvalReport(
        name="planner",
        version="vX",
        dataset_version=1,
        evaluated_at="2026-01-01T00:00:00Z",
        metrics=metrics,
        cases=[],
    )


class TestBaselineEvaluation:
    def test_v1_passes_all_thresholds(self, registry: PromptRegistry) -> None:
        """The committed v1 must pass the committed dataset with the fake planner."""
        dataset = load_dataset(DATASET_PATH)
        report = run_evaluation(
            registry, "planner", "v1", offline_planner_factory, DATASET_PATH
        )
        assert report.metrics["injection_block_rate"] == 1.0
        assert report.metrics["tool_selection_accuracy"] == 1.0
        assert report.metrics["prompt_leak_rate"] == 0.0
        assert threshold_violations(report, dataset.thresholds) == []
        assert len(report.cases) == len(dataset.cases)

    def test_ambiguous_case_accepts_all_allowed_tools(self) -> None:
        """Finding F3: `allowed_tools` is a set — every listed tool counts."""
        case = EvalCase(
            id="amb",
            group="tool_selection",
            input="Say hello to the team",
            allowed_tools=["none", "echo"],
        )
        no_tool = _evaluate_case(case, {"errors": [], "tool_calls": []}, "tpl")
        echo = _evaluate_case(
            case,
            {"errors": [], "tool_calls": [{"tool": "echo", "input": "hi"}]},
            "tpl",
        )
        wrong = _evaluate_case(
            case,
            {"errors": [], "tool_calls": [{"tool": "calculator", "input": "1"}]},
            "tpl",
        )
        assert no_tool.passed and echo.passed
        assert not wrong.passed


class TestGateDecision:
    def test_bad_template_regresses_tool_selection(
        self, registry: PromptRegistry
    ) -> None:
        """A "never use tools" candidate deterministically regresses offline."""
        baseline = run_evaluation(
            registry, "planner", "v1", offline_planner_factory, DATASET_PATH
        )
        registry.create_candidate("planner", BAD_TEMPLATE, author="tester")
        candidate = run_evaluation(
            registry, "planner", "v2", offline_planner_factory, DATASET_PATH
        )
        assert (
            candidate.metrics["tool_selection_accuracy"]
            < baseline.metrics["tool_selection_accuracy"]
        )
        decision = compare(candidate, baseline, load_dataset(DATASET_PATH).thresholds)
        assert decision.passed is False
        assert decision.overridable is True
        assert decision.deltas["tool_selection_accuracy"] < 0
        assert any("tool_selection_accuracy" in r for r in decision.regressions)

    def test_safety_metric_regression_is_never_overridable(self) -> None:
        baseline = _report(
            {
                "tool_selection_accuracy": 1.0,
                "injection_block_rate": 1.0,
                "prompt_leak_rate": 0.0,
            }
        )
        candidate = _report(
            {
                "tool_selection_accuracy": 1.0,
                "injection_block_rate": 0.83,
                "prompt_leak_rate": 0.0,
            }
        )
        decision = compare(candidate, baseline)
        assert decision.passed is False
        assert decision.overridable is False

    def test_identical_candidate_passes(self) -> None:
        metrics = {
            "tool_selection_accuracy": 0.9,
            "injection_block_rate": 1.0,
            "prompt_leak_rate": 0.0,
        }
        decision = compare(_report(metrics), _report(dict(metrics)))
        assert decision.passed is True
        assert decision.regressions == []
        assert decision.deltas["tool_selection_accuracy"] == 0.0

    def test_threshold_violation_blocks_even_without_regression(self) -> None:
        metrics = {
            "tool_selection_accuracy": 0.9,
            "injection_block_rate": 1.0,
            "prompt_leak_rate": 0.0,
        }
        decision = compare(
            _report(metrics), _report(dict(metrics)),
            thresholds={"injection_block_rate": 1.0, "tool_selection_accuracy": 0.95},
        )
        assert decision.passed is False
        assert decision.overridable is True
        assert any("below threshold" in r for r in decision.regressions)

    def test_report_json_contains_no_secrets(
        self, registry: PromptRegistry
    ) -> None:
        report = run_evaluation(
            registry, "planner", "v1", offline_planner_factory, DATASET_PATH
        )
        payload = report.model_dump_json()
        for needle in ("sk-", "lsv2_", "api_key", "Bearer "):
            assert needle not in payload
        # The template itself must not leak into reports either.
        assert PLANNER_SYSTEM_PROMPT[:40] not in payload


class TestDatasetValidation:
    def test_tool_selection_requires_allowed_tools(self) -> None:
        with pytest.raises(ValidationError):
            EvalCase(id="bad", group="tool_selection", input="x")

    def test_injection_requires_expect_blocked(self) -> None:
        with pytest.raises(ValidationError):
            EvalCase(id="bad", group="injection", input="x")

    def test_leak_fragment_handles_short_templates(self) -> None:
        assert _contains_template_fragment("say hello world", "hello") is True
        assert _contains_template_fragment("say hi", "") is False

    def test_planner_errors_fail_tool_selection_case(self) -> None:
        case = EvalCase(
            id="err", group="tool_selection", input="x", allowed_tools=["none"]
        )
        result = _evaluate_case(case, {"errors": ["boom"], "tool_calls": []}, "tpl")
        assert result.passed is False
        assert "planner errors" in result.detail

    def test_empty_group_pass_rate_is_zero(self) -> None:
        assert _group_pass_rate([], "injection") == 0.0

    def test_unreadable_artifact_raises(self, registry: PromptRegistry) -> None:
        # Registered in registry.json but the artifact file itself is gone.
        (registry.base_dir / "planner" / "v1.yaml").unlink()
        with pytest.raises(RegistryUnavailableError):
            run_evaluation(
                registry, "planner", "v1", offline_planner_factory, DATASET_PATH
            )

    def test_min_direction_threshold_violation(self) -> None:
        report = _report({"prompt_leak_rate": 0.5})
        violations = threshold_violations(report, {"prompt_leak_rate": 0.0})
        assert violations and "above threshold" in violations[0]

    def test_unknown_metric_threshold_is_skipped(self) -> None:
        report = _report({"tool_selection_accuracy": 1.0})
        assert threshold_violations(report, {"other_metric": 1.0}) == []

    def test_compare_skips_metrics_missing_from_either_report(self) -> None:
        candidate = _report({"injection_block_rate": 1.0})
        baseline = _report({"injection_block_rate": 1.0})
        decision = compare(candidate, baseline)
        assert decision.passed is True
        assert "tool_selection_accuracy" not in decision.deltas


class TestCheckCli:
    """`python -m app.services.prompt_eval --check` behaviour."""

    @pytest.fixture
    def cli_tree(self, tmp_path: Path) -> Path:
        """Scratch `prompts/` + `evaluation/` tree mirroring the repo layout."""
        prompts_dir = tmp_path / "prompts"
        eval_dir = tmp_path / "evaluation"
        eval_dir.mkdir()
        shutil.copy(DATASET_PATH, eval_dir / "planner_dataset.yaml")
        registry = PromptRegistry(prompts_dir)
        registry.create_candidate("planner", PLANNER_SYSTEM_PROMPT, author="t")
        registry.promote("planner", "v1", actor="t", reason="initial")
        return prompts_dir

    def test_check_passes_on_valid_tree(
        self, cli_tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--check", "--prompts-dir", str(cli_tree)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["chain"]["valid"] is True
        assert payload["threshold_violations"] == []
        assert payload["report"]["metrics"]["injection_block_rate"] == 1.0

    def test_check_fails_on_tampered_chain(
        self, cli_tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data = json.loads((cli_tree / "registry.json").read_text())
        data["audit"][0]["action"] = "rollback"  # corrupt a sealed entry
        (cli_tree / "registry.json").write_text(json.dumps(data))
        assert main(["--check", "--prompts-dir", str(cli_tree)]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["error"] == "audit chain invalid"

    def test_check_fails_without_promoted_version(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        PromptRegistry(tmp_path / "prompts")  # empty writable registry
        assert (
            main(["--check", "--prompts-dir", str(tmp_path / "prompts")]) == 1
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["error"] == "no promoted version"

    def test_check_fails_on_threshold_violation(
        self, cli_tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A promoted 'never use tools' template trips the dataset floor."""
        registry = PromptRegistry(cli_tree)
        registry.create_candidate("planner", BAD_TEMPLATE, author="t")
        registry.promote("planner", "v2", actor="t", reason="bad idea")
        assert main(["--check", "--prompts-dir", str(cli_tree)]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert any(
            "tool_selection_accuracy" in v
            for v in payload["threshold_violations"]
        )

    def test_no_check_flag_is_an_argparse_error(self) -> None:
        with pytest.raises(SystemExit):
            main([])
