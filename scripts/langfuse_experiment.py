#!/usr/bin/env python
"""Run the offline evaluation dataset as a real Langfuse experiment.

Manual, opt-in only — never wired into CI. Uploads the versioned gate dataset
(`evaluation/planner_dataset.yaml`) to Langfuse, runs every case through the
real agent graph (real `gpt-4o-mini`, real Langfuse traces), and scores each
item with the *same* functions the promotion gate uses
(`app.services.prompt_eval._evaluate_case` / `_compute_metrics`), so the
experiment numbers can never drift from `prompt_eval --check`.

Prompt-text protection is identical to serving: the client's `mask` replaces
every served template with its version/hash placeholder before export, and
dataset items carry only the case input and expectations — never a template.

Usage (real keys required):

    python scripts/langfuse_experiment.py                 # promoted version
    python scripts/langfuse_experiment.py --version v2    # specific artifact

Runs are named ``planner@vN-YYYYMMDD`` so a candidate and the promoted version
can be compared side by side in the Langfuse Experiments view.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.graph import (  # noqa: E402
    PlannerOverride,
    build_graph,
    planner_override,
)
from app.core.config import get_settings  # noqa: E402
from app.core.tracing import (  # noqa: E402
    _get_or_create_client,
    langfuse_enabled,
    register_prompt_template,
)
from app.services.prompt_eval import (  # noqa: E402
    METRIC_INJECTION,
    METRIC_TOOL_SELECTION,
    CaseResult,
    _compute_metrics,
    _evaluate_case,
    _initial_state,
    _sha256_text,
    default_dataset_path,
    load_dataset,
)
from app.services.prompt_registry import (  # noqa: E402
    PROMPT_HASH_LENGTH,
    PromptRegistry,
    ResolvedPrompt,
)

# Per-case score names in Langfuse match the gate's metric names exactly.
_GROUP_SCORE_NAME = {
    "tool_selection": METRIC_TOOL_SELECTION,
    "injection": METRIC_INJECTION,
    "prompt_leak": "prompt_leak_clean",  # gate metric is the *rate*; this is per-case
}


def _resolve_artifact(registry: PromptRegistry, name: str, version: str | None) -> ResolvedPrompt:
    """Resolve an explicit version or the promoted one into a `ResolvedPrompt`."""
    artifact = (
        registry.get_version(name, version)
        if version
        else registry.get_promoted(name)
    )
    if artifact is None:
        raise SystemExit(
            f"prompt {name}/{version or 'promoted'} not found in {registry.base_dir}"
        )
    return ResolvedPrompt(
        name=name,
        version=artifact.version,
        template=artifact.template,
        model=artifact.model,
        temperature=artifact.temperature,
        content_hash=_sha256_text(artifact.template)[:PROMPT_HASH_LENGTH],
    )


def _sync_dataset(client: Any, dataset_name: str, cases: Any, dataset_version: int) -> Any:
    """Create the Langfuse dataset if needed and insert any missing items."""
    from langfuse.api.commons.errors import NotFoundError

    try:
        dataset = client.get_dataset(dataset_name)
    except NotFoundError:
        client.create_dataset(
            name=dataset_name,
            description="Offline promotion-gate dataset mirrored for live experiments.",
            metadata={"dataset_version": dataset_version},
        )
        dataset = client.get_dataset(dataset_name)
    existing = {
        (item.metadata or {}).get("case_id") for item in dataset.items
    }
    added = 0
    for case in cases:
        if case.id in existing:
            continue
        client.create_dataset_item(
            dataset_name=dataset_name,
            input=case.input,
            expected_output={
                "allowed_tools": case.allowed_tools,
                "expect_blocked": case.expect_blocked,
            },
            metadata={
                "case_id": case.id,
                "group": case.group,
                "dataset_version": dataset_version,
            },
        )
        added += 1
    print(f"dataset '{dataset_name}': {len(dataset.items) + added} items ({added} added)")
    return client.get_dataset(dataset_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="planner", help="prompt name (default: planner)")
    parser.add_argument(
        "--version",
        default=None,
        help="prompt version to evaluate (default: the promoted version)",
    )
    parser.add_argument(
        "--dataset",
        default="planner_gate_eval",
        help="Langfuse dataset name (default: planner_gate_eval)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="parallel experiment items"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    missing = []
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not langfuse_enabled(settings):
        missing.append("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    if missing:
        print(
            "error: missing required keys: "
            + ", ".join(missing)
            + " — set them in .env; this experiment makes real OpenAI calls "
            "and uploads traces to Langfuse.",
            file=sys.stderr,
        )
        return 2

    from langfuse.experiment import Evaluation
    from langfuse.langchain import CallbackHandler

    registry = PromptRegistry(settings.prompts_dir)
    registry.load()
    resolved = _resolve_artifact(registry, args.name, args.version)
    dataset_file = default_dataset_path(registry.base_dir, args.name)
    dataset = load_dataset(dataset_file)
    cases_by_id = {case.id: case for case in dataset.cases}

    client = _get_or_create_client(settings)  # masked client, same as serving
    register_prompt_template(resolved.name, resolved.version, resolved.template)
    lf_dataset = _sync_dataset(client, args.dataset, dataset.cases, dataset.dataset_version)

    run_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_name = f"{args.name}@{resolved.version}-{run_date}"
    session_id = f"experiment-{run_name}"
    graph = build_graph()
    handler = CallbackHandler(public_key=settings.langfuse_public_key)
    case_results: list[CaseResult] = []

    def task(*, item: Any, **_kwargs: Any) -> dict[str, object]:
        """Run one dataset case through the real graph with tracing attached."""
        config = {
            "callbacks": [handler],
            "tags": ["experiment", f"prompt:{resolved.name}@{resolved.version}"],
            "metadata": {
                "prompt_version": resolved.version,
                "prompt_hash": resolved.content_hash,
                "experiment_item": (item.metadata or {}).get("case_id", ""),
                "langfuse_session_id": session_id,
                "langfuse_tags": [
                    "experiment",
                    f"prompt:{resolved.name}@{resolved.version}",
                ],
            },
        }
        with planner_override(PlannerOverride(prompt=resolved, llm_factory=None)):
            return dict(graph.invoke(_initial_state(item.input), config=config))

    def gate_evaluator(
        *,
        input: Any,
        output: Any,
        expected_output: Any,
        metadata: Any,
        **_kwargs: Any,
    ) -> list:
        """Score one item with the gate's `_evaluate_case` (no reimplementation)."""
        case = cases_by_id[(metadata or {}).get("case_id", "")]
        result = _evaluate_case(
            case, output if isinstance(output, dict) else {}, resolved.template
        )
        case_results.append(result)
        return [
            Evaluation(
                name=_GROUP_SCORE_NAME[case.group],
                value=result.passed,
                data_type="BOOLEAN",
                comment=result.detail,
            ),
            Evaluation(
                name="case_pass", value=result.passed, data_type="BOOLEAN"
            ),
        ]

    def run_gate_metrics(*, item_results: Any, **_kwargs: Any) -> list:
        """Aggregate the run with the gate's `_compute_metrics`."""
        return [
            Evaluation(name=metric, value=value, data_type="NUMERIC")
            for metric, value in _compute_metrics(case_results).items()
        ]

    # Same offline-by-construction rule as the gate: tool *selection* is scored,
    # so web_search never makes outbound calls during an experiment.
    previous_provider = settings.search_provider
    settings.search_provider = "none"
    try:
        result = lf_dataset.run_experiment(
            name=run_name,
            run_name=run_name,
            description=(
                f"Live gate evaluation of {args.name}@{resolved.version} "
                f"(dataset v{dataset.dataset_version})"
            ),
            task=task,
            evaluators=[gate_evaluator],
            run_evaluators=[run_gate_metrics],
            max_concurrency=args.concurrency,
            metadata={
                "prompt_name": resolved.name,
                "prompt_version": resolved.version,
                "prompt_hash": resolved.content_hash,
                "dataset_version": str(dataset.dataset_version),
                "source": "scripts/langfuse_experiment.py",
            },
        )
    finally:
        settings.search_provider = previous_provider
        client.flush()

    processed = len(getattr(result, "item_results", []) or case_results)
    metrics = _compute_metrics(case_results) if case_results else {}
    print(f"\nexperiment run: {run_name}")
    print(f"items processed: {processed}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}")
    failed = [r for r in case_results if not r.passed]
    if failed:
        print(f"failed cases: {[r.id for r in failed]}")
    print("view it under Experiments in your Langfuse project.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
