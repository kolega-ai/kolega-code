"""Developer entry point: ``python -m benchmarks.edit_tools``."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

import yaml

from kolega_code.cli.provider_registry import default_model_for_provider
from kolega_code.config import ModelProvider
from kolega_code.llm.specs import MODEL_SPECS

from .artifacts import load_trial_records, write_json, write_materialized_cases
from .corpus import load_suite
from .models import FileContent, MatrixSpec, ModelRunSpec, SuiteSpec, TaskSpec
from .protocols import PROTOCOLS
from .report import write_report
from .runner import plan_trials, run_benchmark


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_OUTPUT = REPO_ROOT / ".benchmark-runs"
DEFAULT_SUITE = PACKAGE_ROOT / "suites" / "core.yaml"


def _load_matrix(path: Path) -> MatrixSpec:
    return MatrixSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _csv_set(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    result: set[str] = set()
    for value in values:
        result.update(item.strip() for item in value.split(",") if item.strip())
    return result or None


def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", action="append", help="Task id filter; repeat or comma-separate.")
    parser.add_argument("--tag", action="append", help="Task tag filter; repeat or comma-separate.")
    parser.add_argument("--provider", action="append", help="Provider filter; repeat or comma-separate.")
    parser.add_argument("--protocol", action="append", help="Protocol filter; repeat or comma-separate.")
    parser.add_argument(
        "--lane", action="append", choices=["controlled", "coder_agent"], help="Lane filter; repeat as needed."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.edit_tools",
        description="Repository-only model/edit-protocol benchmark harness.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate suites, matrices, protocols, and generated cases.")
    validate.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    validate.add_argument("--matrix", type=Path)

    generate = commands.add_parser("generate", help="Materialize a suite without making model calls.")
    generate.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    generate.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run", help="Run or resume a benchmark matrix.")
    run.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    run.add_argument("--matrix", type=Path, required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--resume", type=Path)
    run.add_argument("--max-trials", type=int)
    run.add_argument("--rerun-infrastructure-failures", action="store_true")
    run.add_argument("--confirm-live", action="store_true", help="Required before calls to paid/live model APIs.")
    run.add_argument("--dry-run", action="store_true")
    _add_filters(run)

    report = commands.add_parser("report", help="Rebuild summaries from a run's raw trial journal.")
    report.add_argument("run_dir", type=Path)

    smoke = commands.add_parser("provider-smoke", help="Run one task/protocol across every catalog provider.")
    smoke.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    smoke.add_argument("--confirm-live", action="store_true")
    smoke.add_argument("--require-complete", action="store_true")
    smoke.add_argument("--resume", type=Path)
    smoke.add_argument("--rerun-infrastructure-failures", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> int:
    suite, tasks = load_suite(args.suite)
    matrix = _load_matrix(args.matrix) if args.matrix else None
    print(f"suite={suite.id} curated={len(suite.curated_tasks)} total={len(tasks)} digest-ready=yes")
    print(f"protocols={','.join(sorted(PROTOCOLS))}")
    if matrix is not None:
        trials = plan_trials(suite, tasks, matrix)
        print(f"matrix={matrix.id} models={sum(item.enabled for item in matrix.models)} trials={len(trials)}")
    return 0


def _generate(args: argparse.Namespace) -> int:
    suite, tasks = load_suite(args.suite)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "suite.json", suite.model_dump(mode="json"))
    write_materialized_cases(args.output, tasks)
    print(f"Materialized {len(tasks)} tasks in {args.output}")
    return 0


def _filters(args: argparse.Namespace) -> dict:
    return {
        "task_ids": _csv_set(getattr(args, "task", None)),
        "tags": _csv_set(getattr(args, "tag", None)),
        "providers": _csv_set(getattr(args, "provider", None)),
        "protocols": _csv_set(getattr(args, "protocol", None)),
        "lanes": _csv_set(getattr(args, "lane", None)),
    }


async def _run(args: argparse.Namespace) -> int:
    suite, tasks = load_suite(args.suite)
    matrix = _load_matrix(args.matrix)
    planned = plan_trials(suite, tasks, matrix, **_filters(args))
    if args.max_trials is not None:
        planned = planned[: args.max_trials]
    print(
        f"Planned {len(planned)} live trial(s) across "
        f"{len({(item.model.provider, item.model.model) for item in planned})} model(s)."
    )
    if args.dry_run:
        for item in planned:
            print(
                f"{item.trial_id} {item.model.provider}/{item.model.model} {item.protocol} "
                f"{item.lane} {item.task.id} r{item.repetition}"
            )
        return 0
    if not args.confirm_live:
        print("Refusing live model calls without --confirm-live. Use --dry-run to inspect the matrix.", file=sys.stderr)
        return 2
    run_dir, records = await run_benchmark(
        repo_root=REPO_ROOT,
        suite=suite,
        tasks=tasks,
        matrix=matrix,
        output_root=args.output_root,
        resume_dir=args.resume,
        rerun_infrastructure_failures=args.rerun_infrastructure_failures,
        max_trials=args.max_trials,
        **_filters(args),
    )
    write_report(run_dir, records)
    print(f"Run artifacts: {run_dir}")
    return 0


def _catalog_smoke_matrix() -> MatrixSpec:
    providers: list[str] = []
    for provider, _ in MODEL_SPECS:
        if provider not in providers:
            providers.append(provider)
    models: list[ModelRunSpec] = []
    for provider_value in providers:
        provider = ModelProvider(provider_value)
        model = "gpt-oss:20b" if provider == ModelProvider.OLLAMA_CLOUD else default_model_for_provider(provider)
        models.append(
            ModelRunSpec(
                provider=provider_value,
                model=model,
                protocols=["search_replace", "codex_apply_patch"],
                max_output_tokens=4096,
            )
        )
    return MatrixSpec(
        id="all-catalog-providers-smoke",
        models=models,
        lanes=["controlled"],
        repetitions=1,
        trial_timeout_seconds=480,
    )


def _provider_smoke_suite() -> tuple[SuiteSpec, list[TaskSpec]]:
    task = TaskSpec(
        id="provider-create-smoke",
        prompt="Create `provider-smoke.txt` containing exactly `provider-ok` followed by a newline.",
        before_files={"README.md": FileContent(text="# Provider smoke\n")},
        expected_files={
            "README.md": FileContent(text="# Provider smoke\n"),
            "provider-smoke.txt": FileContent(text="provider-ok\n"),
        },
        tags=["smoke", "create-file"],
        required_capabilities={"create"},
    )
    suite = SuiteSpec(
        id="provider-smoke",
        description="One minimal edit through every catalog provider and production protocol.",
        curated_tasks=[task],
        default_repetitions=1,
    )
    return suite, [task]


async def _provider_smoke(args: argparse.Namespace) -> int:
    if not args.confirm_live:
        print("Refusing live provider smoke calls without --confirm-live.", file=sys.stderr)
        return 2
    suite, tasks = _provider_smoke_suite()
    matrix = _catalog_smoke_matrix()
    run_dir, records = await run_benchmark(
        repo_root=REPO_ROOT,
        suite=suite,
        tasks=tasks,
        matrix=matrix,
        output_root=args.output_root,
        resume_dir=args.resume,
        rerun_infrastructure_failures=args.rerun_infrastructure_failures,
    )
    write_report(run_dir, records)
    expected = len(matrix.models) * 2
    passed = sum(record.status == "passed" for record in records)
    incomplete = [record for record in records if record.status != "passed"]
    print(f"Provider smoke: {passed}/{expected} passed. Artifacts: {run_dir}")
    if args.require_complete and (len(records) != expected or incomplete):
        for record in incomplete:
            print(f"{record.provider}/{record.model} {record.protocol}: {record.status} {record.error or ''}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        return _validate(args)
    if args.command == "generate":
        return _generate(args)
    if args.command == "run":
        return asyncio.run(_run(args))
    if args.command == "report":
        records = load_trial_records(args.run_dir / "trials.jsonl")
        write_report(args.run_dir, records)
        print(f"Wrote {args.run_dir / 'summary.md'}")
        return 0
    if args.command == "provider-smoke":
        return asyncio.run(_provider_smoke(args))
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
