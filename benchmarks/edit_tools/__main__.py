"""Developer entry point: ``python -m benchmarks.edit_tools``."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path
import sys
import tempfile
import webbrowser

import yaml

from kolega_code.cli.provider_registry import default_model_for_provider
from kolega_code.config import ModelProvider
from kolega_code.agent.edit_protocols import production_edit_protocols
from kolega_code.llm.specs import MODEL_SPECS

from .artifacts import load_trial_records, write_json, write_materialized_cases
from .authoring import FAMILY_COUNTS, author_corpus, import_sources
from .browser import build_corpus_browser
from .corpus import load_suite
from .models import FileContent, MatrixSpec, ModelRunSpec, SuiteSpec, TaskSpec
from .protocols import PROTOCOLS
from .report import write_report
from .runner import plan_trials, run_benchmark
from .workspace import materialize_task, materialize_tree, verify_task


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_OUTPUT = REPO_ROOT / ".benchmark-runs"
DEFAULT_SUITE = PACKAGE_ROOT / "suites" / "core.yaml"
DEFAULT_BROWSER_OUTPUT = REPO_ROOT / ".corpus-builds" / "browser"


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
    validate.add_argument(
        "--verify-oracles",
        action="store_true",
        help="Execute every before/expected oracle; container commands require the pinned verifier image.",
    )

    generate = commands.add_parser("generate", help="Materialize a suite without making model calls.")
    generate.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    generate.add_argument("--output", type=Path, required=True)

    browse = commands.add_parser("browse", help="Build and open a static browser for the benchmark corpus.")
    browse.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    browse.add_argument("--output", type=Path, default=DEFAULT_BROWSER_OUTPUT)
    browse.add_argument("--no-open", action="store_true", help="Build the site without opening a browser.")

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

    corpus_import = commands.add_parser(
        "corpus-import", help="Import pinned public source files and create the 100 mechanical authoring slots."
    )
    corpus_import.add_argument("--config", type=Path, default=PACKAGE_ROOT / "fixtures" / "sources.yaml")
    corpus_import.add_argument("--checkout-root", type=Path, required=True)

    corpus_author = commands.add_parser(
        "corpus-author", help="Author exact mechanical edit recipes with Opus-first validated fallback."
    )
    corpus_author.add_argument("--slots", type=Path, default=PACKAGE_ROOT / "fixtures" / "authoring-slots.yaml")
    corpus_author.add_argument("--output", type=Path, default=PACKAGE_ROOT / "suites" / "corpora" / "edit-core.yaml")
    corpus_author.add_argument("--artifact-root", type=Path, default=REPO_ROOT / ".corpus-builds" / "mechanical-v1")
    corpus_author.add_argument("--concurrency", type=int, default=2)
    corpus_author.add_argument("--task", action="append", help="Author only these task ids; repeat or comma-separate.")
    corpus_author.add_argument("--max-tasks", type=int)
    corpus_author.add_argument("--confirm-live", action="store_true")
    return parser


async def _validate_oracles(tasks: list[TaskSpec]) -> None:
    failures: list[str] = []
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix=f"kolega-corpus-{task.id}-") as temporary:
            root = Path(temporary)
            before = root / "before"
            expected = root / "expected"
            materialize_task(before, task)
            materialize_tree(expected, task.expected_files)
            before_result = await verify_task(before, task)
            expected_result = await verify_task(expected, task)
            if before_result.infrastructure_error:
                failures.append(f"{task.id}: {before_result.infrastructure_error}")
            elif before_result.success:
                failures.append(f"{task.id}: before workspace unexpectedly passes")
            if expected_result.infrastructure_error:
                failures.append(f"{task.id}: {expected_result.infrastructure_error}")
            elif not expected_result.success:
                failures.append(f"{task.id}: expected workspace fails")
    if failures:
        raise ValueError("corpus oracle validation failed:\n" + "\n".join(failures))


def _validate(args: argparse.Namespace) -> int:
    suite, tasks = load_suite(args.suite)
    matrix = _load_matrix(args.matrix) if args.matrix else None
    print(f"suite={suite.id} curated={len(suite.curated_tasks)} total={len(tasks)} digest-ready=yes")
    print(
        "coverage="
        f"languages:{len({task.language for task in tasks})},"
        f"mechanical:{sum(task.shape == 'mechanical' for task in tasks)},"
        f"snapshots:{len({task.snapshot_id for task in tasks if task.snapshot_id})},"
        f"recipes:{sum(task.recipe is not None for task in tasks)}"
    )
    if any(task.id.startswith("synthetic-") for task in tasks):
        missing = [task.id for task in tasks if None in (task.language, task.family, task.difficulty, task.shape)]
        if missing:
            raise ValueError(f"tasks missing benchmark classification: {', '.join(missing)}")
    if suite.id == "edit-core":
        language_counts = Counter(task.language for task in tasks)
        expected_languages = {
            "python": 9,
            "typescript": 9,
            "javascript": 8,
            "go": 9,
            "rust": 9,
            "java": 8,
            "cpp": 8,
            "csharp": 8,
            "ruby": 8,
            "php": 8,
            "swift": 8,
            "kotlin": 8,
        }
        length_counts = Counter(task.target_length for task in tasks)
        expected_lengths = {"short": 10, "normal": 25, "medium": 35, "long": 20, "oversized": 10}
        family_counts = Counter(task.family for task in tasks)
        operation_counts = Counter(_count_group(len(task.recipe.operations) if task.recipe else 0) for task in tasks)
        file_counts = Counter(
            _file_count_group(len({operation.path for operation in task.recipe.operations}) if task.recipe else 0)
            for task in tasks
        )
        errors: list[str] = []
        if len(tasks) != 100:
            errors.append(f"total={len(tasks)} expected=100")
        for name, actual, expected in (
            ("languages", language_counts, expected_languages),
            ("target lengths", length_counts, expected_lengths),
            ("families", family_counts, FAMILY_COUNTS),
            ("operation scopes", operation_counts, {"one": 25, "few": 35, "several": 30, "many": 10}),
            ("file scopes", file_counts, {"one": 50, "few": 35, "many": 15}),
        ):
            if actual != expected:
                errors.append(f"{name}={actual} expected={expected}")
        if len({(task.snapshot_id, task.primary_target) for task in tasks}) != 100:
            errors.append("primary source targets are not unique")
        incomplete = [
            task.id
            for task in tasks
            if task.shape != "mechanical"
            or not task.snapshot_id
            or task.recipe is None
            or not task.authoring
            or not task.prompt
        ]
        if incomplete:
            errors.append(f"tasks missing mechanical metadata: {', '.join(incomplete)}")
        invalid_families: list[str] = []
        for task in tasks:
            assert task.recipe is not None
            kinds = Counter(operation.kind for operation in task.recipe.operations)
            if task.family == "localized-replacement" and kinds != {"replace": 1}:
                invalid_families.append(f"{task.id}: expected one replacement")
            elif task.family == "block-insertion" and not kinds["insert"]:
                invalid_families.append(f"{task.id}: missing insertion")
            elif task.family == "targeted-removal" and not kinds["delete"]:
                invalid_families.append(f"{task.id}: missing deletion")
            elif task.family == "nested-file-creation" and kinds["create"] != 1:
                invalid_families.append(f"{task.id}: expected one created file")
        if invalid_families:
            errors.append("family/operation mismatches: " + ", ".join(invalid_families))
        if errors:
            raise ValueError("edit-core coverage mismatch:\n" + "\n".join(errors))
    print(f"protocols={','.join(sorted(PROTOCOLS))}")
    if matrix is not None:
        trials = plan_trials(suite, tasks, matrix)
        print(f"matrix={matrix.id} models={sum(item.enabled for item in matrix.models)} trials={len(trials)}")
    if args.verify_oracles:
        asyncio.run(_validate_oracles(tasks))
        print(f"oracles=verified:{len(tasks)}")
    return 0


def _count_group(value: int) -> str:
    if value == 1:
        return "one"
    if value <= 3:
        return "few"
    if value <= 8:
        return "several"
    return "many"


def _file_count_group(value: int) -> str:
    if value == 1:
        return "one"
    if value <= 3:
        return "few"
    return "many"


def _generate(args: argparse.Namespace) -> int:
    suite, tasks = load_suite(args.suite)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "suite.json", suite.model_dump(mode="json"))
    write_materialized_cases(args.output, tasks)
    print(f"Materialized {len(tasks)} tasks in {args.output}")
    return 0


def _browse(args: argparse.Namespace) -> int:
    suite, tasks = load_suite(args.suite)
    output = args.output / suite.id
    index = build_corpus_browser(output, suite, tasks, package_root=PACKAGE_ROOT)
    print(f"Corpus browser: {index}")
    if not args.no_open:
        webbrowser.open(index.resolve().as_uri())
    return 0


def _corpus_import(args: argparse.Namespace) -> int:
    slots, selected = import_sources(args.config, args.checkout_root, PACKAGE_ROOT)
    print(
        f"Imported {len(selected)} real source targets into {len({item.snapshot_id for item in selected})} snapshots."
    )
    print(f"Created {len(slots)} deterministic mechanical authoring slots.")
    return 0


def _corpus_author(args: argparse.Namespace) -> int:
    if not args.confirm_live:
        print("Refusing live corpus authoring without --confirm-live.", file=sys.stderr)
        return 2
    if args.concurrency < 1 or args.concurrency > 8:
        raise ValueError("corpus authoring concurrency must be between 1 and 8")
    tasks = asyncio.run(
        author_corpus(
            slots_path=args.slots,
            output_path=args.output,
            package_root=PACKAGE_ROOT,
            repo_root=REPO_ROOT,
            artifact_root=args.artifact_root,
            concurrency=args.concurrency,
            task_ids=_csv_set(args.task),
            max_tasks=args.max_tasks,
        )
    )
    print(f"Mechanical corpus now contains {len(tasks)} authored task(s).")
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
                protocols=[protocol.value for protocol in production_edit_protocols()],
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
    expected = sum(len(model.protocols) for model in matrix.models)
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
    if args.command == "browse":
        return _browse(args)
    if args.command == "corpus-import":
        return _corpus_import(args)
    if args.command == "corpus-author":
        return _corpus_author(args)
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
