"""Suite loading and deterministic synthetic edit-task generation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import random

import yaml

from .models import FileContent, SuiteSpec, TaskSpec


Generator = Callable[[random.Random, int], TaskSpec]


def _files(items: dict[str, str]) -> dict[str, FileContent]:
    return {path: FileContent(text=text) for path, text in items.items()}


def _replace_literal(rng: random.Random, index: int) -> TaskSpec:
    old = rng.randint(2, 40)
    new = old + rng.randint(3, 20)
    return TaskSpec(
        id=f"synthetic-replace-{index:03d}",
        prompt=f"Update `src/settings.py` so `RETRY_LIMIT` is {new}. Do not change anything else.",
        before_files=_files({"src/settings.py": f"RETRY_LIMIT = {old}\nTIMEOUT_SECONDS = 30\n"}),
        expected_files=_files({"src/settings.py": f"RETRY_LIMIT = {new}\nTIMEOUT_SECONDS = 30\n"}),
        tags=["synthetic", "single-file", "replace"],
        required_capabilities={"update"},
        provenance="synthetic",
        seed=index,
        generator="replace_literal",
    )


def _insert_block(rng: random.Random, index: int) -> TaskSpec:
    name = rng.choice(["validate", "normalize", "serialize", "render"])
    before = "def process(value):\n    return value\n"
    expected = f"def {name}(value):\n    return value is not None\n\n{before}"
    return TaskSpec(
        id=f"synthetic-insert-{index:03d}",
        prompt=f"Add a `{name}(value)` function above `process` in `src/service.py`; it must return whether value is not None.",
        before_files=_files({"src/service.py": before}),
        expected_files=_files({"src/service.py": expected}),
        tags=["synthetic", "single-file", "insert"],
        required_capabilities={"update"},
        provenance="synthetic",
        seed=index,
        generator="insert_block",
    )


def _remove_block(rng: random.Random, index: int) -> TaskSpec:
    marker = rng.choice(["legacy", "deprecated", "obsolete"])
    before = f"export const active = true;\nexport const {marker} = false;\nexport const version = 2;\n"
    expected = "export const active = true;\nexport const version = 2;\n"
    return TaskSpec(
        id=f"synthetic-remove-{index:03d}",
        prompt=f"Remove the `{marker}` export from `src/flags.ts` and leave the other exports unchanged.",
        before_files=_files({"src/flags.ts": before}),
        expected_files=_files({"src/flags.ts": expected}),
        tags=["synthetic", "single-file", "remove"],
        required_capabilities={"update"},
        provenance="synthetic",
        seed=index,
        generator="remove_block",
    )


def _create_file(rng: random.Random, index: int) -> TaskSpec:
    title = rng.choice(["Cache", "Queue", "Parser", "Worker"])
    content = f"# {title}\n\nGenerated component documentation.\n"
    before = _files({"README.md": "# Project\n"})
    expected = dict(before)
    expected[f"docs/{title.lower()}.md"] = FileContent(text=content)
    return TaskSpec(
        id=f"synthetic-create-{index:03d}",
        prompt=f"Create `docs/{title.lower()}.md` containing exactly a `{title}` H1, a blank line, and `Generated component documentation.`",
        before_files=before,
        expected_files=expected,
        tags=["synthetic", "create-file"],
        required_capabilities={"create"},
        provenance="synthetic",
        seed=index,
        generator="create_file",
    )


def _same_file_multi(rng: random.Random, index: int) -> TaskSpec:
    major = rng.randint(2, 9)
    expected = f"pub const API_VERSION: u8 = {major};\npub const MIN_CLIENT_VERSION: u8 = {major - 1};\n"
    before = f"pub const API_VERSION: u8 = {major - 1};\npub const MIN_CLIENT_VERSION: u8 = {major - 2};\n"
    return TaskSpec(
        id=f"synthetic-multihunk-{index:03d}",
        prompt=(
            f"In `src/version.rs`, set `API_VERSION` to {major} and `MIN_CLIENT_VERSION` to {major - 1}. "
            "Do not rewrite the rest of the file."
        ),
        before_files=_files({"src/version.rs": before}),
        expected_files=_files({"src/version.rs": expected}),
        tags=["synthetic", "single-file", "multi-hunk"],
        required_capabilities={"update"},
        provenance="synthetic",
        seed=index,
        generator="same_file_multi",
    )


def _multi_file(rng: random.Random, index: int) -> TaskSpec:
    version = f"{rng.randint(2, 8)}.{rng.randint(0, 9)}.0"
    old = "1.0.0"
    return TaskSpec(
        id=f"synthetic-multifile-{index:03d}",
        prompt=f"Update the project version to `{version}` in both `package.json` and `src/version.ts`.",
        before_files=_files(
            {
                "package.json": f'{{"name":"demo","version":"{old}"}}\n',
                "src/version.ts": f'export const VERSION = "{old}";\n',
            }
        ),
        expected_files=_files(
            {
                "package.json": f'{{"name":"demo","version":"{version}"}}\n',
                "src/version.ts": f'export const VERSION = "{version}";\n',
            }
        ),
        tags=["synthetic", "multi-file", "replace"],
        required_capabilities={"update", "multi_file"},
        provenance="synthetic",
        seed=index,
        generator="multi_file",
    )


GENERATORS: dict[str, Generator] = {
    "replace_literal": _replace_literal,
    "insert_block": _insert_block,
    "remove_block": _remove_block,
    "create_file": _create_file,
    "same_file_multi": _same_file_multi,
    "multi_file": _multi_file,
}


def generate_synthetic_tasks(suite: SuiteSpec) -> list[TaskSpec]:
    spec = suite.synthetic
    if spec is None or spec.count == 0:
        return []
    names = spec.generators or list(GENERATORS)
    unknown = sorted(set(names) - set(GENERATORS))
    if unknown:
        raise ValueError(f"unknown synthetic generators: {', '.join(unknown)}")
    rng = random.Random(spec.seed)
    tasks: list[TaskSpec] = []
    for offset in range(spec.count):
        name = names[offset % len(names)]
        case_seed = rng.randrange(0, 2**31)
        task_rng = random.Random(case_seed)
        task = GENERATORS[name](task_rng, offset + 1)
        task.seed = case_seed
        tasks.append(task)
    return tasks


def load_suite(path: Path) -> tuple[SuiteSpec, list[TaskSpec]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    source_tasks: list[dict] = []
    for source in raw.get("curated_sources") or []:
        source_path = (path.parent / source).resolve()
        try:
            source_path.relative_to(path.parent.resolve())
        except ValueError as exc:
            raise ValueError(f"curated source escapes suite directory: {source}") from exc
        loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"curated source must contain a task list: {source_path}")
        source_tasks.extend(loaded)
    combined_tasks = [*source_tasks, *(raw.get("curated_tasks") or [])]
    selected_ids = set(raw.get("curated_task_ids") or [])
    if selected_ids:
        available_ids = {str(task.get("id")) for task in combined_tasks}
        missing = sorted(selected_ids - available_ids)
        if missing:
            raise ValueError(f"unknown curated task ids: {', '.join(missing)}")
        combined_tasks = [task for task in combined_tasks if task.get("id") in selected_ids]
    raw["curated_tasks"] = combined_tasks
    suite = SuiteSpec.model_validate(raw)
    tasks = [*suite.curated_tasks, *generate_synthetic_tasks(suite)]
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("expanded suite contains duplicate task ids")
    return suite, tasks
