"""Suite loading and deterministic synthetic edit-task generation."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
import random
from typing import Literal, cast

import yaml

from .models import AssertionSpec, FileContent, OracleSpec, SuiteSpec, TaskSpec, validate_relative_path
from .recipes import (
    apply_recipe,
    line_count,
    payload_line_count,
    payload_size_bucket,
    render_recipe_prompt,
    target_length_bucket,
)


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
        language="python",
        family="localized-replacement",
        difficulty="easy",
        shape="micro",
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
        oracle=OracleSpec(
            exact_tree=False,
            functional_assertions=[
                AssertionSpec(kind="contains", path="src/service.py", value=f"def {name}(value):"),
                AssertionSpec(kind="contains", path="src/service.py", value="return value is not None"),
            ],
            instruction_assertions=[
                AssertionSpec(kind="contains", path="src/service.py", value="def process(value):"),
                AssertionSpec(kind="contains", path="src/service.py", value="return value"),
            ],
        ),
        provenance="synthetic",
        seed=index,
        generator="insert_block",
        language="python",
        family="block-insertion",
        difficulty="medium",
        shape="micro",
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
        language="typescript",
        family="targeted-removal",
        difficulty="easy",
        shape="micro",
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
        language="markdown",
        family="nested-file-creation",
        difficulty="easy",
        shape="micro",
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
        language="rust",
        family="same-file-multi-hunk",
        difficulty="medium",
        shape="micro",
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
        language="typescript",
        family="coordinated-multi-file",
        difficulty="medium",
        shape="micro",
    )


_EXTENSION_SCHEDULE = cast(
    tuple[tuple[str, Literal["micro", "repository"]], ...],
    (
        *(("go", "repository"),) * 3,
        ("go", "micro"),
        *(("java", "repository"),) * 3,
        ("java", "micro"),
        *(("cpp", "repository"),) * 3,
        ("cpp", "micro"),
        *(("csharp", "repository"),) * 2,
        ("csharp", "micro"),
        *(("ruby", "repository"),) * 2,
        ("ruby", "micro"),
        *(("php", "repository"),) * 2,
        ("php", "micro"),
        *(("swift", "repository"),) * 2,
        ("swift", "micro"),
        *(("kotlin", "repository"),) * 2,
        ("kotlin", "micro"),
        *(("javascript", "repository"),) * 2,
        ("javascript", "micro"),
        ("structured", "repository"),
        *(("structured", "micro"),) * 3,
    ),
)

_EXTENSIONS = {
    "go": "go",
    "java": "java",
    "cpp": "cpp",
    "csharp": "cs",
    "ruby": "rb",
    "php": "php",
    "swift": "swift",
    "kotlin": "kt",
    "javascript": "js",
    "structured": "yaml",
}


def _render_constants(language: str, values: list[tuple[str, int]]) -> str:
    if language == "go":
        body = "\n".join(f"const {name} = {value}" for name, value in values)
        return f"package settings\n\n{body}\n"
    if language == "java":
        body = "\n".join(f"    public static final int {name} = {value};" for name, value in values)
        return f"public final class Settings {{\n{body}\n}}\n"
    if language == "cpp":
        return "\n".join(f"constexpr int {name} = {value};" for name, value in values) + "\n"
    if language == "csharp":
        body = "\n".join(f"    public const int {name} = {value};" for name, value in values)
        return f"public static class Settings\n{{\n{body}\n}}\n"
    if language == "ruby":
        return "\n".join(f"{name} = {value}" for name, value in values) + "\n"
    if language == "php":
        body = "\n".join(f"const {name} = {value};" for name, value in values)
        return f"<?php\n{body}\n"
    if language == "swift":
        return "\n".join(f"let {name} = {value}" for name, value in values) + "\n"
    if language == "kotlin":
        return "\n".join(f"const val {name} = {value}" for name, value in values) + "\n"
    if language == "javascript":
        return "\n".join(f"export const {name} = {value};" for name, value in values) + "\n"
    return "\n".join(f"{name.lower()}: {value}" for name, value in values) + "\n"


def _settings_path(language: str, *, suffix: str = "settings") -> str:
    extension = _EXTENSIONS[language]
    if language == "go":
        return f"internal/{suffix}/{suffix}.{extension}"
    if language == "java":
        return f"src/main/java/example/{suffix.title()}.{extension}"
    if language == "csharp":
        return f"src/{suffix.title()}.{extension}"
    if language == "ruby":
        return f"lib/{suffix}.{extension}"
    if language == "swift":
        return f"Sources/App/{suffix.title()}.{extension}"
    if language == "kotlin":
        return f"src/main/kotlin/{suffix.title()}.{extension}"
    if language == "structured":
        return f"config/{suffix}.{extension}"
    return f"src/{suffix}.{extension}"


def _multilingual_extension(rng: random.Random, index: int) -> TaskSpec:
    schedule_index = index - 37
    if schedule_index < 0 or schedule_index >= len(_EXTENSION_SCHEDULE):
        raise ValueError(f"multilingual extension index is out of range: {index}")
    language, shape = _EXTENSION_SCHEDULE[schedule_index]
    families = (
        "localized-replacement",
        "ambiguous-context",
        "block-insertion",
        "targeted-removal",
        "same-file-multi-hunk",
        "signature-callsite",
        "coordinated-multi-file",
        "nested-file-creation",
    )
    family = families[schedule_index % len(families)]
    old = rng.randint(2, 20)
    new = old + rng.randint(2, 12)
    path = _settings_path(language)
    before_values = [("TARGET_LIMIT", old), ("SAFE_LIMIT", 30)]
    expected_values = [("TARGET_LIMIT", new), ("SAFE_LIMIT", 30)]
    prompt = f"In `{path}`, change only `TARGET_LIMIT` from {old} to {new}."
    capabilities = {"update"}

    if family == "ambiguous-context":
        before_values = [("PRIMARY_LIMIT", old), ("SECONDARY_LIMIT", old)]
        expected_values = [("PRIMARY_LIMIT", old), ("SECONDARY_LIMIT", new)]
        prompt = f"In `{path}`, change only `SECONDARY_LIMIT` from {old} to {new}."
    elif family == "block-insertion":
        before_values = [("SAFE_LIMIT", 30)]
        expected_values = [("TARGET_LIMIT", new), *before_values]
        prompt = f"Add `TARGET_LIMIT` with value {new} above `SAFE_LIMIT` in `{path}`."
    elif family == "targeted-removal":
        before_values = [("SAFE_LIMIT", 30), ("LEGACY_LIMIT", old), ("TARGET_LIMIT", new)]
        expected_values = [("SAFE_LIMIT", 30), ("TARGET_LIMIT", new)]
        prompt = f"Remove only `LEGACY_LIMIT` from `{path}`."
    elif family == "same-file-multi-hunk":
        before_values = [("TARGET_LIMIT", old), ("SAFE_LIMIT", old + 1)]
        expected_values = [("TARGET_LIMIT", new), ("SAFE_LIMIT", new + 1)]
        prompt = f"In `{path}`, set `TARGET_LIMIT` to {new} and `SAFE_LIMIT` to {new + 1}."

    before = _files({path: _render_constants(language, before_values)})
    expected = _files({path: _render_constants(language, expected_values)})

    if shape == "repository":
        before.update(
            _files(
                {
                    "README.md": f"# {language.title()} settings fixture\n",
                    _settings_path(language, suffix="helper"): _render_constants(language, [("HELPER_LIMIT", 8)]),
                    "config/project.json": '{"enabled":true,"owner":"benchmark"}\n',
                }
            )
        )
        expected = dict(before) | {path: expected[path]}

    if family in {"signature-callsite", "coordinated-multi-file"}:
        second_path = _settings_path(language, suffix="client")
        before[second_path] = FileContent(text=_render_constants(language, [("TARGET_LIMIT", old)]))
        expected[second_path] = FileContent(text=_render_constants(language, [("TARGET_LIMIT", new)]))
        expected[path] = FileContent(text=_render_constants(language, [("TARGET_LIMIT", new), ("SAFE_LIMIT", 30)]))
        prompt = f"Update `TARGET_LIMIT` from {old} to {new} in both `{path}` and `{second_path}`."
        capabilities.add("multi_file")
    elif family == "nested-file-creation":
        created = _settings_path(language, suffix=f"generated_{index}")
        expected[created] = FileContent(text=_render_constants(language, [("GENERATED_LIMIT", new)]))
        prompt = f"Create `{created}` containing only the language-appropriate `GENERATED_LIMIT` constant set to {new}."
        capabilities = {"create"}

    easy_indices = {37, 40, 44, 45, 48, 52, 53, 56}
    hard_indices = {38, 46, 54}
    difficulty = (
        "hard"
        if family in {"signature-callsite", "coordinated-multi-file"} or index in hard_indices
        else "easy"
        if index in easy_indices
        else "medium"
    )
    if shape == "micro" and family == "localized-replacement":
        difficulty = "easy"
    return TaskSpec(
        id=f"synthetic-{language}-{family}-{index:03d}",
        prompt=prompt,
        before_files=before,
        expected_files=expected,
        tags=["synthetic", language, family, shape],
        required_capabilities=capabilities,
        provenance="synthetic",
        seed=index,
        generator="multilingual_extension",
        language=language,
        family=family,
        difficulty=difficulty,
        shape=shape,
    )


GENERATORS: dict[str, Generator] = {
    "replace_literal": _replace_literal,
    "insert_block": _insert_block,
    "remove_block": _remove_block,
    "create_file": _create_file,
    "same_file_multi": _same_file_multi,
    "multi_file": _multi_file,
    "multilingual_extension": _multilingual_extension,
}


def generate_synthetic_tasks(suite: SuiteSpec) -> list[TaskSpec]:
    groups = [spec for spec in [suite.synthetic, *suite.synthetic_extensions] if spec and spec.count]
    if not groups:
        return []
    tasks: list[TaskSpec] = []
    start_index = 1
    for spec in groups:
        names = spec.generators or list(GENERATORS)
        unknown = sorted(set(names) - set(GENERATORS))
        if unknown:
            raise ValueError(f"unknown synthetic generators: {', '.join(unknown)}")
        rng = random.Random(spec.seed)
        for offset in range(spec.count):
            name = names[offset % len(names)]
            case_seed = rng.randrange(0, 2**31)
            task_rng = random.Random(case_seed)
            task = GENERATORS[name](task_rng, start_index + offset)
            task.seed = case_seed
            tasks.append(task)
        start_index += spec.count
    return tasks


def _load_snapshot_task(raw: dict, *, package_root: Path) -> dict:
    """Resolve a compact snapshot task into the existing in-memory task shape."""
    snapshot_id = str(raw.get("snapshot_id") or "")
    if not snapshot_id:
        return raw
    validate_relative_path(snapshot_id)
    snapshot_root = package_root / "fixtures" / "snapshots" / snapshot_id
    manifest_path = snapshot_root / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("id") != snapshot_id:
        raise ValueError(f"snapshot manifest id mismatch: {manifest_path}")
    manifest_files = manifest.get("files") or {}
    before_files: dict[str, FileContent] = {}
    workspace_files = [validate_relative_path(str(item)) for item in raw.get("workspace_files") or []]
    if not workspace_files:
        raise ValueError(f"snapshot task {raw.get('id')!r} has no workspace_files")
    tree = snapshot_root / "tree"
    for relative in workspace_files:
        metadata = manifest_files.get(relative)
        if not isinstance(metadata, dict):
            raise ValueError(f"snapshot file is not declared in {snapshot_id}: {relative}")
        source = (tree / relative).resolve()
        try:
            source.relative_to(tree.resolve())
        except ValueError as exc:
            raise ValueError(f"snapshot path escapes tree: {relative}") from exc
        data = source.read_bytes()
        digest = sha256(data).hexdigest()
        if digest != metadata.get("sha256"):
            raise ValueError(f"snapshot hash mismatch for {snapshot_id}/{relative}")
        before_files[relative] = FileContent(text=data.decode("utf-8"))

    resolved = dict(raw)
    resolved["before_files"] = before_files
    from .models import EditRecipeSpec

    recipe_spec = EditRecipeSpec.model_validate(raw.get("recipe"))
    expected_files = apply_recipe(before_files, recipe_spec)
    resolved["expected_files"] = expected_files
    resolved["prompt"] = raw.get("prompt") or render_recipe_prompt(recipe_spec, before_files)
    resolved.setdefault("shape", "mechanical")
    resolved.setdefault("provenance", "synthetic")
    resolved.setdefault("generator", "mechanical-recipe-v1")
    resolved.setdefault("seed", 20260713)
    changed_paths = sorted({operation.path for operation in recipe_spec.operations})
    capabilities = {"create" if operation.kind == "create" else "update" for operation in recipe_spec.operations}
    if len(changed_paths) > 1:
        capabilities.add("multi_file")
    resolved.setdefault("required_capabilities", sorted(capabilities))
    oracle = dict(raw.get("oracle") or {})
    oracle.setdefault("exact_tree", True)
    oracle.setdefault("allowed_changed_paths", changed_paths)
    resolved["oracle"] = oracle
    primary = str(raw.get("primary_target") or changed_paths[0])
    if primary not in before_files:
        primary = next((path for path in changed_paths if path in before_files), workspace_files[0])
    target_lines = line_count(before_files[primary].text)
    resolved["primary_target"] = primary
    resolved["target_length"] = raw.get("target_length") or target_length_bucket(target_lines)
    resolved["payload_size"] = raw.get("payload_size") or payload_size_bucket(
        payload_line_count(recipe_spec, before_files)
    )
    return resolved


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
    package_root = Path(__file__).resolve().parent
    combined_tasks = [
        _load_snapshot_task(task, package_root=package_root)
        for task in [*source_tasks, *(raw.get("curated_tasks") or [])]
    ]
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
