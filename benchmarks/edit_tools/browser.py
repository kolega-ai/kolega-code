"""Generate a static browser for inspecting edit benchmark tasks."""

from __future__ import annotations

from collections import Counter
from difflib import unified_diff
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from .models import SuiteSpec, TaskSpec


ASSET_ROOT = Path(__file__).resolve().parent / "browser_assets"


def _line_count(text: str | None) -> int:
    if text is None:
        return 0
    return len(text.splitlines())


def _repository_url(value: str) -> str:
    return value.removesuffix(".git")


def _snapshot_metadata(package_root: Path, task: TaskSpec) -> dict[str, Any] | None:
    if task.snapshot_id is None:
        return None
    path = package_root / "fixtures" / "snapshots" / task.snapshot_id / "manifest.yaml"
    if not path.is_file():
        return {"id": task.snapshot_id}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    repository = _repository_url(str(raw["repository"]))
    commit = str(raw["commit"])
    return {
        "id": task.snapshot_id,
        "repository": repository,
        "commit": commit,
        "commit_url": f"{repository}/commit/{commit}",
        "license": raw.get("license"),
        "language": raw.get("language"),
    }


def _file_payload(task: TaskSpec, path: str) -> dict[str, Any]:
    before_file = task.before_files.get(path)
    expected_file = task.expected_files.get(path)
    before = before_file.text if before_file is not None else None
    expected = expected_file.text if expected_file is not None else None
    before_lines = [] if before is None else before.splitlines(keepends=True)
    expected_lines = [] if expected is None else expected.splitlines(keepends=True)
    diff = "".join(
        unified_diff(
            before_lines,
            expected_lines,
            fromfile=f"a/{path}" if before is not None else "/dev/null",
            tofile=f"b/{path}" if expected is not None else "/dev/null",
            n=4,
        )
    )
    return {
        "path": path,
        "before": before,
        "expected": expected,
        "diff": diff,
        "changed": before != expected,
        "before_lines": _line_count(before),
        "expected_lines": _line_count(expected),
        "status": "created" if before is None else "deleted" if expected is None else "modified",
    }


def _task_payload(package_root: Path, task: TaskSpec, index: int) -> dict[str, Any]:
    paths = sorted(set(task.before_files) | set(task.expected_files))
    if task.primary_target in paths:
        paths.remove(task.primary_target)
        paths.insert(0, task.primary_target)
    files = [_file_payload(task, path) for path in paths]
    recipe = task.recipe
    operations = []
    if recipe is not None:
        operations = [operation.model_dump(mode="json", exclude_none=True) for operation in recipe.operations]
    authoring = {
        key: task.authoring[key]
        for key in ("provider", "model", "protocol", "attempt", "prompt_template", "syntax_parser")
        if key in task.authoring
    }
    return {
        "index": index,
        "id": task.id,
        "prompt": task.prompt,
        "language": task.language,
        "family": task.family,
        "difficulty": task.difficulty,
        "shape": task.shape,
        "target_length": task.target_length,
        "payload_size": task.payload_size,
        "provenance": task.provenance,
        "generator": task.generator,
        "seed": task.seed,
        "snapshot": _snapshot_metadata(package_root, task),
        "primary_target": task.primary_target,
        "workspace_files": task.workspace_files,
        "tags": task.tags,
        "required_capabilities": sorted(task.required_capabilities),
        "exact_tree": task.oracle.exact_tree,
        "operations": operations,
        "operation_count": len(operations),
        "changed_file_count": sum(item["changed"] for item in files),
        "workspace_file_count": len(files),
        "files": files,
        "authoring": authoring,
    }


def build_corpus_browser(
    output_dir: Path,
    suite: SuiteSpec,
    tasks: list[TaskSpec],
    *,
    package_root: Path | None = None,
) -> Path:
    """Write a self-contained static corpus browser and return its index path."""
    package_root = package_root or Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_tasks = [_task_payload(package_root, task, index) for index, task in enumerate(tasks)]
    repositories = {
        item["snapshot"]["repository"]
        for item in payload_tasks
        if item["snapshot"] and item["snapshot"].get("repository")
    }
    summary = {
        "tasks": len(tasks),
        "languages": len({task.language for task in tasks if task.language}),
        "families": len({task.family for task in tasks if task.family}),
        "repositories": len(repositories),
        "operations": sum(item["operation_count"] for item in payload_tasks),
        "changed_files": sum(item["changed_file_count"] for item in payload_tasks),
        "language_counts": dict(sorted(Counter(task.language for task in tasks if task.language).items())),
        "family_counts": dict(sorted(Counter(task.family for task in tasks if task.family).items())),
    }
    payload = {
        "schema_version": 1,
        "suite": {
            "id": suite.id,
            "description": suite.description,
        },
        "summary": summary,
        "tasks": payload_tasks,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = (
        encoded.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    (output_dir / "data.js").write_text(f"window.BENCHMARK_DATA={encoded};\n", encoding="utf-8")
    for asset in ("index.html", "styles.css", "app.js"):
        shutil.copyfile(ASSET_ROOT / asset, output_dir / asset)
    return output_dir / "index.html"
