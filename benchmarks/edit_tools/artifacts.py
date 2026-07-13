"""Append-only run artifacts, stable trial identities, and secret scrubbing."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable
import uuid

from .models import MatrixSpec, SuiteSpec, TaskSpec, TrialRecord, stable_digest


SECRET_KEY_PATTERN = re.compile(r"(api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|secret)", re.I)
SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{16,}=*", re.I),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scrub(value: Any, *, key: str = "") -> Any:
    """Recursively remove common credential fields and token-shaped values."""
    if key and SECRET_KEY_PATTERN.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item_key): scrub(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, tuple):
        return [scrub(item) for item in value]
    if isinstance(value, str):
        result = value
        for pattern in SECRET_VALUE_PATTERNS:
            result = pattern.sub("[REDACTED]", result)
        return result
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(scrub(value), indent=2, sort_keys=True, default=str) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(scrub(value), sort_keys=True, default=str, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_trial_records(path: Path) -> list[TrialRecord]:
    if not path.exists():
        return []
    records: dict[str, TrialRecord] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = TrialRecord.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001 - report the exact corrupt journal line
            raise ValueError(f"invalid trial journal line {line_number}: {exc}") from exc
        records[record.trial_id] = record
    return list(records.values())


def make_run_id(suite_id: str, matrix_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{suite_id}-{matrix_id}-{uuid.uuid4().hex[:8]}"


def trial_id(
    *,
    suite: SuiteSpec,
    task: TaskSpec,
    lane: str,
    provider: str,
    model: str,
    protocol: str,
    protocol_version: str,
    repetition: int,
    seed: int,
    model_parameters: dict[str, Any],
) -> str:
    digest = stable_digest(
        {
            "suite": suite.id,
            "task_digest": task.digest,
            "lane": lane,
            "provider": provider,
            "model": model,
            "protocol": protocol,
            "protocol_version": protocol_version,
            "repetition": repetition,
            "seed": seed,
            "model_parameters": model_parameters,
        }
    )
    return digest[:24]


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    status = run("status", "--porcelain")
    diff = run("diff", "--binary", "--", "kolega_code", "benchmarks", "tests", "pyproject.toml", ".gitignore")
    harness_hash = sha256()
    harness_root = repo_root / "benchmarks" / "edit_tools"
    if harness_root.exists():
        for path in sorted(harness_root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            harness_hash.update(path.relative_to(repo_root).as_posix().encode("utf-8"))
            harness_hash.update(b"\0")
            harness_hash.update(path.read_bytes())
            harness_hash.update(b"\0")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_digest": stable_digest(status),
        "tracked_diff_digest": stable_digest(diff),
        "harness_source_digest": harness_hash.hexdigest(),
    }


def create_manifest(
    *,
    run_id: str,
    repo_root: Path,
    suite: SuiteSpec,
    tasks: Iterable[TaskSpec],
    matrix: MatrixSpec,
    planned_trials: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": utc_now(),
        "repo": git_metadata(repo_root),
        "suite": suite.model_dump(mode="json", exclude={"curated_tasks"}),
        "suite_digest": stable_digest(suite),
        "task_digests": {task.id: task.digest for task in tasks},
        "matrix": matrix.model_dump(mode="json"),
        "matrix_digest": stable_digest(matrix),
        "planned_trials": planned_trials,
    }


def write_materialized_cases(run_dir: Path, tasks: Iterable[TaskSpec]) -> None:
    case_root = run_dir / "cases"
    for task in tasks:
        case_dir = case_root / task.id
        write_json(case_dir / "task.json", task.model_dump(mode="json"))
        for tree_name, files in (("before", task.before_files), ("expected", task.expected_files)):
            for relative, content in files.items():
                path = case_dir / tree_name / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content.text.encode(content.encoding))
