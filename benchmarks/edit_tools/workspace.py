"""Trial workspace materialization, byte-exact oracles, and safe verifiers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import difflib
from pathlib import Path
from typing import Iterable

from .models import CommandSpec, FileContent, OracleSpec, TaskSpec, validate_relative_path


@dataclass
class CommandResult:
    argv: list[str]
    cwd: str
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str | None = None

    @property
    def success(self) -> bool:
        return not self.timed_out and self.error is None and self.exit_code == 0


@dataclass
class OracleResult:
    success: bool
    tree_success: bool
    command_success: bool
    missing_paths: list[str] = field(default_factory=list)
    unexpected_paths: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    command_results: list[CommandResult] = field(default_factory=list)


def _safe_path(root: Path, relative: str) -> Path:
    normalized = validate_relative_path(relative)
    candidate = (root / normalized).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes trial workspace: {relative!r}") from exc
    return candidate


def materialize_tree(root: Path, files: dict[str, FileContent]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        path = _safe_path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.text.encode(content.encoding))


def materialize_task(root: Path, task: TaskSpec) -> None:
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"trial workspace is not empty: {root}")
    materialize_tree(root, task.before_files)


def _ignored(relative: str, ignored_paths: Iterable[str]) -> bool:
    parts = Path(relative).parts
    for ignored in ignored_paths:
        ignored_parts = Path(ignored).parts
        if parts[: len(ignored_parts)] == ignored_parts or any(part == ignored for part in parts):
            return True
    return False


def read_tree(root: Path, ignored_paths: Iterable[str] = ()) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    if not root.exists():
        return result
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed in benchmark workspaces: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if not _ignored(relative, ignored_paths):
            result[relative] = path.read_bytes()
    return result


def expected_tree(task: TaskSpec) -> dict[str, bytes]:
    return {path: content.text.encode(content.encoding) for path, content in task.expected_files.items()}


async def _run_command(root: Path, spec: CommandSpec) -> CommandResult:
    cwd = root if spec.cwd == "." else _safe_path(root, spec.cwd)
    try:
        process = await asyncio.create_subprocess_exec(
            *spec.argv,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return CommandResult(argv=spec.argv, cwd=spec.cwd, exit_code=None, error=str(exc))
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=spec.timeout_seconds)
    except TimeoutError:
        process.kill()
        stdout, stderr = await process.communicate()
        return CommandResult(
            argv=spec.argv,
            cwd=spec.cwd,
            exit_code=process.returncode,
            stdout=stdout.decode("utf-8", "replace"),
            stderr=stderr.decode("utf-8", "replace"),
            timed_out=True,
        )
    return CommandResult(
        argv=spec.argv,
        cwd=spec.cwd,
        exit_code=process.returncode,
        stdout=stdout.decode("utf-8", "replace"),
        stderr=stderr.decode("utf-8", "replace"),
    )


async def verify_task(root: Path, task: TaskSpec) -> OracleResult:
    oracle: OracleSpec = task.oracle
    actual = read_tree(root, oracle.ignored_paths)
    expected = expected_tree(task)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    changed = sorted(path for path in set(actual) & set(expected) if actual[path] != expected[path])
    tree_success = not (missing or unexpected or changed) if oracle.exact_tree else True
    command_results = [await _run_command(root, command) for command in oracle.commands]
    command_success = all(result.success for result in command_results)
    return OracleResult(
        success=tree_success and command_success,
        tree_success=tree_success,
        command_success=command_success,
        missing_paths=missing,
        unexpected_paths=unexpected,
        changed_paths=changed,
        command_results=command_results,
    )


def workspace_diff(task: TaskSpec, root: Path) -> str:
    before = {path: item.text.encode(item.encoding) for path, item in task.before_files.items()}
    after = read_tree(root, task.oracle.ignored_paths)
    lines: list[str] = []
    for path in sorted(set(before) | set(after)):
        old = before.get(path)
        new = after.get(path)
        if old == new:
            continue
        try:
            old_lines = old.decode("utf-8").splitlines(keepends=True) if old is not None else []
            new_lines = new.decode("utf-8").splitlines(keepends=True) if new is not None else []
        except UnicodeDecodeError:
            lines.append(f"Binary files differ: {path}\n")
            continue
        lines.extend(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{path}" if old is not None else "/dev/null",
                tofile=f"b/{path}" if new is not None else "/dev/null",
            )
        )
    return "".join(lines)


def collateral_paths(task: TaskSpec, root: Path) -> list[str]:
    """Return paths changed by the model that the requested target did not require."""
    before = {path: item.text.encode(item.encoding) for path, item in task.before_files.items()}
    expected = expected_tree(task)
    actual = read_tree(root, task.oracle.ignored_paths)
    intended = {path for path in set(before) | set(expected) if before.get(path) != expected.get(path)}
    actual_changes = {path for path in set(before) | set(actual) if before.get(path) != actual.get(path)}
    return sorted(actual_changes - intended)


def oracle_to_dict(result: OracleResult) -> dict:
    return {
        "success": result.success,
        "tree_success": result.tree_success,
        "command_success": result.command_success,
        "missing_paths": result.missing_paths,
        "unexpected_paths": result.unexpected_paths,
        "changed_paths": result.changed_paths,
        "commands": [
            {
                "argv": item.argv,
                "cwd": item.cwd,
                "exit_code": item.exit_code,
                "stdout": item.stdout,
                "stderr": item.stderr,
                "timed_out": item.timed_out,
                "error": item.error,
                "success": item.success,
            }
            for item in result.command_results
        ],
    }
