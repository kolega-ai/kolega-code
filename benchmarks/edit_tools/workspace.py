"""Trial workspace materialization, byte-exact oracles, and safe verifiers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import difflib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import tomllib
from typing import Iterable
import uuid

import yaml

from .models import AssertionSpec, CommandSpec, FileContent, OracleSpec, TaskSpec, validate_relative_path


@dataclass
class CommandResult:
    argv: list[str]
    cwd: str
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str | None = None
    infrastructure_error: bool = False

    @property
    def success(self) -> bool:
        return not self.timed_out and self.error is None and self.exit_code == 0


@dataclass
class OracleResult:
    success: bool
    tree_success: bool
    command_success: bool
    functional_success: bool
    instruction_success: bool
    exact_match: bool
    collateral_success: bool
    completed_operations: int = 0
    total_operations: int = 0
    operation_success_rate: float = 0.0
    infrastructure_error: str | None = None
    missing_paths: list[str] = field(default_factory=list)
    unexpected_paths: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    command_results: list[CommandResult] = field(default_factory=list)
    assertion_results: list["AssertionResult"] = field(default_factory=list)
    operation_results: list["OperationResult"] = field(default_factory=list)


@dataclass
class AssertionResult:
    kind: str
    path: str
    role: str
    success: bool
    error: str | None = None


@dataclass
class OperationResult:
    id: str
    kind: str
    path: str
    success: bool
    error: str | None = None


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


async def _run_host_command(root: Path, spec: CommandSpec) -> CommandResult:
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


async def _run_container_command(
    root: Path,
    spec: CommandSpec,
    verifier_files: dict[str, FileContent],
) -> CommandResult:
    image = os.getenv("KOLEGA_EDIT_BENCHMARK_VERIFIER_IMAGE", "kolega-edit-verifier:1")
    name = f"kolega-edit-verify-{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory(prefix="kolega-edit-verify-") as temporary:
        temporary_root = Path(temporary)
        workspace = temporary_root / "workspace"
        verifier = temporary_root / "verifier"
        shutil.copytree(root, workspace)
        materialize_tree(verifier, verifier_files)
        container_cwd = "/workspace" if spec.cwd == "." else f"/workspace/{spec.cwd}"
        argv = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--cpus",
            "2",
            "--memory",
            "2g",
            "--mount",
            f"type=bind,src={workspace},dst=/workspace",
            "--mount",
            f"type=bind,src={verifier},dst=/verifier,readonly",
            "--workdir",
            container_cwd,
            image,
            *spec.argv,
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return CommandResult(
                argv=spec.argv,
                cwd=spec.cwd,
                exit_code=None,
                error=str(exc),
                infrastructure_error=True,
            )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=spec.timeout_seconds)
        except TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            cleanup = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await cleanup.communicate()
            return CommandResult(
                argv=spec.argv,
                cwd=spec.cwd,
                exit_code=process.returncode,
                stdout=stdout.decode("utf-8", "replace"),
                stderr=stderr.decode("utf-8", "replace"),
                timed_out=True,
            )
        stderr_text = stderr.decode("utf-8", "replace")
        infrastructure_error = process.returncode in {125, 126, 127}
        return CommandResult(
            argv=spec.argv,
            cwd=spec.cwd,
            exit_code=process.returncode,
            stdout=stdout.decode("utf-8", "replace"),
            stderr=stderr_text,
            error=stderr_text.strip() if infrastructure_error else None,
            infrastructure_error=infrastructure_error,
        )


async def _run_command(
    root: Path,
    spec: CommandSpec,
    verifier_files: dict[str, FileContent],
) -> CommandResult:
    if spec.runtime == "container":
        return await _run_container_command(root, spec, verifier_files)
    return await _run_host_command(root, spec)


def _lookup(value: object, key_path: list[str]) -> object:
    current = value
    for key in key_path:
        if isinstance(current, dict):
            if key not in current:
                raise KeyError(key)
            current = current[key]
        elif isinstance(current, list):
            current = current[int(key)]
        else:
            raise KeyError(key)
    return current


def _evaluate_assertion(root: Path, spec: AssertionSpec, role: str) -> AssertionResult:
    path = _safe_path(root, spec.path)
    try:
        if spec.kind == "path_exists":
            success = path.exists()
        elif spec.kind == "path_absent":
            success = not path.exists()
        else:
            if not path.is_file():
                raise FileNotFoundError(spec.path)
            content = path.read_bytes()
            text = content.decode("utf-8")
            if spec.kind == "contains":
                success = str(spec.value) in text
            elif spec.kind == "not_contains":
                success = str(spec.value) not in text
            elif spec.kind == "regex_count":
                success = len(re.findall(spec.pattern or "", text, re.MULTILINE | re.DOTALL)) == spec.count
            elif spec.kind == "json_value":
                success = _lookup(json.loads(text), spec.key_path) == spec.value
            elif spec.kind == "yaml_value":
                success = _lookup(yaml.safe_load(text), spec.key_path) == spec.value
            elif spec.kind == "toml_value":
                success = _lookup(tomllib.loads(text), spec.key_path) == spec.value
            elif spec.kind == "bom":
                success = content.startswith(b"\xef\xbb\xbf") is spec.value
            elif spec.kind == "line_endings":
                has_crlf = b"\r\n" in content
                has_bare_lf = b"\n" in content.replace(b"\r\n", b"")
                success = (has_crlf and not has_bare_lf) if spec.value == "crlf" else not has_crlf
            elif spec.kind == "final_newline":
                success = content.endswith((b"\n", b"\r")) is spec.value
            else:  # pragma: no cover - Pydantic constrains this union
                raise ValueError(f"unknown assertion kind: {spec.kind}")
        return AssertionResult(
            kind=spec.kind,
            path=spec.path,
            role=role,
            success=success,
            error=None if success else "assertion did not match",
        )
    except (
        OSError,
        UnicodeError,
        ValueError,
        KeyError,
        IndexError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as exc:
        return AssertionResult(kind=spec.kind, path=spec.path, role=role, success=False, error=str(exc))


def _evaluate_operations(task: TaskSpec, actual: dict[str, bytes]) -> list[OperationResult]:
    if task.recipe is None:
        return []
    results: list[OperationResult] = []
    for operation in task.recipe.operations:
        content = actual.get(operation.path)
        if content is None:
            results.append(
                OperationResult(
                    id=operation.id,
                    kind=operation.kind,
                    path=operation.path,
                    success=False,
                    error="target path is missing",
                )
            )
            continue
        if operation.kind == "create":
            expected = operation.new_text.encode("utf-8")
            success = content == expected
            error = None if success else "created file does not match the supplied content"
        elif operation.kind == "insert":
            expected = operation.new_text.encode("utf-8")
            before_content = task.before_files[operation.path].text.encode("utf-8")
            success = bool(expected) and content.count(expected) > before_content.count(expected)
            error = None if success else "supplied inserted content did not gain a verbatim occurrence"
        else:
            before = task.before_files[operation.path].text.splitlines(keepends=True)
            assert operation.start_line is not None and operation.end_line is not None
            old = "".join(before[operation.start_line - 1 : operation.end_line]).encode("utf-8")
            before_content = task.before_files[operation.path].text.encode("utf-8")
            removed = content.count(old) < before_content.count(old)
            if operation.kind == "replace":
                new = operation.new_text.encode("utf-8")
                added = bool(new) and content.count(new) > before_content.count(new)
                success = removed and added
                error = None if success else "replacement did not remove the old region and add the new region"
            else:
                success = removed
                error = None if success else "deleted original region did not lose an occurrence"
        results.append(
            OperationResult(
                id=operation.id,
                kind=operation.kind,
                path=operation.path,
                success=success,
                error=error,
            )
        )
    return results


async def verify_task(root: Path, task: TaskSpec, *, run_commands: bool = True) -> OracleResult:
    oracle: OracleSpec = task.oracle
    actual = read_tree(root, oracle.ignored_paths)
    expected = expected_tree(task)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    changed = sorted(path for path in set(actual) & set(expected) if actual[path] != expected[path])
    exact_match = not (missing or unexpected or changed)
    tree_success = exact_match if oracle.exact_tree else True
    command_results = (
        [await _run_command(root, command, task.verifier_files) for command in oracle.commands] if run_commands else []
    )
    command_success = all(result.success for result in command_results)
    functional_results = [
        _evaluate_assertion(root, assertion, "functional") for assertion in oracle.functional_assertions
    ]
    instruction_results = [
        _evaluate_assertion(root, assertion, "instruction") for assertion in oracle.instruction_assertions
    ]
    functional_success = command_success and all(result.success for result in functional_results)
    instruction_success = all(result.success for result in instruction_results)
    before = {path: item.text.encode(item.encoding) for path, item in task.before_files.items()}
    intended = {path for path in set(before) | set(expected) if before.get(path) != expected.get(path)}
    allowed = set(oracle.allowed_changed_paths) or intended
    actual_changes = {path for path in set(before) | set(actual) if before.get(path) != actual.get(path)}
    collateral_success = actual_changes <= allowed
    operation_results = _evaluate_operations(task, actual)
    completed_operations = sum(result.success for result in operation_results)
    total_operations = len(operation_results)
    operation_success_rate = completed_operations / total_operations if total_operations else float(exact_match)
    infrastructure = next((result.error for result in command_results if result.infrastructure_error), None)
    return OracleResult(
        success=(
            tree_success
            and functional_success
            and instruction_success
            and collateral_success
            and infrastructure is None
        ),
        tree_success=tree_success,
        command_success=command_success,
        functional_success=functional_success,
        instruction_success=instruction_success,
        exact_match=exact_match,
        collateral_success=collateral_success,
        completed_operations=completed_operations,
        total_operations=total_operations,
        operation_success_rate=operation_success_rate,
        infrastructure_error=infrastructure,
        missing_paths=missing,
        unexpected_paths=unexpected,
        changed_paths=changed,
        command_results=command_results,
        assertion_results=[*functional_results, *instruction_results],
        operation_results=operation_results,
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
        "functional_success": result.functional_success,
        "instruction_success": result.instruction_success,
        "exact_match": result.exact_match,
        "collateral_success": result.collateral_success,
        "completed_operations": result.completed_operations,
        "total_operations": result.total_operations,
        "operation_success_rate": result.operation_success_rate,
        "infrastructure_error": result.infrastructure_error,
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
                "infrastructure_error": item.infrastructure_error,
            }
            for item in result.command_results
        ],
        "assertions": [
            {
                "kind": item.kind,
                "path": item.path,
                "role": item.role,
                "success": item.success,
                "error": item.error,
            }
            for item in result.assertion_results
        ],
        "operations": [
            {
                "id": item.id,
                "kind": item.kind,
                "path": item.path,
                "success": item.success,
                "error": item.error,
            }
            for item in result.operation_results
        ],
    }
