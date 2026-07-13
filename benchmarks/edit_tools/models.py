"""Versioned schemas used by the repository-only edit benchmark."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


SCHEMA_VERSION = 1


def stable_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python", exclude_none=True)

    def canonical(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return canonical(item.model_dump(mode="python", exclude_none=True))
        if isinstance(item, dict):
            return {str(key): canonical(child) for key, child in sorted(item.items(), key=lambda pair: str(pair[0]))}
        if isinstance(item, (set, frozenset)):
            children = [canonical(child) for child in item]
            return sorted(children, key=lambda child: json.dumps(child, sort_keys=True, ensure_ascii=False))
        if isinstance(item, (list, tuple)):
            return [canonical(child) for child in item]
        if isinstance(item, Path):
            return item.as_posix()
        return item

    value = canonical(value)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or value in {".", ".."}:
        raise ValueError(f"benchmark paths must be project-relative: {value!r}")
    return path.as_posix()


class FileContent(BaseModel):
    """A text fixture file; ``encoding`` is explicit for future binary support."""

    text: str
    encoding: Literal["utf-8"] = "utf-8"


class CommandSpec(BaseModel):
    argv: list[str]
    cwd: str = "."
    timeout_seconds: float = Field(default=30.0, gt=0, le=600)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: list[str]) -> list[str]:
        if not value or not value[0].strip():
            raise ValueError("verifier argv must contain an executable")
        return value

    @field_validator("cwd")
    @classmethod
    def validate_cwd(cls, value: str) -> str:
        if value == ".":
            return value
        return validate_relative_path(value)


class OracleSpec(BaseModel):
    exact_tree: bool = True
    commands: list[CommandSpec] = Field(default_factory=list)
    ignored_paths: list[str] = Field(
        default_factory=lambda: [".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache"]
    )


class TaskSpec(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    prompt: str = Field(min_length=1)
    before_files: dict[str, FileContent]
    expected_files: dict[str, FileContent]
    tags: list[str] = Field(default_factory=list)
    required_capabilities: set[str] = Field(default_factory=lambda: {"update"})
    oracle: OracleSpec = Field(default_factory=OracleSpec)
    provenance: Literal["curated", "synthetic"] = "curated"
    seed: Optional[int] = None
    generator: Optional[str] = None

    @field_validator("before_files", "expected_files", mode="before")
    @classmethod
    def validate_files(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            value = {
                path: ({"text": content} if isinstance(content, str) else content) for path, content in value.items()
            }
        if not value:
            raise ValueError("fixture trees must contain at least one file")
        return {validate_relative_path(path): content for path, content in value.items()}

    @model_validator(mode="after")
    def validate_change(self) -> "TaskSpec":
        if self.before_files == self.expected_files:
            raise ValueError(f"task {self.id!r} does not change the workspace")
        if self.provenance == "synthetic" and (self.seed is None or not self.generator):
            raise ValueError("synthetic tasks require seed and generator metadata")
        return self

    @property
    def digest(self) -> str:
        return stable_digest(self)


class SyntheticGroupSpec(BaseModel):
    count: int = Field(default=0, ge=0, le=10_000)
    seed: int = 1
    generators: list[str] = Field(default_factory=list)


class SuiteSpec(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    description: str = ""
    curated_sources: list[str] = Field(default_factory=list)
    curated_task_ids: list[str] = Field(default_factory=list)
    curated_tasks: list[TaskSpec] = Field(default_factory=list)
    synthetic: Optional[SyntheticGroupSpec] = None
    default_repetitions: int = Field(default=1, ge=1, le=100)

    @model_validator(mode="after")
    def validate_ids(self) -> "SuiteSpec":
        ids = [task.id for task in self.curated_tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("suite contains duplicate curated task ids")
        if not ids and not (self.synthetic and self.synthetic.count):
            raise ValueError("suite must contain curated or synthetic tasks")
        return self


class ModelRunSpec(BaseModel):
    provider: str
    model: str
    protocols: list[str] = Field(default_factory=lambda: ["search_replace", "codex_apply_patch"])
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    thinking_effort: Optional[str] = None
    max_output_tokens: int = Field(default=8192, ge=128, le=131_072)
    enabled: bool = True

    @field_validator("protocols")
    @classmethod
    def validate_protocols(cls, value: list[str]) -> list[str]:
        result = list(dict.fromkeys(value))
        if not result:
            raise ValueError("a model matrix entry needs at least one protocol")
        return result


class MatrixSpec(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    models: list[ModelRunSpec]
    lanes: list[Literal["controlled", "coder_agent"]] = Field(default_factory=lambda: ["controlled"])
    repetitions: Optional[int] = Field(default=None, ge=1, le=100)
    concurrency: int = Field(default=1, ge=1, le=32)
    trial_timeout_seconds: float = Field(default=480.0, gt=0, le=3600)

    @model_validator(mode="after")
    def validate_models(self) -> "MatrixSpec":
        enabled = [(item.provider, item.model) for item in self.models if item.enabled]
        if not enabled:
            raise ValueError("matrix must contain an enabled model")
        if len(enabled) != len(set(enabled)):
            raise ValueError("matrix contains duplicate provider/model entries")
        return self


TrialStatus = Literal[
    "passed",
    "failed",
    "unsupported",
    "provider_error",
    "harness_error",
    "not_run",
    "cancelled",
]


class ToolAttempt(BaseModel):
    iteration: int
    name: str
    input_kind: Literal["json", "freeform"]
    raw_input: Any
    parse_ok: bool = True
    apply_ok: bool = False
    is_error: bool = False
    error: Optional[str] = None
    elapsed_ms: int = 0


class UsageTotals(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    requests: int = 0


class TrialRecord(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    trial_id: str
    run_id: str
    suite_id: str
    task_id: str
    task_digest: str
    lane: Literal["controlled", "coder_agent"]
    provider: str
    model: str
    protocol: str
    protocol_version: str
    repetition: int
    seed: int
    status: TrialStatus
    task_success: bool = False
    first_attempt_success: bool = False
    oracle_success: bool = False
    terminal_stop: bool = False
    failure_stage: Optional[str] = None
    error: Optional[str] = None
    started_at: str
    finished_at: str
    elapsed_ms: int
    usage: UsageTotals = Field(default_factory=UsageTotals)
    tool_attempts: list[ToolAttempt] = Field(default_factory=list)
    collateral_paths: list[str] = Field(default_factory=list)
    artifact_dir: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def scored(self) -> bool:
        return self.status in {"passed", "failed"}


def load_yaml_model(path: Path, model_type: type[BaseModel]) -> BaseModel:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return model_type.model_validate(raw)
