"""Versioned schemas used by the repository-only edit benchmark."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


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
    runtime: Literal["host", "container"] = "host"

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


class AssertionSpec(BaseModel):
    """Portable content or filesystem assertion evaluated by the harness."""

    kind: Literal[
        "path_exists",
        "path_absent",
        "contains",
        "not_contains",
        "regex_count",
        "json_value",
        "yaml_value",
        "toml_value",
        "bom",
        "line_endings",
        "final_newline",
    ]
    path: str
    value: Any = None
    pattern: Optional[str] = None
    count: Optional[int] = Field(default=None, ge=0)
    key_path: list[str] = Field(default_factory=list)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_relative_path(value)

    @model_validator(mode="after")
    def validate_parameters(self) -> "AssertionSpec":
        if self.kind == "regex_count" and (self.pattern is None or self.count is None):
            raise ValueError("regex_count assertions require pattern and count")
        if self.kind in {"json_value", "yaml_value", "toml_value"} and not self.key_path:
            raise ValueError(f"{self.kind} assertions require key_path")
        if self.kind in {"bom", "final_newline"} and not isinstance(self.value, bool):
            raise ValueError(f"{self.kind} assertions require a boolean value")
        if self.kind == "line_endings" and self.value not in {"lf", "crlf"}:
            raise ValueError("line_endings assertions require value 'lf' or 'crlf'")
        if self.kind in {"contains", "not_contains"} and not isinstance(self.value, str):
            raise ValueError(f"{self.kind} assertions require a string value")
        return self


class OracleSpec(BaseModel):
    exact_tree: bool = True
    commands: list[CommandSpec] = Field(default_factory=list)
    allowed_changed_paths: list[str] = Field(default_factory=list)
    functional_assertions: list[AssertionSpec] = Field(default_factory=list)
    instruction_assertions: list[AssertionSpec] = Field(default_factory=list)
    ignored_paths: list[str] = Field(
        default_factory=lambda: [".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache"]
    )

    @field_validator("allowed_changed_paths")
    @classmethod
    def validate_allowed_changed_paths(cls, value: list[str]) -> list[str]:
        return [validate_relative_path(path) for path in value]


class EditOperationSpec(BaseModel):
    """One tool-neutral edit expressed against the original before file."""

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    kind: Literal["replace", "insert", "delete", "create"]
    path: str
    start_line: Optional[int] = Field(default=None, ge=0)
    end_line: Optional[int] = Field(default=None, ge=1)
    before_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    new_text: str = ""

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_relative_path(value)

    @model_validator(mode="after")
    def validate_coordinates(self) -> "EditOperationSpec":
        if self.kind == "create":
            if self.start_line is not None or self.end_line is not None or self.before_sha256 is not None:
                raise ValueError("create operations do not use source coordinates or a before hash")
            if not self.new_text:
                raise ValueError("create operations require complete file content")
            return self
        if self.kind == "insert":
            if self.start_line is None or self.end_line is not None:
                raise ValueError("insert operations require start_line and no end_line")
            if not self.before_sha256:
                raise ValueError("insert operations require an anchor before_sha256")
            if not self.new_text:
                raise ValueError("insert operations require new_text")
            return self
        if self.start_line is None or self.start_line < 1 or self.end_line is None:
            raise ValueError(f"{self.kind} operations require one-based start_line and end_line")
        if self.end_line < self.start_line:
            raise ValueError("end_line must not precede start_line")
        if not self.before_sha256:
            raise ValueError(f"{self.kind} operations require before_sha256")
        if self.kind == "replace" and not self.new_text:
            raise ValueError("replace operations require new_text")
        if self.kind == "delete" and self.new_text:
            raise ValueError("delete operations cannot contain new_text")
        return self


class EditRecipeSpec(BaseModel):
    """Exact mechanical edits rendered into a protocol-neutral user prompt."""

    renderer_version: Literal["1"] = "1"
    operations: list[EditOperationSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_operations(self) -> "EditRecipeSpec":
        ids = [operation.id for operation in self.operations]
        if len(ids) != len(set(ids)):
            raise ValueError("edit recipe contains duplicate operation ids")
        return self


class TaskSpec(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    prompt: str = Field(min_length=1)
    before_files: dict[str, FileContent]
    expected_files: dict[str, FileContent]
    verifier_files: dict[str, FileContent] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    required_capabilities: set[str] = Field(default_factory=lambda: {"update"})
    oracle: OracleSpec = Field(default_factory=OracleSpec)
    provenance: Literal["curated", "synthetic"] = "curated"
    seed: Optional[int] = None
    generator: Optional[str] = None
    language: Optional[str] = None
    family: Optional[str] = None
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    shape: Optional[Literal["micro", "repository", "mechanical"]] = None
    snapshot_id: Optional[str] = Field(default=None, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    workspace_files: list[str] = Field(default_factory=list)
    primary_target: Optional[str] = None
    target_length: Optional[Literal["short", "normal", "medium", "long", "oversized"]] = None
    payload_size: Optional[Literal["tiny", "small", "medium", "large", "very-large"]] = None
    recipe: Optional[EditRecipeSpec] = None
    authoring: dict[str, Any] = Field(default_factory=dict)

    @field_validator("before_files", "expected_files", "verifier_files", mode="before")
    @classmethod
    def validate_files(cls, value: Any, info: ValidationInfo) -> dict[str, Any]:
        if isinstance(value, dict):
            value = {
                path: ({"text": content} if isinstance(content, str) else content) for path, content in value.items()
            }
        if not value and info.field_name != "verifier_files":
            raise ValueError("fixture trees must contain at least one file")
        return {validate_relative_path(path): content for path, content in value.items()}

    @field_validator("workspace_files")
    @classmethod
    def validate_workspace_files(cls, value: list[str]) -> list[str]:
        result = [validate_relative_path(path) for path in value]
        if len(result) != len(set(result)):
            raise ValueError("workspace_files contains duplicate paths")
        return result

    @field_validator("primary_target")
    @classmethod
    def validate_primary_target(cls, value: Optional[str]) -> Optional[str]:
        return validate_relative_path(value) if value is not None else None

    @model_validator(mode="after")
    def validate_change(self) -> "TaskSpec":
        if self.before_files == self.expected_files:
            raise ValueError(f"task {self.id!r} does not change the workspace")
        if self.provenance == "synthetic" and (self.seed is None or not self.generator):
            raise ValueError("synthetic tasks require seed and generator metadata")
        if self.snapshot_id:
            if not self.workspace_files or not self.primary_target or self.recipe is None:
                raise ValueError("snapshot tasks require workspace_files, primary_target, and recipe")
            if set(self.workspace_files) != set(self.before_files):
                raise ValueError("resolved snapshot files do not match workspace_files")
            if self.primary_target not in self.before_files:
                raise ValueError("primary_target is not present in the before workspace")
            recipe_paths = {operation.path for operation in self.recipe.operations if operation.kind != "create"}
            if not recipe_paths <= set(self.before_files):
                raise ValueError("edit recipe references files outside the before workspace")
        return self

    @property
    def digest(self) -> str:
        if self.snapshot_id:
            return stable_digest(self)
        # Reporting-only metadata is deliberately excluded so adding language
        # classification does not invalidate historical task identities. New
        # verifier behavior is included whenever it is configured.
        payload = self.model_dump(
            mode="python",
            exclude={"language", "family", "difficulty", "shape", "verifier_files"},
            exclude_none=True,
        )
        oracle = payload["oracle"]
        if not self.oracle.allowed_changed_paths:
            oracle.pop("allowed_changed_paths", None)
        if not self.oracle.functional_assertions:
            oracle.pop("functional_assertions", None)
        if not self.oracle.instruction_assertions:
            oracle.pop("instruction_assertions", None)
        if self.verifier_files:
            payload["verifier_files"] = self.verifier_files
        return stable_digest(payload)


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
    synthetic_extensions: list[SyntheticGroupSpec] = Field(default_factory=list)
    default_repetitions: int = Field(default=1, ge=1, le=100)

    @model_validator(mode="after")
    def validate_ids(self) -> "SuiteSpec":
        ids = [task.id for task in self.curated_tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("suite contains duplicate curated task ids")
        if (
            not ids
            and not (self.synthetic and self.synthetic.count)
            and not any(extension.count for extension in self.synthetic_extensions)
        ):
            raise ValueError("suite must contain curated or synthetic tasks")
        return self


class ModelRunSpec(BaseModel):
    provider: str
    model: str
    protocols: list[str] = Field(
        default_factory=lambda: ["search_replace", "codex_apply_patch", "claude_code", "hashline_v2"]
    )
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
    max_iterations: int = Field(default=12, ge=1, le=100)

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
    functional_success: bool = False
    instruction_success: bool = False
    exact_match: bool = False
    collateral_success: bool = False
    operation_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    completed_operations: int = Field(default=0, ge=0)
    total_operations: int = Field(default=0, ge=0)
    first_attempt_file_successes: int = Field(default=0, ge=0)
    first_attempt_files: int = Field(default=0, ge=0)
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
    language: Optional[str] = None
    family: Optional[str] = None
    difficulty: Optional[str] = None
    shape: Optional[str] = None
    target_length: Optional[str] = None
    payload_size: Optional[str] = None
    target_file_count: int = Field(default=0, ge=0)
    artifact_dir: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def populate_backward_compatible_scores(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        oracle_success = bool(result.get("oracle_success", result.get("task_success", False)))
        result.setdefault("functional_success", oracle_success)
        result.setdefault("instruction_success", oracle_success)
        result.setdefault("exact_match", oracle_success)
        result.setdefault("collateral_success", not bool(result.get("collateral_paths")))
        result.setdefault("operation_success_rate", 1.0 if oracle_success else 0.0)
        return result

    @property
    def scored(self) -> bool:
        return self.status in {"passed", "failed"}


def load_yaml_model(path: Path, model_type: type[BaseModel]) -> BaseModel:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return model_type.model_validate(raw)
