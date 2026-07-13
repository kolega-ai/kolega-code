"""Versioned edit-protocol adapters backed by Kolega Code's real tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.tools import ToolCollection, ToolCollectionConfig, ToolExtension
from kolega_code.config import AgentConfig, EditProtocol
from kolega_code.events import AgentConnectionManager, AgentEvent
from kolega_code.llm.models import ToolCall, ToolDefinition, ToolResult
from kolega_code.services.lsp.config import LspConfig
from kolega_code.services.snapshots import SnapshotService

from .models import ToolAttempt


READ_TOOL_NAMES = (
    "list_directory",
    "read_entire_file",
    "read_file_section",
    "search_codebase",
    "find_files_by_pattern",
)


class RecordingConnectionManager(AgentConnectionManager):
    """In-memory event sink used by isolated benchmark trials."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def connect(self, websocket: Any, workspace_id: str, thread_id: str, connection_type: str, user_info=None):
        return None

    def disconnect(self, websocket: Any, workspace_id: str, thread_id: str, connection_type: str) -> None:
        return None

    async def broadcast_event(self, event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        self.events.append(event)

    def get_connection_count(self, workspace_id: str, thread_id: str) -> dict:
        return {}


class BenchmarkCaller:
    agent_name = "edit-benchmark"
    supports_vision = False
    sub_agent = False
    agent_mode = AgentMode.CLI
    session_id = "benchmark"
    custom_agent_catalog = None
    gigacode_enabled = False

    def __init__(self) -> None:
        self.current_tool_execution_id: str | None = None
        self.current_tool_call_id: str | None = None
        self.current_provider_tool_call_id: str | None = None


@dataclass(frozen=True)
class ProtocolAdapter:
    id: str
    version: str
    production_protocol: EditProtocol | None
    tool_names: tuple[str, ...]
    capabilities: frozenset[str]
    extension_factory: Callable[[Path], ToolExtension] | None = None
    definition_factory: Callable[[], list[ToolDefinition]] | None = None

    def definitions(self, collection: ToolCollection) -> list[ToolDefinition]:
        registry = collection.registry()
        definitions = [registry.get(name).definition for name in READ_TOOL_NAMES if name in registry]
        if self.production_protocol is not None:
            definitions.extend(registry.get(name).definition for name in self.tool_names if name in registry)
        elif self.definition_factory is not None:
            definitions.extend(self.definition_factory())
        return definitions

    def supports(self, required: set[str]) -> bool:
        return required.issubset(self.capabilities)


PROTOCOLS: dict[str, ProtocolAdapter] = {
    "search_replace": ProtocolAdapter(
        id="search_replace",
        version="1",
        production_protocol=EditProtocol.SEARCH_REPLACE,
        tool_names=("edit", "multi_edit", "write"),
        capabilities=frozenset({"update", "create", "multi_file"}),
    ),
    "codex_apply_patch": ProtocolAdapter(
        id="codex_apply_patch",
        version="1",
        production_protocol=EditProtocol.CODEX_APPLY_PATCH,
        tool_names=("apply_patch",),
        capabilities=frozenset({"update", "create", "delete", "move", "multi_file"}),
    ),
}


def get_protocol(protocol_id: str) -> ProtocolAdapter:
    try:
        return PROTOCOLS[protocol_id]
    except KeyError as exc:
        raise ValueError(f"unknown edit protocol {protocol_id!r}; available: {', '.join(sorted(PROTOCOLS))}") from exc


def create_tool_collection(
    project_path: Path,
    config: AgentConfig,
    adapter: ProtocolAdapter,
    artifact_root: Path,
) -> tuple[ToolCollection, RecordingConnectionManager, BenchmarkCaller]:
    if adapter.production_protocol is None and (
        adapter.extension_factory is None or adapter.definition_factory is None
    ):
        raise ValueError(f"research protocol {adapter.id!r} requires both extension_factory and definition_factory")
    config = config.model_copy(
        update={
            "edit_protocol": adapter.production_protocol or EditProtocol.SEARCH_REPLACE,
            "lsp": LspConfig(enabled=False),
        }
    )
    connection = RecordingConnectionManager()
    caller = BenchmarkCaller()
    extensions = [adapter.extension_factory(project_path)] if adapter.extension_factory is not None else []
    collection = ToolCollection(
        project_path,
        "benchmark-workspace",
        f"benchmark-{adapter.id}",
        connection,
        config,
        caller,
        tool_config=ToolCollectionConfig(allowed_tools=[*READ_TOOL_NAMES, *adapter.tool_names]),
        tool_extensions=extensions,
    )
    snapshot = SnapshotService(
        project_path,
        "benchmark-workspace",
        f"benchmark-{adapter.id}",
        f"benchmark-{adapter.id}",
        collection.filesystem,
        root=artifact_root / "private-state",
    )
    collection.snapshot_service = snapshot
    collection.edit_tool._snapshot_service = snapshot
    return collection, connection, caller


def normalize_call(call: ToolCall, definitions: dict[str, ToolDefinition]) -> ToolCall:
    definition = definitions.get(call.name)
    if definition is None or definition.input_kind != "freeform":
        return call
    if isinstance(call.input, dict):
        if set(call.input) != {"input"}:
            raise ValueError(f"freeform fallback for {call.name} must contain only an input string")
        raw = call.input.get("input")
        if not isinstance(raw, str):
            raise ValueError(f"freeform fallback for {call.name} requires a string input")
        call.input = raw
    if not isinstance(call.input, str):
        raise ValueError(f"freeform tool {call.name} requires raw string input")
    call.input_kind = "freeform"
    return call


async def execute_call(
    collection: ToolCollection,
    caller: BenchmarkCaller,
    call: ToolCall,
    definitions: dict[str, ToolDefinition],
    iteration: int,
) -> tuple[ToolResult, ToolAttempt]:
    started = time.monotonic()
    raw_input = call.input
    parse_ok = True
    error: str | None = None
    try:
        call = normalize_call(call, definitions)
        raw_input = call.input
    except Exception as exc:  # noqa: BLE001 - malformed model output becomes a tool error
        parse_ok = False
        error = str(exc)

    output = ""
    is_error = False
    if parse_ok:
        try:
            registry = collection.registry()
            if call.name not in registry:
                raise ValueError(f"tool {call.name!r} is not available in this benchmark lane")
            caller.current_provider_tool_call_id = call.id
            caller.current_tool_execution_id = call.execution_id
            caller.current_tool_call_id = call.execution_id
            inputs = {"input": call.input} if call.input_kind == "freeform" else call.input
            if not isinstance(inputs, dict):
                raise ValueError(f"tool {call.name!r} requires object input")
            output = str(await registry.call(call.name, **inputs))
        except Exception as exc:  # noqa: BLE001 - the model may recover on its next iteration
            is_error = True
            error = str(exc)
            output = f"Tool error: {error}"
    else:
        is_error = True
        output = f"Tool error: {error}"

    definition = definitions.get(call.name)
    input_kind = definition.input_kind if definition is not None else call.input_kind
    result = ToolResult(
        tool_use_id=call.id,
        name=call.name,
        content=output,
        is_error=is_error,
        execution_id=call.execution_id,
        input_kind=input_kind,
    )
    attempt = ToolAttempt(
        iteration=iteration,
        name=call.name,
        input_kind=input_kind,
        raw_input=raw_input,
        parse_ok=parse_ok,
        apply_ok=parse_ok and not is_error,
        is_error=is_error,
        error=error,
        elapsed_ms=round((time.monotonic() - started) * 1000),
    )
    return result, attempt
