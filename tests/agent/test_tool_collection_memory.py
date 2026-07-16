from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from kolega_code.agent.tool_backend.memory_tool import MemoryTool
from kolega_code.agent.tools import ToolCollection, ToolCollectionConfig
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.memory import (
    MemoryAccessScope,
    MemoryToolBinding,
    ProjectMemoryManager,
)
from kolega_code.tools import ToolError


def _config() -> AgentConfig:
    model = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model="test-model",
        rate_limits=RateLimitConfig(),
    )
    return AgentConfig(
        anthropic_api_key="test-key",
        long_context_config=model,
        fast_config=model,
        thinking_config=model,
    )


def _caller(
    manager: ProjectMemoryManager,
    *,
    sub_agent: bool = False,
    agent_name: str = "coder",
) -> Mock:
    caller = Mock()
    caller.agent_name = agent_name
    caller.sub_agent = sub_agent
    caller.memory_manager = manager
    caller.supports_vision = False
    caller.edit_protocol = None
    caller.primary_model_config = _config().long_context_config
    caller.session_id = None
    caller.custom_agent_catalog = None
    caller._initialize_system_prompt = Mock()
    return caller


@pytest.mark.asyncio
async def test_private_memory_facade_round_trips_credential_like_content(tmp_path: Path) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir()
    manager = ProjectMemoryManager(project, state)
    caller = _caller(manager)
    tool = MemoryTool(manager, caller)
    write = next(binding for binding in tool.bindings() if binding.name == "write_memory")
    read = next(binding for binding in tool.bindings() if binding.name == "read_memory")
    content = "API_KEY=supersecretvalue123"

    result = await tool.invoke(write, memory_content=content)
    read_result = await tool.invoke(read)

    assert "Project memory updated" in result
    assert content in read_result
    assert manager.read_entry("MEMORY.md").content == content
    caller._initialize_system_prompt.assert_called_once()


@pytest.mark.asyncio
async def test_failed_mutation_does_not_refresh_prompt(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    caller = _caller(manager)
    tool = MemoryTool(manager, caller)
    write = next(binding for binding in tool.bindings() if binding.name == "write_memory")

    with pytest.raises(Exception, match="replace requires expected_sha256"):
        await tool.invoke(write, memory_content="replacement", mode="replace")

    caller._initialize_system_prompt.assert_not_called()


@pytest.mark.asyncio
async def test_committed_mutation_is_not_reported_as_failed_when_prompt_refresh_fails(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    caller = _caller(manager)
    caller._initialize_system_prompt.side_effect = RuntimeError("refresh failed")
    tool = MemoryTool(manager, caller)
    write = next(binding for binding in tool.bindings() if binding.name == "write_memory")

    result = await tool.invoke(write, memory_content="stable fact")

    assert "Project memory updated" in result
    assert "mutation was committed" in result
    assert manager.read_entry("MEMORY.md").content == "stable fact"


@pytest.mark.asyncio
async def test_unexpected_backend_errors_do_not_expose_private_paths(
    tmp_path: Path,
) -> None:
    private_path = tmp_path / "private" / "MEMORY.md"

    def fail(path: str = "MEMORY.md") -> None:
        del path
        raise OSError(f"permission denied: {private_path}")

    manager = Mock()
    manager.tool_bindings.return_value = (
        MemoryToolBinding(
            "read_memory",
            {"name": "read_memory", "input_schema": {"type": "object"}},
            fail,
        ),
    )
    tool = MemoryTool(manager, Mock())

    with pytest.raises(ToolError) as caught:
        await tool.read_memory()
    assert str(private_path) not in str(caught.value)
    assert "without exposing private storage details" in str(caught.value)


def test_dynamic_bindings_and_subagent_access(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    top = _caller(manager)
    top_tools = ToolCollection(
        project,
        "workspace",
        "thread",
        Mock(),
        _config(),
        top,
        tool_config=ToolCollectionConfig(
            include_memory_tools=True,
            memory_write_access=True,
        ),
    )
    top_registry = top_tools.registry()
    assert {"read_memory", "list_memory", "write_memory", "delete_memory"} <= set(top_registry.names())
    assert top_registry.get("read_memory").parallel_safe
    assert top_registry.get("list_memory").parallel_safe
    assert not top_registry.get("write_memory").parallel_safe
    read_definition = top_registry.get("read_memory").definition
    list_definition = top_registry.get("list_memory").definition
    write_definition = top_registry.get("write_memory").definition
    delete_definition = top_registry.get("delete_memory").definition
    assert "MEMORY.md index in the system prompt" in read_definition.description
    assert "sha256 revision" in read_definition.description
    assert "case-insensitive substring of path or content" in list_definition.description
    assert "MEMORY.md index does not surface" in list_definition.description
    assert "Read the target first" in write_definition.description
    assert "current revision" in write_definition.description
    assert "one topic per file" in write_definition.description
    assert "Read the target first" in delete_definition.description
    assert "current revision" in delete_definition.description
    assert "stale or now authoritative" in delete_definition.description
    for definition in (read_definition, list_definition, write_definition, delete_definition):
        assert "compare-and-swap" not in definition.description
    assert read_definition.input_schema is not None
    assert list_definition.input_schema is not None
    assert write_definition.input_schema is not None
    assert delete_definition.input_schema is not None
    assert "MEMORY.md or list_memory" in read_definition.input_schema["properties"]["path"]["description"]
    assert "path and content" in list_definition.input_schema["properties"]["query"]["description"]
    assert "full replacement content" in write_definition.input_schema["properties"]["memory_content"]["description"]
    assert "topics/build.md" in write_definition.input_schema["properties"]["path"]["description"]
    assert "requires expected_sha256" in write_definition.input_schema["properties"]["mode"]["description"]
    assert delete_definition.input_schema["properties"]["path"]["description"] == "Memory file path to delete."
    write_revision = write_definition.input_schema["properties"]["expected_sha256"]
    delete_revision = delete_definition.input_schema["properties"]["expected_sha256"]
    assert "Current revision returned by read_memory" in write_revision["description"]
    assert "Current revision returned by read_memory" in delete_revision["description"]

    scoped = manager.with_scope(MemoryAccessScope.SUBAGENT)
    sub = _caller(scoped, sub_agent=True)
    sub_tools = ToolCollection(
        project,
        "workspace",
        "thread",
        Mock(),
        _config(),
        sub,
        tool_config=ToolCollectionConfig(include_memory_tools=True),
    )
    sub_registry = sub_tools.registry()
    assert "read_memory" in sub_registry
    assert "list_memory" in sub_registry
    assert sub_registry.get("read_memory").parallel_safe
    assert sub_registry.get("list_memory").parallel_safe
    assert "write_memory" not in sub_registry
    assert "delete_memory" not in sub_registry


@pytest.mark.asyncio
async def test_list_memory_output_formats_entries(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    tool = MemoryTool(manager, _caller(manager))

    assert await tool.list_memory() == "No memory entries found."
    assert await tool.list_memory("") == "No memory entries found."
    assert await tool.list_memory(" \t ") == "No memory entries found."
    assert await tool.list_memory("missing") == "No memory entries found matching 'missing'."

    index_content = "# Project Index\nBuild links"
    build_content = "# Build Notes\nUse the wrapper"
    plain_content = "Unindexed deployment detail"
    assert manager.append_entry("MEMORY.md", index_content).ok
    assert manager.append_entry("topics/build.md", build_content).ok
    assert manager.append_entry("topics/plain.md", plain_content).ok
    dates = {
        entry.reference: datetime.fromtimestamp(entry.modified_ns / 1_000_000_000, tz=UTC).strftime("%Y-%m-%d")
        for entry in manager.list_entries()
        if entry.modified_ns is not None
    }

    assert await tool.list_memory() == "\n".join(
        [
            "3 memory entries:",
            f"- MEMORY.md — Project Index ({len(index_content.encode()):,} bytes, modified {dates['MEMORY.md']})",
            (
                f"- topics/build.md — Build Notes ({len(build_content.encode()):,} bytes, "
                f"modified {dates['topics/build.md']})"
            ),
            (f"- topics/plain.md ({len(plain_content.encode()):,} bytes, modified {dates['topics/plain.md']})"),
        ]
    )
    assert await tool.list_memory(" bUiLd ") == "\n".join(
        [
            "2 memory entries matching 'bUiLd':",
            f"- MEMORY.md — Project Index ({len(index_content.encode()):,} bytes, modified {dates['MEMORY.md']})",
            (
                f"- topics/build.md — Build Notes ({len(build_content.encode()):,} bytes, "
                f"modified {dates['topics/build.md']})"
            ),
        ]
    )


def test_memory_write_access_flag_gates_writes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")

    renamed = ToolCollection(
        project,
        "workspace",
        "thread",
        Mock(),
        _config(),
        _caller(manager, agent_name="renamed-custom"),
        tool_config=ToolCollectionConfig(
            include_memory_tools=True,
            memory_write_access=True,
        ),
    )
    renamed_names = set(renamed.registry().names())
    assert {"write_memory", "delete_memory"} <= renamed_names

    coder = ToolCollection(
        project,
        "workspace",
        "thread",
        Mock(),
        _config(),
        _caller(manager),
        tool_config=ToolCollectionConfig(
            include_memory_tools=True,
            memory_write_access=False,
        ),
    )
    coder_names = set(coder.registry().names())
    assert "write_memory" not in coder_names
    assert "delete_memory" not in coder_names

    scoped = manager.with_scope(MemoryAccessScope.SUBAGENT)
    subagent = ToolCollection(
        project,
        "workspace",
        "thread",
        Mock(),
        _config(),
        _caller(scoped, sub_agent=True),
        tool_config=ToolCollectionConfig(
            include_memory_tools=True,
            memory_write_access=True,
        ),
    )
    subagent_names = set(subagent.registry().names())
    assert {"read_memory", "list_memory"} <= subagent_names
    assert "write_memory" not in subagent_names
    assert "delete_memory" not in subagent_names

    mismatched_subagent = ToolCollection(
        project,
        "workspace",
        "thread",
        Mock(),
        _config(),
        _caller(manager, sub_agent=True),
        tool_config=ToolCollectionConfig(
            include_memory_tools=True,
            memory_write_access=True,
        ),
    )
    mismatched_names = set(mismatched_subagent.registry().names())
    assert {"read_memory", "list_memory"} <= mismatched_names
    assert "write_memory" not in mismatched_names
    assert "delete_memory" not in mismatched_names


@pytest.mark.asyncio
async def test_index_budget_warning_in_write_output(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    tool = MemoryTool(manager, _caller(manager))
    content = "".join(f"line {index}\n" for index in range(201))

    output = await tool.write_memory(content)

    assert "only the first 200 lines" in output


def test_exact_allowlist_and_disabled_manager_are_final(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    caller = _caller(manager)
    tools = ToolCollection(
        project,
        "workspace",
        "thread",
        Mock(),
        _config(),
        caller,
        tool_config=ToolCollectionConfig(
            include_memory_tools=True,
            allowed_tools=["read_memory"],
        ),
    )
    assert tools.registry().names() == ["read_memory"]

    manager.set_enabled(False)
    assert "read_memory" not in tools.registry()
