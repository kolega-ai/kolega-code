import errno
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.tool_backend.memory_tool import MemoryTool
from kolega_code.agent.tools import ToolCollection, ToolCollectionConfig
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.memory import MemoryAccessScope, MemoryToolBinding, ProjectMemoryManager
from kolega_code.tools import ToolError, ToolInputError


MEMORY_TOOL_NAMES = [
    "read_memory",
    "list_memory",
    "write_memory",
    "edit_memory",
    "delete_memory",
]


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


def _tools(
    project: Path,
    manager: ProjectMemoryManager,
    *,
    sub_agent: bool = False,
    include_memory_tools: bool = True,
    write_access: bool = True,
    allowed_tools: list[str] | None = None,
) -> ToolCollection:
    connection_manager = Mock()
    connection_manager.broadcast_event = AsyncMock()
    return ToolCollection(
        project,
        "workspace",
        "thread",
        connection_manager,
        _config(),
        _caller(manager, sub_agent=sub_agent),
        tool_config=ToolCollectionConfig(
            include_memory_tools=include_memory_tools,
            memory_write_access=write_access,
            allowed_tools=allowed_tools,
        ),
    )


@pytest.mark.asyncio
async def test_revision_free_read_write_and_delete_outputs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    caller = _caller(manager)
    tool = MemoryTool(manager, caller)
    content = "API_KEY=supersecretvalue123"

    write_result = await tool.write_memory(content)
    read_result = await tool.read_memory()
    delete_result = await tool.delete_memory("MEMORY.md")

    assert write_result == f"Project memory updated: `MEMORY.md` ({len(content)} bytes)."
    assert read_result == f"Memory `MEMORY.md` ({len(content)} bytes):\n\n{content}"
    assert delete_result == "Project memory deleted: `MEMORY.md` (0 bytes)."
    assert not manager.read_entry("MEMORY.md").present
    assert caller._initialize_system_prompt.call_count == 2
    for output in (write_result, read_result, delete_result):
        lowered = output.lower()
        assert "revision" not in lowered
        assert "sha256" not in lowered
        assert "hash" not in lowered


@pytest.mark.asyncio
async def test_edit_memory_replaces_one_exact_unique_occurrence(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    assert manager.write_entry("topics/build.md", "Use uv. Run pytest.").ok
    caller = _caller(manager)
    tool = MemoryTool(manager, caller)

    result = await tool.edit_memory("Run pytest.", "Run the focused pytest file.", "topics/build.md")

    expected = "Use uv. Run the focused pytest file."
    assert result == f"Project memory updated: `topics/build.md` ({len(expected)} bytes)."
    assert manager.read_entry("topics/build.md").content == expected
    caller._initialize_system_prompt.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "old_string", "error"),
    [
        ("alpha", "", "old_string must not be empty"),
        ("alpha", "missing", "old_string was not found"),
        ("repeat / repeat", "repeat", "old_string appears 2 times"),
    ],
)
async def test_edit_memory_failure_leaves_file_unchanged_and_does_not_refresh_prompt(
    tmp_path: Path,
    content: str,
    old_string: str,
    error: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    assert manager.write_entry("MEMORY.md", content).ok
    caller = _caller(manager)
    tool = MemoryTool(manager, caller)

    with pytest.raises(ToolError, match=error):
        await tool.edit_memory(old_string, "replacement")

    assert manager.read_entry("MEMORY.md").content == content
    caller._initialize_system_prompt.assert_not_called()


@pytest.mark.asyncio
async def test_existing_linked_equivalent_fact_requires_zero_mutations(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    index = "- [Release workflow](topics/release-workflow.md): Carry releases through completion.\n"
    topic = "# Release workflow\n\nRun release checks and finish the release with minimal user intervention.\n"
    assert manager.write_entry("MEMORY.md", index).ok
    assert manager.write_entry("topics/release-workflow.md", topic).ok
    before = manager.list_entries()
    caller = _caller(manager)
    tool = MemoryTool(manager, caller)

    startup_context = manager.prompt_context().text
    recalled = await tool.read_memory("topics/release-workflow.md")

    assert "topics/release-workflow.md" in startup_context
    assert "minimal user intervention" in recalled
    assert manager.read_entry("MEMORY.md").content == index
    assert manager.read_entry("topics/release-workflow.md").content == topic
    assert manager.list_entries() == before
    caller._initialize_system_prompt.assert_not_called()


@pytest.mark.asyncio
async def test_concise_memory_is_one_index_edit(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    original = "# Project memory\n\n- Use uv for Python commands.\n"
    assert manager.write_entry("MEMORY.md", original).ok
    caller = _caller(manager)
    tool = MemoryTool(manager, caller)

    await tool.edit_memory(
        "- Use uv for Python commands.",
        "- Use uv for Python commands.\n- Run focused tests before the full suite.",
    )

    assert manager.read_entry("MEMORY.md").content == (
        "# Project memory\n\n- Use uv for Python commands.\n- Run focused tests before the full suite.\n"
    )
    assert [call.args for call in caller._initialize_system_prompt.call_args_list] == [()]


@pytest.mark.asyncio
async def test_detailed_topic_is_written_before_index_is_edited(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    assert manager.write_entry("MEMORY.md", "# Project memory\n").ok
    caller = _caller(manager)
    tool = MemoryTool(manager, caller)
    topic = "# Build workflow\n\nUse `uv run` so the locked environment is respected.\n"

    write_result = await tool.write_memory(topic, "topics/build.md")
    edit_result = await tool.edit_memory(
        "# Project memory",
        "# Project memory\n\n- [Build workflow](topics/build.md): Use the locked Python environment.",
    )

    assert write_result.startswith("Project memory updated: `topics/build.md`")
    assert edit_result.startswith("Project memory updated: `MEMORY.md`")
    assert manager.read_entry("topics/build.md").content == topic
    assert manager.read_entry("MEMORY.md").content == (
        "# Project memory\n\n- [Build workflow](topics/build.md): Use the locked Python environment.\n"
    )
    assert caller._initialize_system_prompt.call_count == 2


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

    result = await tool.write_memory("stable fact")

    assert "Project memory updated" in result
    assert "mutation was committed" in result
    assert manager.read_entry("MEMORY.md").content == "stable fact"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("make_error", "kind"),
    [
        (lambda path: OSError(f"permission denied: {path}"), "(OSError)"),
        (
            lambda path: PermissionError(errno.EACCES, "Permission denied", str(path)),
            "(PermissionError: Permission denied)",
        ),
    ],
    ids=["message-only", "errno"],
)
async def test_unexpected_backend_errors_name_the_failure_and_log_it_without_exposing_paths(
    tmp_path: Path,
    make_error: Callable[[Path], OSError],
    kind: str,
) -> None:
    private_path = tmp_path / "private" / "MEMORY.md"

    def fail(path: str = "MEMORY.md") -> None:
        del path
        raise make_error(private_path)

    manager = Mock()
    manager.tool_bindings.return_value = (
        MemoryToolBinding(
            "read_memory",
            {
                "name": "read_memory",
                "description": "Read memory.",
                "input_schema": {"type": "object"},
            },
            fail,
        ),
    )
    caller = Mock()
    caller.log_error = AsyncMock()
    tool = MemoryTool(manager, caller)

    with pytest.raises(ToolError) as caught:
        await tool.read_memory()

    assert str(private_path) not in str(caught.value)
    assert kind in str(caught.value)
    assert "local session log" in str(caught.value)
    caller.log_error.assert_awaited_once()
    logged = caller.log_error.await_args.args[0]
    assert logged.startswith("Project memory read_memory failed unexpectedly:")
    assert "Traceback" in logged
    assert str(private_path) in logged


@pytest.mark.asyncio
async def test_dispatch_aliases_file_path_for_memory_tools(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    registry = _tools(project, manager).registry()

    write_result = await registry.call("write_memory", content="alpha", file_path="topics/notes.md")
    edit_result = await registry.call("edit_memory", old_string="alpha", new_string="beta", file_path="topics/notes.md")
    read_result = await registry.call("read_memory", file_path="topics/notes.md")
    delete_result = await registry.call("delete_memory", file_path="topics/notes.md")

    assert write_result.startswith("Project memory updated: `topics/notes.md`")
    assert edit_result.startswith("Project memory updated: `topics/notes.md`")
    assert read_result == "Memory `topics/notes.md` (4 bytes):\n\nbeta"
    assert delete_result.startswith("Project memory deleted: `topics/notes.md`")
    assert not manager.read_entry("topics/notes.md").present


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "inputs", "problem", "usage"),
    [
        (
            "write_memory",
            {"path": "MEMORY.md"},
            "missing a required argument: 'content'",
            "write_memory(content, path='MEMORY.md')",
        ),
        (
            "write_memory",
            {"content": "fact", "title": "Fact"},
            "got an unexpected keyword argument 'title'",
            "write_memory(content, path='MEMORY.md')",
        ),
        (
            "edit_memory",
            {"old_string": "a"},
            "missing a required argument: 'new_string'",
            "edit_memory(old_string, new_string, path='MEMORY.md')",
        ),
    ],
)
async def test_dispatch_rejects_malformed_memory_arguments_with_usage(
    tmp_path: Path,
    tool_name: str,
    inputs: dict[str, str],
    problem: str,
    usage: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    registry = _tools(project, manager).registry()

    with pytest.raises(ToolInputError) as caught:
        await registry.call(tool_name, **inputs)

    assert problem in str(caught.value)
    assert f"Usage: {usage}" in str(caught.value)
    assert not manager.read_entry("MEMORY.md").present


@pytest.mark.asyncio
async def test_non_text_memory_content_is_a_memory_error(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    registry = _tools(project, manager).registry()

    with pytest.raises(ToolError, match="Project memory error: memory content must be text"):
        await registry.call("write_memory", content={"fact": "alpha"})


def test_tool_schema_order_availability_and_revision_free_file_api(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    registry = _tools(project, manager).registry()
    memory_names = [name for name in registry.names() if name in MEMORY_TOOL_NAMES]

    assert memory_names == MEMORY_TOOL_NAMES
    assert registry.get("read_memory").parallel_safe
    assert registry.get("list_memory").parallel_safe
    assert all(not registry.get(name).parallel_safe for name in MEMORY_TOOL_NAMES[2:])

    definitions = {name: registry.get(name).definition for name in MEMORY_TOOL_NAMES}
    schemas: dict[str, dict] = {}
    for name, definition in definitions.items():
        assert definition.input_schema is not None
        schemas[name] = definition.input_schema
    assert list(schemas["read_memory"]["properties"]) == ["path"]
    assert schemas["read_memory"]["properties"]["path"]["default"] == "MEMORY.md"
    assert list(schemas["list_memory"]["properties"]) == ["query"]
    assert list(schemas["write_memory"]["properties"]) == ["content", "path"]
    assert schemas["write_memory"]["required"] == ["content"]
    assert schemas["write_memory"]["properties"]["path"]["default"] == "MEMORY.md"
    assert list(schemas["edit_memory"]["properties"]) == ["old_string", "new_string", "path"]
    assert schemas["edit_memory"]["required"] == ["old_string", "new_string"]
    assert schemas["edit_memory"]["properties"]["path"]["default"] == "MEMORY.md"
    assert list(schemas["delete_memory"]["properties"]) == ["path"]
    assert schemas["delete_memory"]["required"] == ["path"]

    rendered = repr([(definition.description, definition.input_schema) for definition in definitions.values()])
    for removed in ("mode", "expected_sha256", "revision", "sha256", "hash"):
        assert removed not in rendered.lower()
    assert "MEMORY.md index in the system prompt" in definitions["read_memory"].description
    assert "case-insensitive substring of path or content" in definitions["list_memory"].description
    assert "do nothing when the fact is already covered" in definitions["write_memory"].description
    assert "exact, unique" in definitions["edit_memory"].description
    assert "remove its link from MEMORY.md before deleting" in definitions["delete_memory"].description


def test_include_memory_tools_false_removes_all_memory_tools(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")

    enabled = _tools(project, manager, include_memory_tools=True).registry()
    disabled = _tools(project, manager, include_memory_tools=False).registry()

    assert [name for name in enabled.names() if name in MEMORY_TOOL_NAMES] == MEMORY_TOOL_NAMES
    assert not any(name in MEMORY_TOOL_NAMES for name in disabled.names())
    # Non-memory tools are unaffected by the toggle.
    assert "read" in disabled.names()


def test_subagent_exposes_only_read_and_list_even_when_write_is_requested(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    scoped = manager.with_scope(MemoryAccessScope.SUBAGENT)

    scoped_registry = _tools(project, scoped, sub_agent=True, write_access=True).registry()
    mismatched_registry = _tools(project, manager, sub_agent=True, write_access=True).registry()

    for registry in (scoped_registry, mismatched_registry):
        names = [name for name in registry.names() if name in MEMORY_TOOL_NAMES]
        assert names == ["read_memory", "list_memory"]
        assert registry.get("read_memory").parallel_safe
        assert registry.get("list_memory").parallel_safe


@pytest.mark.asyncio
async def test_list_memory_output_formats_revision_free_entries(tmp_path: Path) -> None:
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
    assert manager.write_entry("MEMORY.md", index_content).ok
    assert manager.write_entry("topics/build.md", build_content).ok
    assert manager.write_entry("topics/plain.md", plain_content).ok
    dates = {
        entry.reference: datetime.fromtimestamp(entry.modified_ns / 1_000_000_000, tz=UTC).strftime("%Y-%m-%d")
        for entry in manager.list_entries()
        if entry.modified_ns is not None
    }

    output = await tool.list_memory()
    assert output == "\n".join(
        [
            "3 memory entries:",
            f"- MEMORY.md — Project Index ({len(index_content.encode()):,} bytes, modified {dates['MEMORY.md']})",
            (
                f"- topics/build.md — Build Notes ({len(build_content.encode()):,} bytes, "
                f"modified {dates['topics/build.md']})"
            ),
            f"- topics/plain.md ({len(plain_content.encode()):,} bytes, modified {dates['topics/plain.md']})",
        ]
    )
    filtered = await tool.list_memory(" bUiLd ")
    assert filtered == "\n".join(
        [
            "2 memory entries matching 'bUiLd':",
            f"- MEMORY.md — Project Index ({len(index_content.encode()):,} bytes, modified {dates['MEMORY.md']})",
            (
                f"- topics/build.md — Build Notes ({len(build_content.encode()):,} bytes, "
                f"modified {dates['topics/build.md']})"
            ),
        ]
    )
    assert "revision" not in (output + filtered).lower()
    assert "sha256" not in (output + filtered).lower()


def test_write_access_flag_allowlist_and_disabled_manager_are_dynamic(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")

    writable = _tools(project, manager, write_access=True)
    assert [name for name in writable.registry().names() if name in MEMORY_TOOL_NAMES] == MEMORY_TOOL_NAMES

    read_only = _tools(project, manager, write_access=False)
    assert [name for name in read_only.registry().names() if name in MEMORY_TOOL_NAMES] == [
        "read_memory",
        "list_memory",
    ]

    allowlisted = _tools(project, manager, allowed_tools=["read_memory"])
    assert allowlisted.registry().names() == ["read_memory"]

    manager.set_enabled(False)
    assert not any(name in MEMORY_TOOL_NAMES for name in writable.registry().names())
    manager.set_enabled(True)
    assert [name for name in writable.registry().names() if name in MEMORY_TOOL_NAMES] == MEMORY_TOOL_NAMES


@pytest.mark.asyncio
async def test_binding_rejects_source_backend_swap_and_fresh_binding_succeeds(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    tool = MemoryTool(manager, _caller(manager))
    stale_read = next(binding for binding in tool.bindings() if binding.name == "read_memory")

    manager.select_backend("markdown", {"generation": 2})

    with pytest.raises(ToolError, match="configuration changed; refresh tools and retry"):
        await tool.invoke(stale_read)
    assert await tool.read_memory() == "Memory `MEMORY.md` is missing (bytes: 0)."


@pytest.mark.asyncio
async def test_index_budget_warning_in_write_output(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    tool = MemoryTool(manager, _caller(manager))
    content = "".join(f"line {index}\n" for index in range(201))

    output = await tool.write_memory(content)

    assert "only the first 200 lines" in output
