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


def _caller(manager: ProjectMemoryManager, *, sub_agent: bool = False) -> Mock:
    caller = Mock()
    caller.agent_name = "coder"
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
async def test_private_memory_facade_persists_outside_repository(tmp_path: Path) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir()
    manager = ProjectMemoryManager(project, state)
    caller = _caller(manager)
    tool = MemoryTool(manager, caller)
    write = next(binding for binding in tool.bindings() if binding.name == "write_memory")

    result = await tool.invoke(write, memory_content="stable fact")

    assert "Project memory updated" in result
    assert not (project / "AGENT_MEMORY.md").exists()
    assert manager.read_entry("MEMORY.md").content == "stable fact"
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
        tool_config=ToolCollectionConfig(include_memory_tools=True),
    )
    assert {"read_memory", "write_memory", "delete_memory"} <= set(top_tools.registry().names())
    assert top_tools.registry().get("read_memory").parallel_safe
    assert not top_tools.registry().get("write_memory").parallel_safe

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
    assert "read_memory" in sub_tools.registry()
    assert "write_memory" not in sub_tools.registry()
    assert "delete_memory" not in sub_tools.registry()


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
