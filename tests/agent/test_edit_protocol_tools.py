from pathlib import Path
from unittest.mock import AsyncMock, Mock

from kolega_code.agent.tools import ToolCollection, ToolCollectionConfig
from kolega_code.config import AgentConfig, EditProtocol, ModelConfig, ModelProvider


def collection(tmp_path: Path, protocol: EditProtocol, *, read_only: bool = False) -> ToolCollection:
    model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="test-model")
    config = AgentConfig(
        anthropic_api_key="test",
        long_context_config=model,
        fast_config=model,
        thinking_config=model,
        edit_protocol=protocol,
    )
    caller = Mock()
    caller.agent_name = "test-agent"
    caller.supports_vision = False
    caller.sub_agent = False
    caller.session_id = "session"
    return ToolCollection(
        tmp_path,
        "workspace",
        "thread",
        AsyncMock(),
        config,
        caller,
        tool_config=ToolCollectionConfig(read_only=read_only),
    )


def test_default_protocol_keeps_search_replace_tools(tmp_path: Path) -> None:
    registry = collection(tmp_path, EditProtocol.SEARCH_REPLACE).registry()

    assert {"edit", "multi_edit", "write"}.issubset(registry.names())
    assert "apply_patch" not in registry


def test_codex_protocol_replaces_ordinary_write_tools(tmp_path: Path) -> None:
    registry = collection(tmp_path, EditProtocol.CODEX_APPLY_PATCH).registry()

    assert "apply_patch" in registry
    assert not {"edit", "multi_edit", "write"}.intersection(registry.names())
    assert "lsp_edit" in registry
    definition = registry.get("apply_patch").definition
    assert definition.input_kind == "freeform"
    assert definition.freeform_format is not None
    assert definition.freeform_format["syntax"] == "lark"


def test_read_only_agent_never_gets_apply_patch(tmp_path: Path) -> None:
    registry = collection(tmp_path, EditProtocol.CODEX_APPLY_PATCH, read_only=True).registry()

    assert "apply_patch" not in registry
