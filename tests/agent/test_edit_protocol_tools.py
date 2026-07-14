from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.tools import ToolCollection, ToolCollectionConfig
from kolega_code.config import AgentConfig, EditProtocol, ModelConfig, ModelProvider


def collection(
    tmp_path: Path,
    protocol: EditProtocol,
    *,
    read_only: bool = False,
    allowed_tools: list[str] | None = None,
) -> ToolCollection:
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
        tool_config=ToolCollectionConfig(read_only=read_only, allowed_tools=allowed_tools),
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


def test_claude_protocol_reuses_lowercase_names_with_exact_string_schema(tmp_path: Path) -> None:
    registry = collection(tmp_path, EditProtocol.CLAUDE_CODE).registry()

    assert {"edit", "write"}.issubset(registry.names())
    assert "multi_edit" not in registry
    assert "apply_patch" not in registry
    edit_parameters = registry.get("edit").definition.parameters
    write_parameters = registry.get("write").definition.parameters
    assert {parameter.name for parameter in edit_parameters} == {
        "file_path",
        "old_string",
        "new_string",
        "replace_all",
    }
    assert {parameter.name for parameter in edit_parameters if parameter.required} == {
        "file_path",
        "old_string",
        "new_string",
    }
    assert {parameter.name for parameter in write_parameters} == {"file_path", "content"}
    assert all(parameter.required for parameter in write_parameters)


def test_read_only_agent_never_gets_claude_edit_tools(tmp_path: Path) -> None:
    registry = collection(tmp_path, EditProtocol.CLAUDE_CODE, read_only=True).registry()

    assert "edit" not in registry
    assert "write" not in registry


def test_hashline_protocol_exposes_edit_write_and_nested_schema(tmp_path: Path) -> None:
    registry = collection(tmp_path, EditProtocol.HASHLINE_V2).registry()

    assert {"edit", "write"}.issubset(registry.names())
    assert "multi_edit" not in registry
    assert "apply_patch" not in registry
    definition = registry.get("edit").definition
    assert "1#BM:MAX_RETRIES = 3" in definition.description
    assert "display-only metadata" in definition.description
    schema = definition.input_schema
    assert schema is not None
    assert schema["required"] == ["path", "edits"]
    assert len(schema["properties"]["edits"]["items"]["anyOf"]) == 5
    assert "only CONTENT to content fields" in schema["properties"]["edits"]["description"]
    declarations = registry.get("edit").definition.to_google().function_declarations
    assert declarations is not None
    assert declarations[0].parameters is not None


@pytest.mark.asyncio
async def test_hashline_protocol_adds_anchors_to_read_and_search(tmp_path: Path) -> None:
    tools = collection(tmp_path, EditProtocol.HASHLINE_V2)
    (tmp_path / "sample.py").write_text("def run():\n    return 42\n")

    entire = await tools.read_entire_file("sample.py")
    section = await tools.read_file_section("sample.py", 2, 2)
    search = await tools.search_codebase("return", case_sensitive=True)

    assert "1#" in entire and ":def run():" in entire
    assert "2#" in section and ":    return 42" in section
    assert "2#" in search and ":    return 42" in search


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocol",
    [EditProtocol.SEARCH_REPLACE, EditProtocol.CODEX_APPLY_PATCH, EditProtocol.CLAUDE_CODE],
)
async def test_other_protocols_never_add_hashline_anchors(tmp_path: Path, protocol: EditProtocol) -> None:
    tools = collection(tmp_path, protocol)
    (tmp_path / "sample.py").write_text("def run():\n    return 42\n")

    entire = await tools.read_entire_file("sample.py")
    search = await tools.search_codebase("return", case_sensitive=True)

    assert "1#" not in entire and "2#" not in entire
    assert "2#" not in search
    assert "Line 2: return 42" in search


@pytest.mark.asyncio
async def test_read_only_hashline_collection_does_not_emit_unusable_anchors(tmp_path: Path) -> None:
    tools = collection(tmp_path, EditProtocol.HASHLINE_V2, read_only=True)
    (tmp_path / "sample.py").write_text("one\ntwo\n")

    assert "edit" not in tools.registry()
    assert "1#" not in await tools.read_entire_file("sample.py")


@pytest.mark.asyncio
async def test_restricted_hashline_collection_does_not_emit_unusable_anchors(tmp_path: Path) -> None:
    tools = collection(
        tmp_path,
        EditProtocol.HASHLINE_V2,
        allowed_tools=["read_entire_file", "search_codebase"],
    )
    (tmp_path / "sample.py").write_text("one\ntwo\n")

    assert "edit" not in tools.registry()
    assert "1#" not in await tools.read_entire_file("sample.py")
    assert "Line 2: two" in await tools.search_codebase("two", case_sensitive=True)


@pytest.mark.asyncio
async def test_hashline_read_never_anchors_partial_or_phantom_lines(tmp_path: Path) -> None:
    tools = collection(tmp_path, EditProtocol.HASHLINE_V2)
    (tmp_path / "long.txt").write_text("x" * 100_050)
    (tmp_path / "section.txt").write_text("one\ntwo\nthree\n")

    long_output = await tools.read_entire_file("long.txt")
    section = await tools.read_file_section("section.txt", 1, 2)

    assert "1#" not in long_output
    assert "File truncated by size" in long_output
    assert "1#" in section and "2#" in section
    assert "3#" not in section


@pytest.mark.asyncio
async def test_hashline_line_one_anchor_ignores_bom_and_can_be_applied(tmp_path: Path) -> None:
    tools = collection(tmp_path, EditProtocol.HASHLINE_V2)
    tools.edit_tool._snapshot_service = None
    target = tmp_path / "bom.txt"
    target.write_text("\ufeffbefore\nafter\n")

    read = await tools.read_entire_file("bom.txt")
    tagged_line = next(line for line in read.splitlines() if line.startswith("1#"))
    tag, displayed = tagged_line.split(":", 1)
    result = await tools.call(
        "edit",
        path="bom.txt",
        edits=[{"op": "set", "tag": tag, "content": ["updated"]}],
    )

    assert displayed == "before"
    assert result == "Updated bom.txt"
    assert target.read_text() == "\ufeffupdated\nafter\n"
