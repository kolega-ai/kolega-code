from pathlib import Path

import pytest

from benchmarks.edit_tools.protocols import ProtocolAdapter, create_tool_collection, execute_call, get_protocol
from kolega_code.agent.tools import ToolExtension
from kolega_code.config import AgentConfig, EditProtocol, ModelConfig, ModelProvider
from kolega_code.llm.models import ToolCall, ToolDefinition
from kolega_code.services.lsp.config import LspConfig


def config(protocol: EditProtocol) -> AgentConfig:
    model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5-20251001")
    return AgentConfig(
        anthropic_api_key="test",
        long_context_config=model,
        fast_config=model,
        thinking_config=model,
        edit_protocol=protocol,
        lsp=LspConfig(enabled=False),
    )


@pytest.mark.asyncio
async def test_search_replace_adapter_uses_production_tools(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("before\n")
    adapter = get_protocol("search_replace")
    collection, _, caller = create_tool_collection(workspace, config(EditProtocol.SEARCH_REPLACE), adapter, tmp_path)
    definitions = {item.name: item for item in adapter.definitions(collection)}
    call = ToolCall(
        id="call-1",
        name="edit",
        input={
            "path": "a.txt",
            "block": "<<<<<<< SEARCH\nbefore\n=======\nafter\n>>>>>>> REPLACE",
        },
    )
    try:
        result, attempt = await execute_call(collection, caller, call, definitions, 1)
    finally:
        await collection.cleanup()

    assert not result.is_error
    assert attempt.apply_ok
    assert (workspace / "a.txt").read_text() == "after\n"


@pytest.mark.asyncio
async def test_codex_adapter_normalizes_json_fallback_and_applies_patch(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("before\n")
    adapter = get_protocol("codex_apply_patch")
    collection, _, caller = create_tool_collection(workspace, config(EditProtocol.CODEX_APPLY_PATCH), adapter, tmp_path)
    definitions = {item.name: item for item in adapter.definitions(collection)}
    patch = "*** Begin Patch\n*** Update File: a.txt\n@@\n-before\n+after\n*** End Patch\n"
    call = ToolCall(id="call-1", name="apply_patch", input={"input": patch})
    try:
        result, attempt = await execute_call(collection, caller, call, definitions, 1)
    finally:
        await collection.cleanup()

    assert not result.is_error
    assert result.input_kind == "freeform"
    assert attempt.input_kind == "freeform"
    assert (workspace / "a.txt").read_text() == "after\n"


@pytest.mark.asyncio
async def test_malformed_freeform_envelope_is_a_parse_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("before\n")
    adapter = get_protocol("codex_apply_patch")
    collection, _, caller = create_tool_collection(workspace, config(EditProtocol.CODEX_APPLY_PATCH), adapter, tmp_path)
    definitions = {item.name: item for item in adapter.definitions(collection)}
    call = ToolCall(id="call-1", name="apply_patch", input={"input": "x", "extra": "y"})
    try:
        result, attempt = await execute_call(collection, caller, call, definitions, 1)
    finally:
        await collection.cleanup()

    assert result.is_error
    assert not attempt.parse_ok
    assert not attempt.apply_ok


@pytest.mark.asyncio
async def test_research_adapter_can_run_without_production_edit_enum(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("before\n")

    def extension_factory(root: Path) -> ToolExtension:
        async def candidate_edit(input: str) -> str:
            (root / "a.txt").write_text(input)
            return "updated"

        return ToolExtension(name="candidate", tools={"candidate_edit": candidate_edit})

    def definition_factory() -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="candidate_edit",
                description="Replace a.txt with the raw input.",
                parameters=[],
                input_kind="freeform",
            )
        ]

    adapter = ProtocolAdapter(
        id="candidate",
        version="draft-1",
        production_protocol=None,
        tool_names=("candidate_edit",),
        capabilities=frozenset({"update"}),
        extension_factory=extension_factory,
        definition_factory=definition_factory,
    )
    collection, _, caller = create_tool_collection(workspace, config(EditProtocol.SEARCH_REPLACE), adapter, tmp_path)
    definitions = {item.name: item for item in adapter.definitions(collection)}
    call = ToolCall(id="call-1", name="candidate_edit", input={"input": "after\n"})
    try:
        result, attempt = await execute_call(collection, caller, call, definitions, 1)
    finally:
        await collection.cleanup()

    assert not result.is_error
    assert attempt.apply_ok
    assert (workspace / "a.txt").read_text() == "after\n"
