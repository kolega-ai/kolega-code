"""Tests for the declarative param-alias layer (kolega_code.tools.param_aliases).

Table-driven: every registry entry needs exactly one row in ALIAS_CASES (and a
CANONICAL_WINS_CASES row for renames). The tests run the real dispatch choke
point (Tool.call) against handlers carrying the real ToolCollection signatures,
so an alias that would still trip argument validation fails here.
"""

import inspect

import pytest

from kolega_code.agent.tools import ToolCollection
from kolega_code.llm.models import ToolDefinition
from kolega_code.tools import Tool, ToolError
from kolega_code.tools.param_aliases import PARAM_ALIASES

# One row per registry alias: (tool, call kwargs, kwargs the handler receives).
ALIAS_CASES = [
    # timeout unit heuristic: < 1000 reads as seconds, >= 1000 as milliseconds
    ("exec_command", {"command": "ls", "timeout": 60}, {"command": "ls", "yield_time_ms": 60000}),
    ("exec_command", {"command": "ls", "timeout": 5000}, {"command": "ls", "yield_time_ms": 5000}),
    ("exec_command", {"command": "ls", "title": "list files"}, {"command": "ls"}),
    ("eval", {"command": "print(1)"}, {"code": "print(1)"}),
]

# One row per rename alias: alias + canonical together -> canonical wins.
CANONICAL_WINS_CASES = [
    ("exec_command", {"command": "ls", "timeout": 60, "yield_time_ms": 500}, {"command": "ls", "yield_time_ms": 500}),
    ("eval", {"command": "print(1)", "code": "print(2)"}, {"code": "print(2)"}),
]


def _real_signature(tool_name: str) -> inspect.Signature:
    """The model-facing signature: the ToolCollection method minus ``self``."""
    signature = inspect.signature(getattr(ToolCollection, tool_name))
    return signature.replace(parameters=[p for p in signature.parameters.values() if p.name != "self"])


def _capture_tool(tool_name: str, received: dict) -> Tool:
    """A Tool whose handler records its kwargs but binds like the real tool."""

    async def handler(**kwargs):
        received.update(kwargs)
        return "ok"

    handler.__signature__ = _real_signature(tool_name)
    return Tool(
        name=tool_name,
        definition=ToolDefinition(name=tool_name, description=f"{tool_name} tool", parameters=[]),
        handler=handler,
    )


def test_every_registry_entry_has_a_case_row():
    registered = {(tool, alias) for tool, aliases in PARAM_ALIASES.items() for alias in aliases}
    covered = {
        (tool, alias)
        for tool, call_kwargs, _ in ALIAS_CASES
        for alias in call_kwargs
        if alias in PARAM_ALIASES.get(tool, {})
    }
    assert registered == covered, "each PARAM_ALIASES entry needs exactly one ALIAS_CASES row"


def test_aliases_target_real_parameters_and_never_shadow_them():
    for tool_name, aliases in PARAM_ALIASES.items():
        parameters = _real_signature(tool_name).parameters
        for alias_name, alias in aliases.items():
            # An alias that becomes a real parameter must be deleted from the
            # registry, or it would silently rewrite legitimate calls.
            assert alias_name not in parameters, f"alias {tool_name}.{alias_name} shadows a real parameter"
            if alias.canonical is not None:
                assert alias.canonical in parameters, f"{tool_name}.{alias_name} renames to unknown `{alias.canonical}`"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name, call_kwargs, expected_kwargs", ALIAS_CASES)
async def test_aliased_call_behaves_as_canonical(tool_name, call_kwargs, expected_kwargs):
    received = {}
    tool = _capture_tool(tool_name, received)
    assert await tool.call(**call_kwargs) == "ok"
    assert received == expected_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name, call_kwargs, expected_kwargs", CANONICAL_WINS_CASES)
async def test_canonical_wins_over_alias(tool_name, call_kwargs, expected_kwargs):
    received = {}
    tool = _capture_tool(tool_name, received)
    assert await tool.call(**call_kwargs) == "ok"
    assert received == expected_kwargs


@pytest.mark.asyncio
async def test_unusable_alias_value_is_dropped_not_fatal():
    received = {}
    tool = _capture_tool("exec_command", received)
    assert await tool.call(command="ls", timeout="a while") == "ok"
    assert received == {"command": "ls"}


@pytest.mark.asyncio
async def test_unregistered_unknown_param_still_errors():
    # Aliasing must not widen anything else: an unknown name outside the
    # registry errors exactly as before, even on a tool that has aliases.
    tool = _capture_tool("exec_command", {})
    with pytest.raises(ToolError) as exc_info:
        await tool.call(command="ls", bogus=1)
    assert "unexpected keyword argument" in str(exc_info.value)
    assert "Invalid arguments for `exec_command`" in str(exc_info.value)


@pytest.mark.asyncio
async def test_alias_hit_emits_debug_log():
    messages = []

    async def log_debug(message: str) -> None:
        messages.append(message)

    async def handler(code: str):
        return code

    tool = Tool(
        name="eval",
        definition=ToolDefinition(name="eval", description="eval tool", parameters=[]),
        handler=handler,
        log_debug=log_debug,
    )
    assert await tool.call(command="print(1)") == "print(1)"
    assert messages == ["Param alias on `eval`: `command` renamed to `code`"]
