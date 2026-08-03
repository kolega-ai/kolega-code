"""The at-ceiling truncation warning for DeepSeek models.

DeepSeek reports a SERVER-side output cutoff as a clean finish (probed
2026-08-03, tests/agent/llm/test_deepseek_output_cap_live.py), so BaseAgent
flags any DeepSeek response that lands at/near the wire output cap without an
honest "max_tokens" stop — a probable silent truncation, surfaced as a
log_message warning that lands in trajectory jsonl. Hermetic: the LLM is
scripted with FakeLLM and the wire cap is set directly on the agent.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult

from .compaction_helpers import FakeLLM

CAP = 64000
SLACK = 1024


def _usage(output_tokens: int) -> dict:
    # get_output_tokens reads the provider stamped on the metadata, so the
    # fixture agent's (anthropic) primary config does not matter here.
    return {"provider": "deepseek", "prompt_tokens": 10, "completion_tokens": output_tokens}


def _message(stop_reason: str, output_tokens: int | None, text: str = "partial answer that was cut of") -> Message:
    return Message(
        role="assistant",
        content=[TextBlock(text=text)],
        stop_reason=stop_reason,
        usage_metadata=_usage(output_tokens) if output_tokens is not None else None,
    )


def _configure(agent, message_script: list[Message], cap: int | None = CAP) -> None:
    agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
    agent.tool_collection = MagicMock()
    agent.tool_collection.get_tool_list = MagicMock(return_value=[])
    agent.llm = FakeLLM(token_script=[100], message_script=message_script)
    agent.log_info = AsyncMock()
    agent.log_error = AsyncMock()
    agent.log_warning = AsyncMock()
    agent.deepseek_wire_output_cap = cap


def _warnings(agent) -> list[str]:
    return [call.args[0] for call in agent.log_warning.await_args_list]


@pytest.mark.asyncio
async def test_at_ceiling_end_turn_warns(base_agent) -> None:
    _configure(base_agent, [_message("end_turn", CAP - 100)])

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    warnings = _warnings(base_agent)
    assert len(warnings) == 1
    assert "truncated" in warnings[0]
    assert str(CAP) in warnings[0]
    assert "'end_turn'" in warnings[0]


@pytest.mark.asyncio
async def test_at_ceiling_tool_use_warns(base_agent) -> None:
    # A tool call assembled at the ceiling likely has truncated arguments, so
    # tool_use stops warn too.
    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    truncated = Message(
        role="assistant",
        content=[tool_call],
        stop_reason="tool_use",
        tool_calls=[tool_call],
        usage_metadata=_usage(CAP - 100),
    )
    _configure(base_agent, [truncated, _message("end_turn", 50)])
    base_agent.process_tool_calls = AsyncMock(
        return_value=[ToolResult(tool_use_id=tool_call.id, name=tool_call.name, content="ok", is_error=False)]
    )

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    warnings = _warnings(base_agent)
    assert len(warnings) == 1
    assert "truncated" in warnings[0]


@pytest.mark.asyncio
async def test_honest_max_tokens_stop_does_not_warn(base_agent) -> None:
    # An honest truncation needs no warning: the stop reason already tells the
    # truth and the loop/guard react to it.
    _configure(base_agent, [_message("max_tokens", CAP)])

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    assert _warnings(base_agent) == []


@pytest.mark.asyncio
async def test_below_threshold_does_not_warn(base_agent) -> None:
    _configure(base_agent, [_message("end_turn", CAP - SLACK - 1)])

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    assert _warnings(base_agent) == []


@pytest.mark.asyncio
async def test_non_deepseek_model_never_warns(base_agent) -> None:
    # deepseek_wire_output_cap is None for every non-DeepSeek model (set in
    # __init__ from is_deepseek_model), which disables the check entirely.
    _configure(base_agent, [_message("end_turn", CAP - 100)], cap=None)

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    assert _warnings(base_agent) == []


@pytest.mark.asyncio
async def test_missing_usage_does_not_warn(base_agent) -> None:
    _configure(base_agent, [_message("end_turn", None)])

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    assert _warnings(base_agent) == []
