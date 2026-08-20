"""Turn cancellation must survive tools that swallow ``CancelledError``.

Terminal tools (``exec_command``/``write_stdin``) convert a turn cancellation
into a structured ``status="cancelled"`` result — killing the process and
keeping the ToolCall/ToolResult pair valid in history — which consumes the
``CancelledError`` instead of propagating it. The agent loop must still honor
the cancellation afterwards: journal the tool result, then end the turn as
cancelled instead of issuing another model request (the pre-fix bug: the first
Ctrl-C cancelled only the command and the agent kept working).
"""

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from kolega_code.agent.tools import ToolCollection
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult

from .compaction_helpers import FakeLLM


def _tool_call_message(name: str, tool_input: dict) -> Message:
    tool_call = ToolCall(id="tool_1", name=name, input=tool_input)
    return Message(role="assistant", content=[tool_call], stop_reason="tool_use", tool_calls=[tool_call])


def _tool_results(history: list[Message]) -> list[ToolResult]:
    return [block for message in history for block in (message.content or []) if isinstance(block, ToolResult)]


class _HangingStream:
    """A provider stream that never produces events (cancel lands mid-stream)."""

    def __init__(self, entered: asyncio.Event) -> None:
        self._entered = entered

    async def __aenter__(self) -> "_HangingStream":
        self._entered.set()
        return self

    async def __aexit__(self, *exc: object) -> bool:
        _ = exc
        return False

    def __aiter__(self) -> "_HangingStream":
        return self

    async def __anext__(self) -> None:
        await asyncio.sleep(3600)
        raise StopAsyncIteration

    async def get_final_message(self) -> Message:
        raise AssertionError("hanging stream never completed")


@pytest.mark.asyncio
class TestTurnCancellation:
    async def test_cancel_during_exec_command_ends_turn_after_journaling_result(
        self, base_agent, agent_config, mock_connection_manager, tmp_path
    ):
        """End-to-end: Ctrl-C during a real exec_command must cancel the turn.

        Drives the real TerminalTool and a real PTY session. Without the loop's
        pending-cancellation check, the tool's structured "cancelled" result
        swallows the CancelledError and the loop issues a second model request.
        """
        mock_caller = Mock()
        mock_caller.agent_name = "test_agent"
        mock_caller.scratchpad_dir = None
        collection = ToolCollection(
            tmp_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_caller
        )
        base_agent.tool_collection = collection
        base_agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
        base_agent.llm = FakeLLM(
            message_script=[
                _tool_call_message("exec_command", {"command": "sleep 30", "yield_time_ms": 30000}),
                Message(
                    role="assistant", content=[TextBlock(text="kept working after cancel")], stop_reason="end_turn"
                ),
            ]
        )
        base_agent.log_info = AsyncMock()
        base_agent.log_error = AsyncMock()
        base_agent.log_warning = AsyncMock()

        manager = collection.terminal_tool.terminal_manager

        async def run_turn() -> list[dict]:
            return [chunk async for chunk in base_agent.process_message_stream("run the sleep")]

        task = asyncio.create_task(run_turn())
        deadline = time.monotonic() + 10
        while not manager.sessions and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert manager.sessions, "terminal session never started"
        session = next(iter(manager.sessions.values()))
        assert session.running, "command should still be in flight"

        task.cancel()  # what Ctrl-C does through Textual's Worker.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # The turn ended on the first cancel: no second model request.
        assert base_agent.llm.stream.await_count == 1
        # The recovery result kept the ToolCall/ToolResult pair in history.
        results = _tool_results(base_agent.history)
        assert len(results) == 1
        assert "Status: cancelled" in str(results[0].content)
        assert results[0].is_error is False

    async def test_cancel_swallowed_before_natural_loop_exit_still_ends_as_cancelled(self, base_agent):
        """A swallowed cancel must not let the turn record itself as completed.

        Covers the natural-exit path: the tool batch finishes after the swallow
        and ``should_stop_after_tools`` breaks the loop before another request.
        """
        base_agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
        base_agent.tool_collection = MagicMock()
        base_agent.tool_collection.get_tool_list = MagicMock(return_value=[])
        base_agent.llm = FakeLLM(message_script=[_tool_call_message("read", {"file_path": "/tmp/x"})])
        base_agent.log_info = AsyncMock()
        base_agent.log_error = AsyncMock()
        base_agent.log_warning = AsyncMock()

        started = asyncio.Event()

        async def swallowing_tool(tool_use_block: ToolCall) -> ToolResult:
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                # Mimic TerminalTool: return a structured cancelled result
                # instead of propagating.
                return ToolResult(
                    tool_use_id=tool_use_block.id,
                    name=tool_use_block.name,
                    content="Status: cancelled",
                    is_error=False,
                )
            raise AssertionError("swallowing tool should have been cancelled, not completed")

        base_agent.execute_single_tool = swallowing_tool  # type: ignore[method-assign]
        base_agent.should_stop_after_tools = lambda: True  # type: ignore[method-assign]

        async def run_turn() -> list[dict]:
            return [chunk async for chunk in base_agent.process_message_stream("go")]

        task = asyncio.create_task(run_turn())
        await asyncio.wait_for(started.wait(), timeout=10)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert base_agent.llm.stream.await_count == 1
        results = _tool_results(base_agent.history)
        assert len(results) == 1
        assert "Status: cancelled" in str(results[0].content)

    async def test_cancel_during_llm_stream_propagates(self, base_agent):
        """Control: cancellation on the streaming path is unchanged by the fix."""
        base_agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
        base_agent.tool_collection = MagicMock()
        base_agent.tool_collection.get_tool_list = MagicMock(return_value=[])
        entered = asyncio.Event()
        base_agent.llm = FakeLLM()
        base_agent.llm.stream = AsyncMock(return_value=_HangingStream(entered))
        base_agent.log_info = AsyncMock()
        base_agent.log_error = AsyncMock()

        async def run_turn() -> list[dict]:
            return [chunk async for chunk in base_agent.process_message_stream("go")]

        task = asyncio.create_task(run_turn())
        await asyncio.wait_for(entered.wait(), timeout=10)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert base_agent.llm.stream.await_count == 1
