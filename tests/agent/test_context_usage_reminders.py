"""Live context-window utilization reminders emitted after tool-call rounds."""

from __future__ import annotations

import asyncio
import base64
import copy
import io
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kolega_code.agent.conversation import Conversation
from kolega_code.agent.errors import MaxAgentIterationsExceeded
from kolega_code.agent.prompts import CONTEXT_USAGE_REMINDER_SOURCE, build_context_usage_reminder
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.tui.transcript import _is_standalone_system_reminder_history_item
from kolega_code.llm.exceptions import LLMRateLimitError
from kolega_code.llm.models import ImageBlock, Message, MessageHistory, TextBlock, ToolCall, ToolDefinition, ToolResult
from kolega_code.llm.providers.responses_common import to_responses_input

from .compaction_helpers import FakeLLM, build_agent, long_history


class RecordingToolLLM(FakeLLM):
    """FakeLLM that captures serialized Anthropic payloads on each stream call."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.payloads: list[dict[str, Any]] = []
        self.raw_histories: list[MessageHistory] = []

    async def _stream(self, *args: Any, **kwargs: Any):
        messages = kwargs.get("messages")
        system = kwargs.get("system")
        tools = kwargs.get("tools")
        if messages is not None:
            self.raw_histories.append(MessageHistory(list(messages)))
        self.payloads.append(
            {
                "messages": copy.deepcopy(messages.to_anthropic()) if messages is not None else [],
                "system": copy.deepcopy([block.to_anthropic() for block in system.content]) if system else [],
                "tools": copy.deepcopy([tool.to_anthropic() for tool in tools]) if tools else [],
            }
        )
        return await super()._stream(*args, **kwargs)


def _strip_cache_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_cache_control(item) for item in value]
    return value


def _count_cache_breakpoints(value: Any) -> int:
    if isinstance(value, dict):
        own = 1 if "cache_control" in value else 0
        return own + sum(_count_cache_breakpoints(v) for v in value.values())
    if isinstance(value, list):
        return sum(_count_cache_breakpoints(item) for item in value)
    return 0


def _context_usage_reminders(history: list[Message]) -> list[str]:
    prefix = f'<system-reminder source="{CONTEXT_USAGE_REMINDER_SOURCE}">'
    return [
        msg.get_text_content() for msg in history if msg.role == "user" and msg.get_text_content().startswith(prefix)
    ]


def _tool_msg(*calls: ToolCall, billed_input: int | None = None) -> Message:
    usage = {"input_tokens": billed_input} if billed_input is not None else {}
    return Message(
        role="assistant",
        content=list(calls),
        stop_reason="tool_use",
        tool_calls=list(calls),
        usage_metadata=usage,
    )


def _end_msg(text: str = "done") -> Message:
    return Message(role="assistant", content=[TextBlock(text=text)], stop_reason="end_turn")


def _result(call: ToolCall, content: Any = "ok") -> ToolResult:
    return ToolResult(tool_use_id=call.id, name=call.name, content=content, is_error=False)


def _oversized_png_base64() -> str:
    """Create a PNG exceeding Anthropic's max dimension so adaptation resizes/copies it."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (2400, 100), color=(40, 120, 200)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def _run_turn(agent: Any, prompt: str = "work") -> None:
    async for _chunk in agent.process_message_stream(prompt):
        pass


class TestBuildContextUsageReminder:
    def test_formats_issue_example_exactly(self) -> None:
        text = build_context_usage_reminder(
            input_tokens=24_100,
            max_input_tokens=57_344,
            compression_threshold=0.95,
        )
        assert text == (
            '<system-reminder source="context_usage">\n'
            "Context usage: 24,100 / 57,344 input tokens (42%; automatic compaction at 95%).\n"
            "</system-reminder>"
        )

    def test_handles_boundaries_and_fractional_thresholds(self) -> None:
        zero = build_context_usage_reminder(input_tokens=0, max_input_tokens=50_000, compression_threshold=1.0)
        assert zero is not None
        assert "0 / 50,000 input tokens (0%; automatic compaction at 100%)." in zero

        over = build_context_usage_reminder(input_tokens=58_500, max_input_tokens=57_344, compression_threshold=0.855)
        assert over is not None
        assert "58,500 / 57,344 input tokens (102%; automatic compaction at 85.5%)." in over

        clamped = build_context_usage_reminder(input_tokens=-10, max_input_tokens=1_000, compression_threshold=0.8)
        assert clamped is not None
        assert "0 / 1,000 input tokens (0%; automatic compaction at 80%)." in clamped

        assert build_context_usage_reminder(input_tokens=100, max_input_tokens=0, compression_threshold=0.95) is None
        assert build_context_usage_reminder(input_tokens=100, max_input_tokens=-5, compression_threshold=0.95) is None

    def test_satisfies_tui_standalone_reminder_filter(self) -> None:
        text = build_context_usage_reminder(input_tokens=10_000, max_input_tokens=100_000, compression_threshold=0.95)
        assert text is not None
        item = Message(role="user", content=[TextBlock(text=text)]).to_dict()
        assert _is_standalone_system_reminder_history_item(item) is True


class TestAgentLoopCadenceAndFreshness:
    @pytest.mark.asyncio
    async def test_no_reminder_on_single_reply_turn(self, tmp_path) -> None:
        llm = RecordingToolLLM(token_script=[1_000], message_script=[_end_msg("hello")])
        agent, _ = build_agent(tmp_path, llm=llm, model_context_length=10_000)

        await _run_turn(agent, "hi")

        assert _context_usage_reminders(list(agent.history)) == []
        assert len(llm.payloads) == 1

    @pytest.mark.asyncio
    async def test_no_reminder_on_continue_from_history_without_new_tools(self, tmp_path) -> None:
        llm = RecordingToolLLM(token_script=[1_000], message_script=[_end_msg("resumed")])
        agent, _ = build_agent(tmp_path, llm=llm, model_context_length=10_000)
        agent.history = MessageHistory([Message(role="user", content=[TextBlock(text="prior")])])

        async for _chunk in agent.continue_from_history_stream():
            pass

        assert _context_usage_reminders(list(agent.history)) == []

    @pytest.mark.asyncio
    async def test_emits_once_per_tool_round_including_parallel_batch_and_residual(self, tmp_path) -> None:
        batch_1 = [
            ToolCall(id="c1", name="read_file", input={"path": "a.py"}),
            ToolCall(id="c2", name="read_file", input={"path": "b.py"}),
        ]
        batch_2 = [ToolCall(id="c3", name="exec_command", input={"command": "pytest"})]
        # Round 1 raw count=10_000; provider bills 12_000 -> residual=+2_000.
        # Round 2 raw count=22_100 + residual 2_000 = 24_100; provider bills 22_100 -> residual=0.
        # Round 3 raw count=40_000 + residual 0 = 40_000.
        llm = RecordingToolLLM(
            token_script=[10_000, 22_100, 40_000],
            message_script=[
                _tool_msg(*batch_1, billed_input=12_000),
                _tool_msg(*batch_2, billed_input=22_100),
                _end_msg("all done"),
            ],
        )
        agent, _ = build_agent(tmp_path, llm=llm, model_context_length=57_344)
        agent.history_compression_threshold = 0.95
        agent.compressor.threshold = 0.95
        agent.process_tool_calls = AsyncMock(
            side_effect=[
                [_result(batch_1[0], "a"), _result(batch_1[1], "b")],
                [_result(batch_2[0], "passed")],
            ]
        )

        await _run_turn(agent, "run work")

        reminders = _context_usage_reminders(list(agent.history))
        assert len(reminders) == 2
        assert "Context usage: 24,100 / 57,344 input tokens (42%; automatic compaction at 95%)." in reminders[0]
        assert "Context usage: 40,000 / 57,344 input tokens (70%; automatic compaction at 95%)." in reminders[1]
        # First model call had no reminder; 2nd saw round 1's; 3rd saw both.
        assert len(_context_usage_reminders(list(llm.raw_histories[0]))) == 0
        assert len(_context_usage_reminders(list(llm.raw_histories[1]))) == 1
        assert len(_context_usage_reminders(list(llm.raw_histories[2]))) == 2
        # No extra count_tokens calls beyond the 3 iterations.
        assert llm.count_tokens.await_count == 3

    @pytest.mark.asyncio
    async def test_reports_post_compaction_count_when_tool_output_triggers_compaction(self, tmp_path) -> None:
        call = ToolCall(id="c1", name="read_file", input={"path": "big.log"})
        # Iteration 1: 500 (under 80% of 1000).
        # Iteration 2: 950 (over 80%) -> auto-compacts -> compress_history recounts (250) ->
        # _auto_compact_once recounts (250) -> reminder must report 250 (25%), not 950 (95%).
        llm = RecordingToolLLM(
            token_script=[500, 950, 250, 250],
            message_script=[_tool_msg(call), _end_msg("finished")],
            summary_text="Condensed earlier work.",
        )
        agent, _ = build_agent(tmp_path, llm=llm, model_context_length=1_000)
        agent.history = long_history(4)
        agent.process_tool_calls = AsyncMock(return_value=[_result(call, "huge output")])

        await _run_turn(agent, "inspect log")

        assert agent.conversation.summary is not None
        reminders = _context_usage_reminders(list(agent.history))
        assert len(reminders) == 1
        assert "Context usage: 250 / 1,000 input tokens (25%; automatic compaction at 80%)." in reminders[0]
        assert "950" not in reminders[0]


class TestPrefixStabilityAndCacheBreakpoints:
    @pytest.mark.asyncio
    async def test_prefix_stays_append_only_and_within_four_breakpoints(self, tmp_path) -> None:
        c1 = ToolCall(id="c1", name="read_file", input={"path": "a.py"})
        c2 = ToolCall(id="c2", name="read_file", input={"path": "b.py"})
        llm = RecordingToolLLM(
            token_script=[1_000, 2_000, 3_000],
            message_script=[_tool_msg(c1), _tool_msg(c2), _end_msg("done")],
        )
        agent, _ = build_agent(tmp_path, llm=llm, model_context_length=100_000)
        agent.system_prompt = agent.system_prompt_message("static system prompt")
        tool_def = ToolDefinition(name="read_file", description="Read a file", parameters=[], cache_checkpoint=True)
        assert agent.tool_collection is not None
        agent.tool_collection.get_tool_list = lambda: [tool_def]
        agent.process_tool_calls = AsyncMock(side_effect=[[_result(c1, "alpha")], [_result(c2, "beta")]])

        await _run_turn(agent, "multi-step")

        assert len(llm.payloads) == 3
        assert llm.payloads[0]["system"] == llm.payloads[1]["system"] == llm.payloads[2]["system"]
        for earlier, later in zip(llm.payloads, llm.payloads[1:]):
            earlier_msgs = _strip_cache_control(earlier["messages"])
            later_msgs = _strip_cache_control(later["messages"])
            assert later_msgs[: len(earlier_msgs)] == earlier_msgs
        for payload in llm.payloads:
            assert _count_cache_breakpoints(payload) <= 4
            assert _count_cache_breakpoints(payload["messages"]) <= 2
        # On round 3, the two rolling breakpoints sit on round 2's reminder and round 3's reminder.
        marked_indices = [
            idx for idx, msg in enumerate(llm.payloads[2]["messages"]) if _count_cache_breakpoints(msg) > 0
        ]
        assert len(marked_indices) == 2
        assert marked_indices[1] == len(llm.payloads[2]["messages"]) - 1
        assert marked_indices[1] - marked_indices[0] >= 3

    @pytest.mark.asyncio
    async def test_resized_image_tool_result_keeps_breakpoints_within_four(self, tmp_path) -> None:
        c1 = ToolCall(id="img1", name="read_image", input={"path": "shot1.png"})
        c2 = ToolCall(id="img2", name="read_image", input={"path": "shot2.png"})
        img_block = ImageBlock(image_type="base64", media_type="image/png", data=_oversized_png_base64())
        llm = RecordingToolLLM(
            token_script=[1_000, 5_000, 9_000],
            message_script=[_tool_msg(c1), _tool_msg(c2), _end_msg("saw both")],
        )
        agent, _ = build_agent(tmp_path, llm=llm, model_context_length=100_000)
        agent.supports_vision = True
        agent.system_prompt = agent.system_prompt_message("static system prompt")
        tool_def = ToolDefinition(name="read_image", description="Read image", parameters=[], cache_checkpoint=True)
        assert agent.tool_collection is not None
        agent.tool_collection.get_tool_list = lambda: [tool_def]
        agent.process_tool_calls = AsyncMock(
            side_effect=[
                [_result(c1, [img_block])],
                [_result(c2, [img_block])],
            ]
        )

        await _run_turn(agent, "inspect screenshots")

        for payload in llm.payloads:
            assert _count_cache_breakpoints(payload) <= 4
            assert _count_cache_breakpoints(payload["messages"]) <= 2

    def test_advance_latest_cache_checkpoint_clears_copied_block_after_compaction(self) -> None:
        old_msg = Message(role="user", content=[TextBlock(text="old")])
        tail_msg = Message(role="user", content=[TextBlock(text="tail")])
        conversation = Conversation([old_msg, tail_msg])
        conversation.mark_cache_checkpoint()
        # Simulate compaction folding away old_msg while tail_msg remains.
        conversation.apply_compaction("summary", split_point=1)
        conversation.mark_cache_checkpoint()
        # Simulate provider adaptation copying tail_msg's marked block in request_history.
        copied_tail_block = TextBlock(text="tail copy", cache_checkpoint=True)
        copied_tail = Message(role="user", content=[copied_tail_block])
        assert conversation.summary is not None
        request_history = MessageHistory([conversation.summary, copied_tail])
        reminder_block = TextBlock(text="reminder")
        reminder = Message(role="user", content=[reminder_block])
        conversation.history.append(reminder)
        request_history.append(reminder)

        conversation.advance_latest_cache_checkpoint(request_history)

        marked = [
            block
            for msg in request_history
            if isinstance(msg.content, list)
            for block in msg.content
            if getattr(block, "cache_checkpoint", False)
        ]
        assert marked == [reminder_block]
        assert copied_tail_block.cache_checkpoint is False


class TestEdgeCasesAndGating:
    @pytest.mark.asyncio
    async def test_transient_retry_does_not_duplicate_reminder(self, tmp_path) -> None:
        call = ToolCall(id="c1", name="read_file", input={})
        base_llm = RecordingToolLLM(
            token_script=[1_000, 2_000, 2_000],
            message_script=[_tool_msg(call), _end_msg("recovered")],
        )
        attempts = 0
        original_stream = base_llm._stream

        async def flaky_stream(*args: Any, **kwargs: Any):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise LLMRateLimitError("429 retry me")
            return await original_stream(*args, **kwargs)

        base_llm.stream = AsyncMock(side_effect=flaky_stream)
        agent, _ = build_agent(tmp_path, llm=base_llm, model_context_length=10_000)
        agent.handle_llm_error = AsyncMock()
        agent.process_tool_calls = AsyncMock(return_value=[_result(call)])

        await _run_turn(agent, "retry test")

        assert attempts == 3
        assert len(_context_usage_reminders(list(agent.history))) == 1

    @pytest.mark.asyncio
    async def test_no_dangling_reminder_on_should_stop_after_tools(self, tmp_path, monkeypatch) -> None:
        call = ToolCall(id="c1", name="write_plan", input={})
        llm = RecordingToolLLM(token_script=[1_000], message_script=[_tool_msg(call)])
        agent, _ = build_agent(tmp_path, llm=llm, model_context_length=10_000)
        agent.process_tool_calls = AsyncMock(return_value=[_result(call)])
        monkeypatch.setattr(agent, "should_stop_after_tools", lambda: True)

        await _run_turn(agent, "plan")

        assert _context_usage_reminders(list(agent.history)) == []

    @pytest.mark.asyncio
    async def test_no_dangling_reminder_on_hook_end_turn(self, tmp_path) -> None:
        call = ToolCall(id="c1", name="exec_command", input={})
        llm = RecordingToolLLM(token_script=[1_000], message_script=[_tool_msg(call)])
        agent, _ = build_agent(tmp_path, llm=llm, model_context_length=10_000)

        async def stop_via_hook(_calls: list[ToolCall]) -> list[ToolResult]:
            agent._hook_end_turn = True
            return [_result(call)]

        agent.process_tool_calls = AsyncMock(side_effect=stop_via_hook)

        await _run_turn(agent, "hook stop")

        assert _context_usage_reminders(list(agent.history)) == []

    @pytest.mark.asyncio
    async def test_no_dangling_reminder_on_max_iterations(self, tmp_path) -> None:
        call = ToolCall(id="c1", name="read_file", input={})
        llm = RecordingToolLLM(token_script=[1_000], message_script=[_tool_msg(call)])
        agent, _ = build_agent(tmp_path, llm=llm, model_context_length=10_000)
        agent.max_iterations = 1
        agent.process_tool_calls = AsyncMock(return_value=[_result(call)])

        with pytest.raises(MaxAgentIterationsExceeded):
            await _run_turn(agent, "loop cap")

        assert _context_usage_reminders(list(agent.history)) == []

    @pytest.mark.asyncio
    async def test_no_dangling_reminder_on_swallowed_cancel(self, tmp_path) -> None:
        call = ToolCall(id="c1", name="exec_command", input={})
        started = asyncio.Event()
        llm = RecordingToolLLM(token_script=[1_000], message_script=[_tool_msg(call)])
        agent, _ = build_agent(tmp_path, llm=llm, model_context_length=10_000)

        async def swallowing_tool(_calls: list[ToolCall]) -> list[ToolResult]:
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                return [_result(call, "Status: cancelled")]
            raise AssertionError("expected cancellation")

        agent.process_tool_calls = AsyncMock(side_effect=swallowing_tool)

        task = asyncio.create_task(_run_turn(agent, "cancel me"))
        await asyncio.wait_for(started.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)

        assert _context_usage_reminders(list(agent.history)) == []

    @pytest.mark.asyncio
    async def test_strict_context_budget_skips_when_insufficient_headroom(self, tmp_path) -> None:
        call = ToolCall(id="c1", name="read_file", input={})
        # Strict window=4000, output=500 -> max_input=3500.
        # Round 2 counts 3450 (only 50 tokens below cap, less than len(reminder_text)): must skip.
        llm = RecordingToolLLM(token_script=[1_000, 3_450], message_script=[_tool_msg(call), _end_msg("tight fit")])
        agent, _ = build_agent(tmp_path, llm=llm, strict_budget=(4_000, 500))
        agent.history_compression_threshold = 1.0
        agent.compressor.threshold = 1.0
        agent.process_tool_calls = AsyncMock(return_value=[_result(call)])

        await _run_turn(agent, "tight budget")

        assert _context_usage_reminders(list(agent.history)) == []

    @pytest.mark.asyncio
    async def test_strict_context_budget_emits_when_headroom_available(self, tmp_path) -> None:
        call = ToolCall(id="c1", name="read_file", input={})
        llm = RecordingToolLLM(token_script=[1_000, 2_000], message_script=[_tool_msg(call), _end_msg("plenty")])
        agent, _ = build_agent(tmp_path, llm=llm, strict_budget=(4_000, 500))
        agent.process_tool_calls = AsyncMock(return_value=[_result(call)])

        await _run_turn(agent, "budgeted")

        reminders = _context_usage_reminders(list(agent.history))
        assert len(reminders) == 1
        assert "2,000 / 3,500 input tokens (57%; automatic compaction at 80%)." in reminders[0]

    @pytest.mark.asyncio
    async def test_disabled_via_config_and_inherited_by_sub_agent(self, tmp_path) -> None:
        call = ToolCall(id="c1", name="read_file", input={})
        llm = RecordingToolLLM(token_script=[1_000, 2_000], message_script=[_tool_msg(call), _end_msg("quiet")])
        agent, _ = build_agent(tmp_path, llm=llm, model_context_length=10_000)
        agent.context_usage_reminders_enabled = False
        agent.process_tool_calls = AsyncMock(return_value=[_result(call)])

        await _run_turn(agent, "no reminders")
        assert _context_usage_reminders(list(agent.history)) == []

        sub_llm = RecordingToolLLM(token_script=[500, 1_500], message_script=[_tool_msg(call), _end_msg("sub done")])
        sub, _ = build_agent(tmp_path, llm=sub_llm, sub_agent=True, model_context_length=10_000)
        sub.process_tool_calls = AsyncMock(return_value=[_result(call)])

        await _run_turn(sub, "sub work")
        sub_reminders = _context_usage_reminders(list(sub.history))
        assert len(sub_reminders) == 1
        assert "1,500 / 10,000 input tokens (15%; automatic compaction at 80%)." in sub_reminders[0]


class TestJournalReplayAndProviderSerialization:
    @pytest.mark.asyncio
    async def test_journal_replay_order_and_provider_conversions(self, tmp_path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        store = SessionStore(tmp_path / "state")
        session = store.create(project, "code", {})

        call = ToolCall(id="c1", name="read_file", input={"path": "README.md"})
        llm = RecordingToolLLM(token_script=[1_000, 4_200], message_script=[_tool_msg(call), _end_msg("summarized")])
        agent, _ = build_agent(tmp_path, llm=llm, model_context_length=10_000)
        agent.session_recorder = store.recorder(session.session_id)
        agent.process_tool_calls = AsyncMock(return_value=[_result(call, "readme text")])

        await _run_turn(agent, "read readme")

        event_types = [event.event_type for event in store.journal(session.session_id).read_events()]
        assert event_types[-8:] == [
            "turn.started",
            "context.system",
            "context.message",
            "assistant.message",
            "tool.results",
            "context.message",
            "assistant.message",
            "turn.completed",
        ]

        replayed_dicts = store.load(session.session_id).history
        reminder_dict = replayed_dicts[-2]
        assert _is_standalone_system_reminder_history_item(reminder_dict) is True
        replayed = MessageHistory([Message.from_dict(item) for item in replayed_dicts])

        anthropic_msgs = replayed.to_anthropic()
        assert anthropic_msgs[-2]["role"] == "user"
        assert "4,200 / 10,000 input tokens" in anthropic_msgs[-2]["content"][0]["text"]

        openai_msgs = replayed.to_openai()
        roles = [msg["role"] for msg in openai_msgs]
        tool_idx = roles.index("tool")
        assert openai_msgs[tool_idx + 1]["role"] == "user"
        assert "4,200 / 10,000 input tokens" in str(openai_msgs[tool_idx + 1]["content"])

        responses_items = to_responses_input(replayed)
        out_idx = next(i for i, item in enumerate(responses_items) if item.get("type") == "function_call_output")
        assert responses_items[out_idx + 1]["role"] == "user"
        assert "4,200 / 10,000 input tokens" in str(responses_items[out_idx + 1]["content"])

        google_contents = replayed.to_google()
        assert google_contents[-2].role == "user"
        assert "4,200 / 10,000 input tokens" in (google_contents[-2].parts[0].text or "")
