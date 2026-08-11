"""continue_from_history_stream: the ordinary agent loop entered from restored
history with no user message — same recording, compaction, tool execution,
chunk format, and failure bookkeeping as a normal turn."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.agent.errors import MaxAgentIterationsExceeded
from kolega_code.cli.session_store import SessionStore
from kolega_code.hooks.events import HookEvent
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult

from .compaction_helpers import FakeLLM, build_agent, long_history
from .test_llm_call_origin import OriginProbeLLM


def _restored_pending_tool_history(agent, call_id="t1"):
    """Restore ``[user P, assistant ToolCall]`` via the public dump/restore API,
    then append the matching ToolResult the way an external caller would."""
    tool_call = ToolCall(id=call_id, name="probe", input={})
    history = [
        Message(role="user", content=[TextBlock(text="P")]).to_dict(),
        Message(role="assistant", content=[tool_call], tool_calls=[tool_call], stop_reason="tool_use").to_dict(),
    ]
    agent.restore_message_history(history)
    agent.restore_compaction_state(None)
    agent.append_user_message([ToolResult(tool_use_id=call_id, name="probe", content="probe-output", is_error=False)])


def _child_response(text="child answer"):
    return Message(role="assistant", content=[TextBlock(text=text)], stop_reason="end_turn")


@pytest.mark.asyncio
async def test_continuation_streams_child_response_without_new_user_message(tmp_path):
    agent, _ = build_agent(tmp_path, llm=FakeLLM(token_script=[100], final_message=_child_response()))
    _restored_pending_tool_history(agent)
    before = [message.to_dict() for message in agent.history]

    chunks = [chunk async for chunk in agent.continue_from_history_stream()]

    # History grew by exactly the child assistant message; nothing user-authored
    # was inserted.
    after = [message.to_dict() for message in agent.history]
    assert after[: len(before)] == before
    assert len(after) == len(before) + 1
    assert agent.history[-1].role == "assistant"
    assert agent.history[-1].get_text_content() == "child answer"
    # Chunk shape matches process_message_stream.
    final = [chunk for chunk in chunks if chunk.get("complete")]
    assert final and set(final[-1]) == {"type", "content", "complete", "uuid"}
    assert final[-1]["type"] == "response"
    # Tool pairing stays provider-valid.
    assert agent._is_history_valid_for_anthropic()


@pytest.mark.asyncio
async def test_continuation_journal_and_replay_roundtrip(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", {})
    agent, _ = build_agent(tmp_path, llm=FakeLLM(token_script=[100], final_message=_child_response()))
    agent.session_recorder = store.recorder(session.session_id)
    _restored_pending_tool_history(agent)

    async for _chunk in agent.continue_from_history_stream():
        pass

    events = store.journal(session.session_id).read_events()
    # No context.message: a continuation never injects a volatile-context turn.
    assert [event.event_type for event in events][-4:] == [
        "turn.started",
        "context.system",
        "assistant.message",
        "turn.completed",
    ]
    started = next(event for event in events if event.event_type == "turn.started")
    assert started.actor == "system"
    assert started.payload == {"continuation": True}
    assert "message" not in started.payload
    # The session stays loadable and replay contributes no message for the
    # continuation boundary: the replayed history is exactly the recorded
    # assistant message (restored history predates this journal).
    replayed = store.load(session.session_id).history
    assert [Message.from_dict(item).get_text_content() for item in replayed] == ["child answer"]


@pytest.mark.asyncio
async def test_continuation_auto_compaction_matches_normal_turn(tmp_path):
    agent, _ = build_agent(tmp_path, llm=FakeLLM(token_script=[900, 300]))
    agent.history = long_history(6)
    fired = []
    original = agent.fire_hook

    async def spy(name, payload, **kwargs):
        fired.append(name)
        return await original(name, payload, **kwargs)

    agent.fire_hook = spy

    async for _chunk in agent.continue_from_history_stream():
        pass

    assert HookEvent.PRE_COMPACT in fired
    assert HookEvent.POST_COMPACT in fired
    assert agent.conversation.summary is not None


@pytest.mark.asyncio
async def test_continuation_executes_tools_like_normal_turn(tmp_path):
    next_call = ToolCall(id="t2", name="probe", input={})
    script = [
        Message(role="assistant", content=[next_call], tool_calls=[next_call], stop_reason="tool_use"),
        _child_response("after tools"),
    ]
    agent, _ = build_agent(tmp_path, llm=FakeLLM(token_script=[100], message_script=script))
    _restored_pending_tool_history(agent)
    agent.process_tool_calls = AsyncMock(
        return_value=[ToolResult(tool_use_id="t2", name="probe", content="ok", is_error=False)]
    )

    async for _chunk in agent.continue_from_history_stream():
        pass

    agent.process_tool_calls.assert_awaited_once()
    assert agent.history[-1].get_text_content() == "after tools"
    assert agent._is_history_valid_for_anthropic()


@pytest.mark.asyncio
async def test_continuation_empty_history_raises_before_any_event(tmp_path):
    agent, cm = build_agent(tmp_path, llm=FakeLLM(token_script=[100]))
    recorder = MagicMock()
    recorder.current_turn_id = None
    agent.session_recorder = recorder

    with pytest.raises(ValueError, match="non-empty conversation"):
        async for _chunk in agent.continue_from_history_stream():
            pass

    recorder.start_continuation_turn.assert_not_called()
    recorder.finish_turn.assert_not_called()
    assert not any(
        getattr(call.args[0] if call.args else call.kwargs.get("event"), "event_type", None) == "turn_started"
        for call in cm.broadcast_event.call_args_list
    )


@pytest.mark.asyncio
async def test_continuation_cancellation_bookkeeping(tmp_path):
    class BlockingLLM(FakeLLM):
        async def _stream(self, *args, **kwargs):
            await asyncio.Event().wait()

    agent, _ = build_agent(tmp_path, llm=BlockingLLM(token_script=[100]))
    _restored_pending_tool_history(agent)

    class Recorder:
        current_turn_id = None
        terminal_status = None

        def start_continuation_turn(self):
            self.current_turn_id = "turn"

        def record_system_context(self, text):
            return False

        def record_tool_definitions(self, tools):
            return False

        def finish_turn(self, status, *, error=None):
            self.terminal_status = status
            self.current_turn_id = None

    agent.session_recorder = Recorder()

    async def consume():
        async for _chunk in agent.continue_from_history_stream():
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.session_recorder.terminal_status == "cancelled"


@pytest.mark.asyncio
async def test_continuation_provider_failure_records_failed_turn(tmp_path):
    from kolega_code.llm.exceptions import LLMAuthenticationError

    failing = FakeLLM(token_script=[100])
    failing.stream = AsyncMock(side_effect=LLMAuthenticationError("bad key"))
    agent, _ = build_agent(tmp_path, llm=failing)
    _restored_pending_tool_history(agent)

    class Recorder:
        current_turn_id = None
        terminal_status = None

        def start_continuation_turn(self):
            self.current_turn_id = "turn"

        def record_system_context(self, text):
            return False

        def record_tool_definitions(self, tools):
            return False

        def finish_turn(self, status, *, error=None):
            self.terminal_status = status
            self.current_turn_id = None

    agent.session_recorder = Recorder()

    with pytest.raises(LLMAuthenticationError):
        async for _chunk in agent.continue_from_history_stream():
            pass

    assert agent.session_recorder.terminal_status == "failed"


@pytest.mark.asyncio
async def test_continuation_max_iterations_parity(tmp_path):
    tool_call = ToolCall(id="loop-1", name="probe", input={})
    looping = Message(role="assistant", content=[tool_call], tool_calls=[tool_call], stop_reason="tool_use")
    agent, _ = build_agent(tmp_path, llm=FakeLLM(token_script=[100], final_message=looping))
    agent.max_iterations = 2
    _restored_pending_tool_history(agent)
    agent.process_tool_calls = AsyncMock(
        return_value=[ToolResult(tool_use_id="loop-1", name="probe", content="ok", is_error=False)]
    )

    with pytest.raises(MaxAgentIterationsExceeded, match="max_iterations=2"):
        async for _chunk in agent.continue_from_history_stream():
            pass


@pytest.mark.asyncio
async def test_continuation_tool_error_produces_error_results(tmp_path):
    next_call = ToolCall(id="t3", name="probe", input={})
    script = [
        Message(role="assistant", content=[next_call], tool_calls=[next_call], stop_reason="tool_use"),
        _child_response("recovered"),
    ]
    agent, _ = build_agent(tmp_path, llm=FakeLLM(token_script=[100], message_script=script))
    _restored_pending_tool_history(agent)
    agent.process_tool_calls = AsyncMock(side_effect=RuntimeError("tool backend down"))

    async for _chunk in agent.continue_from_history_stream():
        pass

    # The loop synthesized per-call error results and kept going, same as a
    # normal turn.
    error_results = [
        block
        for message in agent.history
        for block in message.content
        if isinstance(block, ToolResult) and block.tool_use_id == "t3"
    ]
    assert error_results and error_results[0].is_error
    assert agent.history[-1].get_text_content() == "recovered"


@pytest.mark.asyncio
async def test_continuation_stop_sequence_terminal(tmp_path):
    final = Message(role="assistant", content=[TextBlock(text="cut")], stop_reason="stop_sequence")
    agent, _ = build_agent(tmp_path, llm=FakeLLM(token_script=[100], final_message=final))
    _restored_pending_tool_history(agent)

    chunks = [chunk async for chunk in agent.continue_from_history_stream()]

    assert chunks[-1]["complete"] is True
    assert agent.history[-1].stop_reason == "stop_sequence"


@pytest.mark.asyncio
async def test_continuation_delivers_queued_inputs(tmp_path):
    next_call = ToolCall(id="t4", name="probe", input={})
    script = [
        Message(role="assistant", content=[next_call], tool_calls=[next_call], stop_reason="tool_use"),
        _child_response("done"),
    ]
    agent, _ = build_agent(tmp_path, llm=FakeLLM(token_script=[100], message_script=script))
    _restored_pending_tool_history(agent)
    agent.process_tool_calls = AsyncMock(
        return_value=[ToolResult(tool_use_id="t4", name="probe", content="ok", is_error=False)]
    )
    from kolega_code.agent.baseagent import QueuedUserInput

    queued = [QueuedUserInput(text="queued user note", attachments=None)]

    async def provider():
        return [queued.pop()] if queued else []

    agent.set_queued_input_provider(provider)

    async for _chunk in agent.continue_from_history_stream():
        pass

    texts = [message.get_text_content() for message in agent.history]
    assert any("queued user note" in text for text in texts)


@pytest.mark.asyncio
async def test_continuation_origin_matches_main_loop(tmp_path):
    llm = OriginProbeLLM()
    agent, _ = build_agent(tmp_path, llm=llm)
    _restored_pending_tool_history(agent)

    async for _chunk in agent.continue_from_history_stream():
        pass

    assert {phase for phase, _ in llm.probes} == {"stream", "aenter", "anext", "final"}
    for phase, origin in llm.probes:
        assert origin is not None and origin.kind == "primary", phase
