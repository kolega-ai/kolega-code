import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.agent.baseagent import QueuedUserInput
from kolega_code.cli.session_store import SessionStore
from kolega_code.llm.models import ImageBlock, Message, TextBlock, ToolCall, ToolResult

from .compaction_helpers import FakeLLM


def _tool_call_message(tool_call: ToolCall) -> Message:
    return Message(
        role="assistant",
        content=[tool_call],
        stop_reason="tool_use",
        tool_calls=[tool_call],
    )


def _end_turn_message(text: str = "done") -> Message:
    return Message(role="assistant", content=[TextBlock(text=text)], stop_reason="end_turn")


def _tool_result(tool_call: ToolCall) -> ToolResult:
    return ToolResult(
        tool_use_id=tool_call.id,
        name=tool_call.name,
        content="ok",
        is_error=False,
    )


def _configure_agent(base_agent, message_script: list[Message]) -> None:
    base_agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
    base_agent.tool_collection = MagicMock()
    base_agent.tool_collection.get_tool_list = MagicMock(return_value=[])
    base_agent.llm = FakeLLM(token_script=[100], message_script=message_script)
    base_agent.log_info = AsyncMock()
    base_agent.log_error = AsyncMock()


@pytest.mark.asyncio
async def test_queued_input_injected_after_tool_batch(base_agent) -> None:
    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    _configure_agent(base_agent, [_tool_call_message(tool_call), _end_turn_message()])
    base_agent.process_tool_calls = AsyncMock(return_value=[_tool_result(tool_call)])
    provider = AsyncMock(side_effect=[[QueuedUserInput("also do Y")], []])
    base_agent.set_queued_input_provider(provider)

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    assert [message.role for message in base_agent.history] == [
        "user",
        "assistant",
        "user",
        "user",
        "assistant",
    ]
    assert isinstance(base_agent.history[1].content[0], ToolCall)
    assert all(isinstance(block, ToolResult) for block in base_agent.history[2].content)
    assert len(base_agent.history[3].content) == 1
    assert isinstance(base_agent.history[3].content[0], TextBlock)
    assert base_agent.history[3].content[0].text == "also do Y"
    assert base_agent.history[4].stop_reason == "end_turn"
    assert base_agent.conversation.is_valid_for_anthropic() is True
    provider.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_no_injection_when_should_stop_after_tools(base_agent, monkeypatch) -> None:
    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    _configure_agent(base_agent, [_tool_call_message(tool_call)])
    base_agent.process_tool_calls = AsyncMock(return_value=[_tool_result(tool_call)])
    provider = AsyncMock(return_value=[QueuedUserInput("also do Y")])
    base_agent.set_queued_input_provider(provider)
    monkeypatch.setattr(base_agent, "should_stop_after_tools", lambda: True)

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    provider.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_injection_when_hook_end_turn(base_agent) -> None:
    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    _configure_agent(base_agent, [_tool_call_message(tool_call)])

    def end_turn_after_tools(_tool_calls):
        base_agent._hook_end_turn = True
        return [_tool_result(tool_call)]

    base_agent.process_tool_calls = AsyncMock(side_effect=end_turn_after_tools)
    provider = AsyncMock(return_value=[QueuedUserInput("also do Y")])
    base_agent.set_queued_input_provider(provider)

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    provider.assert_not_awaited()
    assert base_agent._hook_end_turn is False


@pytest.mark.asyncio
async def test_no_injection_when_no_tool_calls(base_agent) -> None:
    _configure_agent(base_agent, [_end_turn_message()])
    provider = AsyncMock(return_value=[QueuedUserInput("also do Y")])
    base_agent.set_queued_input_provider(provider)

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    provider.assert_not_awaited()


@pytest.mark.asyncio
async def test_injection_journal_ordering_and_replay(base_agent, tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", {})
    base_agent.session_recorder = store.recorder(session.session_id)

    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    _configure_agent(base_agent, [_tool_call_message(tool_call), _end_turn_message()])
    base_agent.process_tool_calls = AsyncMock(return_value=[_tool_result(tool_call)])
    provider = AsyncMock(return_value=[QueuedUserInput("also do Y")])
    base_agent.set_queued_input_provider(provider)

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    event_types = [event.event_type for event in store.journal(session.session_id).read_events()]
    assert event_types[-6:] == [
        "turn.started",
        "assistant.message",
        "tool.results",
        "context.message",
        "assistant.message",
        "turn.completed",
    ]

    replayed = [Message.from_dict(item) for item in store.load(session.session_id).history]
    assert [message.role for message in replayed] == ["user", "assistant", "user", "user", "assistant"]
    assert all(isinstance(block, ToolResult) for block in replayed[2].content)
    assert len(replayed[3].content) == 1
    assert isinstance(replayed[3].content[0], TextBlock)
    assert replayed[3].content[0].text == "also do Y"


@pytest.mark.asyncio
async def test_injection_with_attachments(base_agent) -> None:
    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    _configure_agent(base_agent, [_tool_call_message(tool_call), _end_turn_message()])
    base_agent.process_tool_calls = AsyncMock(return_value=[_tool_result(tool_call)])
    image_data = base64.b64encode(b"fake-image-data").decode("utf-8")
    provider = AsyncMock(
        return_value=[
            QueuedUserInput(
                "look at this too",
                attachments=[
                    {
                        "type": "image",
                        "media_type": "image/png",
                        "data": image_data,
                        "filename": "queued.png",
                    }
                ],
            )
        ]
    )
    base_agent.set_queued_input_provider(provider)

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    injected = base_agent.history[3]
    assert len(injected.content) == 2
    assert isinstance(injected.content[0], TextBlock)
    assert injected.content[0].text == "look at this too"
    assert isinstance(injected.content[1], ImageBlock)
    assert injected.content[1].media_type == "image/png"
    assert injected.content[1].data == image_data
