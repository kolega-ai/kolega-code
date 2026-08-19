"""Session context is the first hidden reminder in ordinary conversation history."""

from unittest.mock import MagicMock

import pytest

from kolega_code.cli.session_store import SessionStore
from kolega_code.llm.models import ImageBlock, Message, MessageHistory, TextBlock, ToolCall
from kolega_code.llm.providers.responses_common import to_responses_input
from kolega_code.llm.providers.tinker import _openai_messages as to_tinker_openai_messages

from .compaction_helpers import FakeLLM, build_agent, long_history


def _session_message(text: str = "runtime session context") -> Message:
    return Message(
        role="user",
        content=[TextBlock(text=f'<system-reminder source="session">\n{text}\n</system-reminder>')],
    )


@pytest.mark.asyncio
async def test_session_context_is_built_once_and_inserted_at_history_start(base_agent):
    build = MagicMock(return_value=_session_message())
    base_agent.build_session_context_message = build
    base_agent.history = MessageHistory(
        [
            Message(role="assistant", content=[ToolCall(id="call-1", name="read", input={})]),
            Message(role="user", content=[TextBlock(text="actual user message")]),
        ]
    )

    first = await base_agent._history_for_llm_async()
    second = await base_agent._history_for_llm_async()

    assert build.call_count == 1
    assert base_agent.history[0] is base_agent.session_context_message
    assert base_agent.history[0].get_text_content() == _session_message().get_text_content()
    assert first[0] is base_agent.history[0]
    assert second[0] is first[0]
    assert sum(base_agent._is_session_context_message(message) for message in base_agent.history) == 1
    assert base_agent._is_history_valid_for_anthropic(list(first))


@pytest.mark.asyncio
async def test_session_context_reenters_empty_or_replaced_history(base_agent):
    session_context = _session_message()
    base_agent.session_context_message = session_context
    base_agent.history = MessageHistory([Message(role="user", content=[TextBlock(text="before clear")])])

    assert (await base_agent._history_for_llm_async())[0] is session_context
    assert base_agent.dump_message_history()[0] == session_context.to_dict()

    base_agent.clear_history()
    assert list(base_agent.history) == []
    assert list(await base_agent._history_for_llm_async()) == [session_context]

    restored = [
        session_context.to_dict(),
        Message(role="user", content=[TextBlock(text="restored")]).to_dict(),
    ]
    base_agent.restore_message_history(restored)
    assert (await base_agent._history_for_llm_async())[0].get_text_content() == session_context.get_text_content()
    assert sum(base_agent._is_session_context_message(message) for message in base_agent.history) == 1

    base_agent.history = MessageHistory([Message(role="user", content=[TextBlock(text="replacement")])])
    assert (await base_agent._history_for_llm_async())[0].get_text_content() == session_context.get_text_content()
    assert [message.get_text_content() for message in base_agent.history[1:]] == ["replacement"]


@pytest.mark.asyncio
async def test_session_context_is_normal_compaction_input(tmp_path):
    llm = FakeLLM(token_script=[100])
    agent, _ = build_agent(tmp_path, llm=llm)
    agent.session_context_message = _session_message()
    agent.history = long_history(6)

    result = await agent.compress_history()

    assert result.ok
    assert agent.history[0] is agent.session_context_message
    compaction_messages = llm.stream.call_args_list[0].kwargs["messages"]
    compression_count_messages = llm.count_tokens.call_args_list[0].kwargs["messages"]
    assert compression_count_messages is compaction_messages
    assert len(compaction_messages) == 1
    assert "runtime session context" in compaction_messages[0].get_text_content()
    # Like any old historical message, the reminder is represented by the
    # summary rather than forcibly prepended to the compacted effective view.
    assert agent.get_effective_history_for_llm()[0] is agent.conversation.summary


@pytest.mark.asyncio
async def test_resume_reuses_persisted_first_session_reminder(tmp_path):
    from kolega_code.agent.baseagent import BaseAgent

    class SessionAgent(BaseAgent):
        def build_session_context_message(self) -> Message:
            self.build_calls = getattr(self, "build_calls", 0) + 1
            return _session_message(f"runtime for {self.project_path}")

    first, _ = build_agent(tmp_path, agent_cls=SessionAgent, llm=FakeLLM(token_script=[100]))
    await first._history_for_llm_async()
    first.append_user_message("saved turn")
    serialized = first.dump_message_history()

    resumed, _ = build_agent(tmp_path, agent_cls=SessionAgent, llm=FakeLLM(token_script=[100]))
    resumed.restore_message_history(serialized)
    outbound = await resumed._history_for_llm_async()

    assert resumed.build_calls == 1
    assert (
        outbound[0].get_text_content()
        == f'<system-reminder source="session">\nruntime for {tmp_path}\n</system-reminder>'
    )
    assert outbound[1].get_text_content() == "saved turn"
    assert sum(resumed._is_session_context_message(message) for message in resumed.history) == 1


@pytest.mark.asyncio
async def test_count_stream_and_history_share_the_same_session_message(tmp_path):
    llm = FakeLLM(token_script=[100])
    agent, _ = build_agent(tmp_path, llm=llm)
    agent.session_context_message = _session_message()

    async for _chunk in agent.process_message_stream("hello"):
        pass

    counted = llm.count_tokens.call_args.kwargs["messages"]
    streamed = llm.stream.call_args.kwargs["messages"]
    assert counted is streamed
    assert counted[0] is agent.history[0]
    assert [message.get_text_content() for message in streamed[:2]] == [
        _session_message().get_text_content(),
        "hello",
    ]
    assert [message.get_text_content() for message in agent.history[:2]] == [
        _session_message().get_text_content(),
        "hello",
    ]


@pytest.mark.asyncio
async def test_session_context_journals_before_first_user_turn_and_replays_first(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", {})
    agent, _ = build_agent(tmp_path, llm=FakeLLM(token_script=[100]))
    agent.session_recorder = store.recorder(session.session_id)
    agent.session_context_message = _session_message()

    async for _chunk in agent.process_message_stream("hello"):
        pass

    events = store.journal(session.session_id).read_events()
    event_types = [event.event_type for event in events]
    context_index = event_types.index("context.message")
    turn_index = event_types.index("turn.started")
    assert context_index < turn_index
    assert events[context_index].turn_id is None

    replayed = [Message.from_dict(item).get_text_content() for item in store.load(session.session_id).history]
    assert replayed[:2] == [_session_message().get_text_content(), "hello"]


@pytest.mark.parametrize(
    "message, error",
    [
        (Message(role="system", content=[TextBlock(text="wrong role")]), "user role"),
        (Message(role="user", content=[]), "exactly one TextBlock"),
        (Message(role="user", content="plain text"), "exactly one TextBlock"),
        (
            Message(
                role="user",
                content=[ImageBlock(image_type="base64", media_type="image/png", data="AA==")],
            ),
            "text-only",
        ),
        (
            Message(role="user", content=[TextBlock(text="one"), TextBlock(text="two")]),
            "exactly one TextBlock",
        ),
        (Message(role="user", content=[TextBlock(text="cached", cache_checkpoint=True)]), "cache checkpoints"),
        (Message(role="user", content=[TextBlock(text="not wrapped")]), 'source="session"'),
    ],
)
def test_session_context_validation_rejects_unsafe_messages(message, error):
    with pytest.raises(ValueError, match=error):
        from kolega_code.agent.baseagent import BaseAgent

        BaseAgent._validate_session_context_message(message)


def test_adjacent_session_and_user_messages_serialize_for_supported_provider_shapes():
    history = MessageHistory([_session_message(), Message(role="user", content=[TextBlock(text="actual request")])])
    session_text = _session_message().get_text_content()

    anthropic = history.to_anthropic()
    assert [item["role"] for item in anthropic] == ["user", "user"]
    assert [item["content"][0]["text"] for item in anthropic] == [session_text, "actual request"]

    openai = history.to_openai()
    assert [item["role"] for item in openai] == ["user", "user"]
    assert [item["content"][0]["text"] for item in openai] == [session_text, "actual request"]

    google = history.to_google()
    assert [item.role for item in google] == ["user", "user"]
    assert [item.parts[0].text for item in google] == [session_text, "actual request"]

    responses = to_responses_input(history)
    assert [item["role"] for item in responses] == ["user", "user"]
    assert [item["content"][0]["text"] for item in responses] == [session_text, "actual request"]

    tinker, system_text = to_tinker_openai_messages(history, None)
    assert system_text == ""
    assert [item["role"] for item in tinker] == ["user", "user"]
    assert [item["content"][0]["text"] for item in tinker] == [session_text, "actual request"]
