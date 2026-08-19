"""Generation-scoped session context stays outside history but on every request."""

from unittest.mock import MagicMock

import pytest

from kolega_code.llm.models import ImageBlock, Message, MessageHistory, TextBlock, ToolCall
from kolega_code.llm.providers.responses_common import to_responses_input
from kolega_code.llm.providers.tinker import _openai_messages as to_tinker_openai_messages

from .compaction_helpers import FakeLLM, build_agent, long_history


def _session_message(text: str = "runtime session context") -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


@pytest.mark.asyncio
async def test_session_context_is_built_once_and_prepended_after_history_repair(base_agent):
    build = MagicMock(return_value=_session_message())
    base_agent.build_session_context_message = build
    base_agent.history = MessageHistory(
        [
            Message(role="assistant", content=[ToolCall(id="call-1", name="read", input={})]),
            Message(role="user", content=[TextBlock(text="actual user message")]),
        ]
    )

    first = await base_agent._messages_for_llm_async()
    second = await base_agent._messages_for_llm_async()

    assert build.call_count == 1
    assert first[0] is base_agent.session_context_message
    assert first[0].get_text_content() == "runtime session context"
    assert second[0] is first[0]
    assert base_agent.history[0].role == "assistant"
    assert all(message is not first[0] for message in base_agent.history)
    # Repair completed before prepending, so the runtime context did not become
    # the user message paired with the orphaned tool call.
    assert base_agent._is_history_valid_for_anthropic(list(first[1:]))
    assert first[1].role == "assistant"


@pytest.mark.asyncio
async def test_session_context_survives_clear_restore_and_history_replacement(base_agent):
    session_context = _session_message()
    base_agent.session_context_message = session_context
    base_agent.history = MessageHistory([Message(role="user", content=[TextBlock(text="before clear")])])

    assert (await base_agent._messages_for_llm_async())[0] is session_context
    base_agent.clear_history()
    assert list(base_agent.history) == []
    assert list(await base_agent._messages_for_llm_async()) == [session_context]

    base_agent.restore_message_history([Message(role="user", content=[TextBlock(text="restored")]).to_dict()])
    assert (await base_agent._messages_for_llm_async())[0] is session_context
    assert [message.get_text_content() for message in base_agent.history] == ["restored"]

    base_agent.history = MessageHistory([Message(role="user", content=[TextBlock(text="replacement")])])
    assert (await base_agent._messages_for_llm_async())[0] is session_context
    assert all(
        Message.from_dict(message).get_text_content() != "runtime session context"
        for message in base_agent.dump_message_history()
    )


@pytest.mark.asyncio
async def test_session_context_survives_compaction_without_entering_summary_input(tmp_path):
    llm = FakeLLM(token_script=[100])
    agent, _ = build_agent(tmp_path, llm=llm)
    agent.session_context_message = _session_message()
    agent.history = long_history(6)

    result = await agent.compress_history()

    assert result.ok
    outbound = await agent._messages_for_llm_async()
    assert outbound[0] is agent.session_context_message
    assert outbound[0].get_text_content() == "runtime session context"
    assert all(message.get_text_content() != "runtime session context" for message in agent.history)
    compaction_messages = llm.stream.call_args_list[0].kwargs["messages"]
    assert all(message.get_text_content() != "runtime session context" for message in compaction_messages)


@pytest.mark.asyncio
async def test_new_agent_generation_rebuilds_session_context_after_resume(tmp_path):
    from kolega_code.agent.baseagent import BaseAgent

    class SessionAgent(BaseAgent):
        def build_session_context_message(self) -> Message:
            self.build_calls = getattr(self, "build_calls", 0) + 1
            return _session_message(f"runtime for {self.project_path}")

    first, _ = build_agent(tmp_path, agent_cls=SessionAgent, llm=FakeLLM(token_script=[100]))
    first.history = MessageHistory([Message(role="user", content=[TextBlock(text="saved turn")])])
    assert (await first._messages_for_llm_async())[0].get_text_content() == f"runtime for {tmp_path}"
    serialized = first.dump_message_history()

    resumed, _ = build_agent(tmp_path, agent_cls=SessionAgent, llm=FakeLLM(token_script=[100]))
    resumed.restore_message_history(serialized)
    outbound = await resumed._messages_for_llm_async()

    assert resumed.build_calls == 1
    assert outbound[0].get_text_content() == f"runtime for {tmp_path}"
    assert outbound[1].get_text_content() == "saved turn"
    assert all(message.get_text_content() != f"runtime for {tmp_path}" for message in resumed.history)


@pytest.mark.asyncio
async def test_count_and_stream_receive_the_same_composed_history(tmp_path):
    llm = FakeLLM(token_script=[100])
    agent, _ = build_agent(tmp_path, llm=llm)
    agent.session_context_message = _session_message()

    async for _chunk in agent.process_message_stream("hello"):
        pass

    counted = llm.count_tokens.call_args.kwargs["messages"]
    streamed = llm.stream.call_args.kwargs["messages"]
    assert counted is streamed
    assert [message.get_text_content() for message in streamed[:2]] == [
        "runtime session context",
        "hello",
    ]
    assert [message.get_text_content() for message in agent.history[:1]] == ["hello"]


@pytest.mark.asyncio
async def test_session_context_cannot_change_after_initialization(base_agent):
    base_agent.session_context_message = _session_message("stable")
    await base_agent._messages_for_llm_async()

    content = base_agent.session_context_message.content
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, TextBlock)
    block.text = "changed"

    with pytest.raises(RuntimeError, match="cannot change"):
        await base_agent._messages_for_llm_async()


@pytest.mark.parametrize(
    "message, error",
    [
        (Message(role="system", content=[TextBlock(text="wrong role")]), "user role"),
        (Message(role="user", content=[]), "TextBlock content"),
        (Message(role="user", content="plain text"), "TextBlock content"),
        (
            Message(
                role="user",
                content=[ImageBlock(image_type="base64", media_type="image/png", data="AA==")],
            ),
            "text-only",
        ),
        (Message(role="user", content=[TextBlock(text="cached", cache_checkpoint=True)]), "cache checkpoints"),
    ],
)
def test_session_context_validation_rejects_unsafe_messages(message, error):
    with pytest.raises(ValueError, match=error):
        from kolega_code.agent.baseagent import BaseAgent

        BaseAgent._validate_session_context_message(message)


def test_adjacent_session_and_user_messages_serialize_for_supported_provider_shapes():
    history = MessageHistory([_session_message(), Message(role="user", content=[TextBlock(text="actual request")])])

    anthropic = history.to_anthropic()
    assert [item["role"] for item in anthropic] == ["user", "user"]
    assert [item["content"][0]["text"] for item in anthropic] == ["runtime session context", "actual request"]

    openai = history.to_openai()
    assert [item["role"] for item in openai] == ["user", "user"]
    assert [item["content"][0]["text"] for item in openai] == ["runtime session context", "actual request"]

    google = history.to_google()
    assert [item.role for item in google] == ["user", "user"]
    assert [item.parts[0].text for item in google] == ["runtime session context", "actual request"]

    responses = to_responses_input(history)
    assert [item["role"] for item in responses] == ["user", "user"]
    assert [item["content"][0]["text"] for item in responses] == ["runtime session context", "actual request"]

    tinker, system_text = to_tinker_openai_messages(history, None)
    assert system_text == ""
    assert [item["role"] for item in tinker] == ["user", "user"]
    assert [item["content"][0]["text"] for item in tinker] == ["runtime session context", "actual request"]
