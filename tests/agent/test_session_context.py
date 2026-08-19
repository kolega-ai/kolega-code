"""Session-context history behavior."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from kolega_code.cli.session_store import SessionStore
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.providers.responses_common import to_responses_input
from kolega_code.llm.providers.tinker import _openai_messages as to_tinker_openai_messages

from .compaction_helpers import FakeLLM, build_agent, long_history, text_msg


def _session_context_text(agent) -> str:
    message = agent.session_context_message
    assert message.role == "user"
    assert isinstance(message.content, list)
    assert len(message.content) == 1
    block = message.content[0]
    assert isinstance(block, TextBlock)
    return block.text


def test_new_agent_starts_with_one_session_context_message(base_agent):
    assert list(base_agent.history) == [base_agent.session_context_message]

    text = _session_context_text(base_agent)
    assert text.startswith('<system-reminder source="session">\n')
    assert text.endswith("\n</system-reminder>")
    assert f"Working directory: {base_agent.project_path}" in text
    assert f"Model: {base_agent.build_prompt_context().model_name}" in text


@pytest.mark.asyncio
async def test_session_context_is_first_provider_history_message(base_agent):
    base_agent.history.append(text_msg("user", "hello"))

    prepared = await base_agent._history_for_llm_async()

    assert prepared[0] is base_agent.session_context_message
    assert prepared[1].get_text_content() == "hello"


def test_clear_resets_history_to_constructor_session_context(base_agent):
    base_agent.history.append(text_msg("user", "before clear"))

    base_agent.clear_history()

    assert list(base_agent.history) == [base_agent.session_context_message]


def test_restore_is_authoritative_for_legacy_history(base_agent):
    restored = text_msg("user", "restored")

    base_agent.restore_message_history([restored.to_dict()])

    assert [message.to_dict() for message in base_agent.history] == [restored.to_dict()]


@pytest.mark.asyncio
async def test_restore_is_authoritative_for_empty_history(base_agent):
    base_agent.restore_message_history([])

    assert list(base_agent.history) == []
    assert list(await base_agent._history_for_llm_async()) == []


def test_restore_preserves_persisted_session_context_without_rebuilding(base_agent):
    serialized = [base_agent.session_context_message.to_dict(), text_msg("user", "restored").to_dict()]
    persisted_text = '<system-reminder source="session">\npersisted runtime\n</system-reminder>'
    serialized[0]["content"][0]["text"] = persisted_text

    base_agent.restore_message_history(serialized)

    assert [message.get_text_content() for message in base_agent.history] == [persisted_text, "restored"]


@pytest.mark.asyncio
async def test_compaction_receives_session_context_as_ordinary_history(tmp_path):
    llm = FakeLLM(summary_text="SUMMARY: includes runtime")
    agent, _ = build_agent(tmp_path, llm=llm)
    agent.history.extend(long_history(6))

    result = await agent.compress_history()

    assert result.ok
    compaction_input = llm.stream.call_args_list[0].kwargs["messages"]
    assert _session_context_text(agent) in compaction_input[0].get_text_content()


def test_fresh_session_context_is_journaled_during_construction(tmp_path):
    recorder = MagicMock()
    agent, _ = build_agent(tmp_path, session_recorder=recorder)

    recorder.record_context_message.assert_called_once_with(agent.session_context_message)


def test_restored_legacy_history_is_not_backfilled(tmp_path):
    recorder = MagicMock()
    agent, _ = build_agent(tmp_path)
    restored = text_msg("user", "legacy session")
    agent.restore_message_history([restored.to_dict()])
    agent.session_recorder = recorder

    assert [message.get_text_content() for message in agent.history] == ["legacy session"]
    recorder.record_context_message.assert_not_called()


def test_clear_starts_new_epoch_with_same_session_context(tmp_path):
    recorder = MagicMock()
    agent, _ = build_agent(tmp_path, session_recorder=recorder)
    agent.history.append(text_msg("user", "before clear"))

    agent.clear_history()

    assert list(agent.history) == [agent.session_context_message]
    recorder.start_epoch.assert_called_once_with("agent_clear_command")
    assert [call.args for call in recorder.record_context_message.call_args_list] == [
        (agent.session_context_message,),
        (agent.session_context_message,),
    ]


@pytest.mark.asyncio
async def test_restore_keeps_persisted_session_context_for_provider_dispatch(tmp_path):
    original, _ = build_agent(tmp_path)
    original.history.append(text_msg("user", "prior request"))
    serialized_history = [message.to_dict() for message in original.history]
    persisted_text = '<system-reminder source="session">\npersisted runtime\n</system-reminder>'
    serialized_history[0]["content"][0]["text"] = persisted_text

    resumed, _ = build_agent(tmp_path)
    resumed.restore_message_history(deepcopy(serialized_history))

    prepared = await resumed._history_for_llm_async()
    assert prepared[0].get_text_content() == persisted_text
    assert len(prepared) == len(serialized_history)


def test_session_context_round_trips_through_message_serialization(base_agent):
    serialized = base_agent.session_context_message.to_dict()

    assert Message.from_dict(serialized).to_dict() == serialized


def test_session_context_is_counted_by_default(tmp_path):
    token_counter = FakeLLM(proxy=True)
    agent, _ = build_agent(tmp_path, llm=token_counter)
    agent.history.append(text_msg("user", "hello"))

    counted = agent.get_effective_history_for_llm()

    assert counted == agent.history
    assert counted[0] is agent.session_context_message
    assert len(counted) == 2


@pytest.mark.asyncio
async def test_session_context_journals_before_first_user_turn_and_replays_first(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", {})
    agent, _ = build_agent(
        tmp_path,
        llm=FakeLLM(token_script=[100]),
        session_recorder=store.recorder(session.session_id),
    )

    async for _chunk in agent.process_message_stream("hello"):
        pass

    events = store.journal(session.session_id).read_events()
    event_types = [event.event_type for event in events]
    assert event_types.index("context.message") < event_types.index("turn.started")
    replayed = [Message.from_dict(item).get_text_content() for item in store.load(session.session_id).history]
    assert replayed[:2] == [_session_context_text(agent), "hello"]


def test_adjacent_session_and_user_messages_serialize_for_supported_provider_shapes(base_agent):
    history = MessageHistory([base_agent.session_context_message, text_msg("user", "actual request")])
    session_text = _session_context_text(base_agent)

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
