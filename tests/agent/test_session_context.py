"""Focused session-context lifecycle coverage."""

from unittest.mock import MagicMock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.llm.models import TextBlock

from .compaction_helpers import FakeLLM, build_agent, long_history, text_msg


def _session_context_text(agent: BaseAgent) -> str:
    message = agent.session_context_message
    assert message.role == "user"
    assert isinstance(message.content, list)
    assert len(message.content) == 1
    block = message.content[0]
    assert isinstance(block, TextBlock)
    return block.text


def test_new_agent_starts_with_session_context(base_agent):
    assert list(base_agent.history) == [base_agent.session_context_message]

    context = base_agent.build_prompt_context()
    assert _session_context_text(base_agent) == "\n".join(
        [
            '<system-reminder source="session">',
            f"Working directory: {context.project_path}",
            f"Is directory a git repo: {str(context.is_git_repo).lower()}",
            f"Platform: {context.platform}",
            f"Model: {context.model_name}",
            f"Model supports vision: {str(context.model_supports_vision).lower()}",
            "</system-reminder>",
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "serialized",
    [
        pytest.param([], id="empty"),
        pytest.param([text_msg("user", "legacy").to_dict()], id="legacy"),
    ],
)
async def test_restore_is_authoritative(base_agent, serialized):
    base_agent.restore_message_history(serialized)

    assert [message.to_dict() for message in base_agent.history] == serialized
    assert [message.to_dict() for message in await base_agent._history_for_llm_async()] == serialized


def test_constructor_journals_session_context_and_clear_starts_new_epoch(
    tmp_path, mock_connection_manager, agent_config
):
    recorder = MagicMock()
    agent = BaseAgent(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id="test_thread",
        connection_manager=mock_connection_manager,
        config=agent_config,
        session_recorder=recorder,
    )
    agent.history.append(text_msg("user", "before clear"))

    agent.clear_history()

    assert list(agent.history) == [agent.session_context_message]
    recorder.start_epoch.assert_called_once_with("agent_clear_command")
    assert [call.args for call in recorder.record_context_message.call_args_list] == [
        (agent.session_context_message,),
        (agent.session_context_message,),
    ]


@pytest.mark.asyncio
async def test_compaction_receives_session_context_as_ordinary_history(tmp_path):
    llm = FakeLLM(summary_text="SUMMARY: includes runtime")
    agent, _ = build_agent(tmp_path, llm=llm)
    agent.history.extend(long_history(6))

    result = await agent.compress_history()

    assert result.ok
    compaction_input = llm.stream.call_args_list[0].kwargs["messages"]
    assert _session_context_text(agent) in compaction_input[0].get_text_content()
