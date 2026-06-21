"""Auto-compaction must work for sub-agent types too (shared BaseAgent loop)."""

import pytest

from .compaction_helpers import FakeLLM, build_agent, long_history


@pytest.mark.asyncio
async def test_general_agent_auto_compacts(tmp_path):
    from kolega_code.agent.generalagent import GeneralAgent

    agent, _cm = build_agent(
        tmp_path,
        agent_cls=GeneralAgent,
        sub_agent=True,
        llm=FakeLLM(token_script=[900, 300]),  # over budget, then under after compaction
    )
    agent.history = long_history(6)

    # Drain the shared loop; the sub-agent should compact before generating.
    async for _chunk in agent.process_message_stream("do the task"):
        pass

    # The summarization ran inside the sub-agent (shared BaseAgent loop).
    assert agent.conversation.summary is not None
    assert agent.last_compression_index is not None
