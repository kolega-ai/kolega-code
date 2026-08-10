"""Every agent subclass must accept and store the llm_trace_sink kwarg, landing
it on both the flat attribute and the context telemetry — the same contract as
usage_ledger — and create_llm_client must forward it to plain and instrumented
clients as LLMClient's trace_sink."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.agent.browseragent import BrowserAgent
from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.generalagent import GeneralAgent
from kolega_code.agent.investigationagent import InvestigationAgent
from kolega_code.agent.planningagent import PlanningAgent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.instrumented_client import InstrumentedLLMClient

from .compaction_helpers import make_agent_config


def _sink(record):
    pass


@pytest.mark.parametrize("agent_cls", [CoderAgent, GeneralAgent, InvestigationAgent, PlanningAgent, BrowserAgent])
def test_agent_subclasses_carry_llm_trace_sink(tmp_path, agent_cls):
    agent = agent_cls(
        project_path=tmp_path,
        workspace_id="test_ws",
        thread_id=str(uuid.uuid4()),
        connection_manager=AsyncMock(spec=AgentConnectionManager),
        config=make_agent_config(),
        agent_mode=AgentMode.CLI,
        llm_trace_sink=_sink,
    )
    assert agent.llm_trace_sink is _sink
    assert agent.context.telemetry.llm_trace_sink is _sink


@pytest.mark.parametrize("agent_cls", [CoderAgent, PlanningAgent])
def test_llm_trace_sink_defaults_to_none(tmp_path, agent_cls):
    agent = agent_cls(
        project_path=tmp_path,
        workspace_id="test_ws",
        thread_id=str(uuid.uuid4()),
        connection_manager=AsyncMock(spec=AgentConnectionManager),
        config=make_agent_config(),
        agent_mode=AgentMode.CLI,
    )
    assert agent.llm_trace_sink is None
    assert agent.context.telemetry.llm_trace_sink is None


def test_create_llm_client_forwards_trace_sink_to_plain_client(tmp_path):
    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="test_ws",
        thread_id=str(uuid.uuid4()),
        connection_manager=AsyncMock(spec=AgentConnectionManager),
        config=make_agent_config(),
        agent_mode=AgentMode.CLI,
        llm_trace_sink=_sink,
    )
    client = agent.context.create_llm_client(agent.agent_name)
    assert not isinstance(client, InstrumentedLLMClient)
    assert client._trace_sink is _sink


def test_create_llm_client_forwards_trace_sink_to_instrumented_client(tmp_path):
    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="test_ws",
        thread_id=str(uuid.uuid4()),
        connection_manager=AsyncMock(spec=AgentConnectionManager),
        config=make_agent_config(),
        agent_mode=AgentMode.CLI,
        langfuse_client=MagicMock(),
        llm_trace_sink=_sink,
    )
    client = agent.context.create_llm_client(agent.agent_name)
    assert isinstance(client, InstrumentedLLMClient)
    assert client._trace_sink is _sink
