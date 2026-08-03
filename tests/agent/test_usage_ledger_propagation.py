"""Every agent subclass must accept and store the usage_ledger kwarg, landing it
on both the flat attribute and the context telemetry (which feeds
create_llm_client for the agent's own LLM client)."""

import uuid
from unittest.mock import AsyncMock

import pytest

from kolega_code.agent.browseragent import BrowserAgent
from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.generalagent import GeneralAgent
from kolega_code.agent.investigationagent import InvestigationAgent
from kolega_code.agent.planningagent import PlanningAgent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.ledger import UsageLedger

from .compaction_helpers import make_agent_config


@pytest.mark.parametrize("agent_cls", [CoderAgent, GeneralAgent, InvestigationAgent, PlanningAgent, BrowserAgent])
def test_agent_subclasses_carry_usage_ledger(tmp_path, agent_cls):
    ledger = UsageLedger()
    agent = agent_cls(
        project_path=tmp_path,
        workspace_id="test_ws",
        thread_id=str(uuid.uuid4()),
        connection_manager=AsyncMock(spec=AgentConnectionManager),
        config=make_agent_config(),
        agent_mode=AgentMode.CLI,
        usage_ledger=ledger,
    )
    assert agent.usage_ledger is ledger
    assert agent.context.telemetry.usage_ledger is ledger
