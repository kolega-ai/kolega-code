"""Every helper that constructs its own LLM client must thread the caller's
usage_ledger into it: think_hard (both branches), web_fetch, the terminal
security check, and hook prompt runners."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.events import AgentConnectionManager
from kolega_code.agent.tool_backend import terminal_tool as terminal_tool_module
from kolega_code.agent.tool_backend import think_hard_tool as think_hard_module
from kolega_code.agent.tool_backend import web_fetch_tool as web_fetch_module
from kolega_code.agent.tool_backend.think_hard_tool import ThinkHardTool
from kolega_code.agent.tool_backend.web_fetch_tool import WebFetchTool
from kolega_code.agent.tool_backend.terminal_tool import TerminalTool
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.llm.ledger import UsageLedger

SENTINEL_LEDGER = UsageLedger()


def _model_config(model="claude-opus-5"):
    return ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model=model,
        rate_limits=RateLimitConfig(requests_per_minute=10, tokens_per_minute=100000, max_retries=1),
    )


def _config():
    return AgentConfig(
        anthropic_api_key="test-key",
        fast_config=_model_config(),
        thinking_config=_model_config(),
    )


class _CapturingClient:
    """Stands in for LLMClient/InstrumentedLLMClient; records ctor kwargs."""

    captured: list[dict] = []
    langfuse: object = None
    environment: str = "test"
    usage_recorder: object = None

    def __init__(self, **kwargs):
        type(self).captured.append(kwargs)

    async def generate(self, **kwargs):
        return SimpleNamespace(get_text_content=lambda: "yes")

    def stream(self, **kwargs):
        raise RuntimeError("stream not exercised in threading tests")


@pytest.fixture(autouse=True)
def _reset_capture():
    _CapturingClient.captured = []
    yield
    _CapturingClient.captured = []


@pytest.mark.asyncio
async def test_think_hard_plain_branch_threads_ledger(monkeypatch):
    caller = Mock()
    caller.agent_name = "test_agent"
    caller.usage_ledger = SENTINEL_LEDGER
    caller.llm = object()  # not an InstrumentedLLMClient -> plain branch
    tool = ThinkHardTool(
        project_path="/tmp",
        workspace_id="ws",
        thread_id="th",
        connection_manager=AsyncMock(spec=AgentConnectionManager),
        config=_config(),
        caller=caller,
    )
    tool.log_info = AsyncMock()
    tool.log_error = AsyncMock()
    monkeypatch.setattr(think_hard_module, "LLMClient", _CapturingClient)

    await tool.think_hard("problem")  # client.stream raises; tool swallows into its error result

    assert len(_CapturingClient.captured) == 1
    assert _CapturingClient.captured[0]["usage_ledger"] is SENTINEL_LEDGER


@pytest.mark.asyncio
async def test_think_hard_instrumented_branch_threads_ledger(monkeypatch):
    class _CapturingInstrumented(_CapturingClient):
        pass

    caller_llm = _CapturingInstrumented()
    _CapturingClient.captured = []  # ignore the fixture instance above
    caller_llm.langfuse = Mock()
    caller_llm.environment = "test"
    caller = Mock()
    caller.agent_name = "test_agent"
    caller.workspace_id = "ws"
    caller.thread_id = "th"
    caller.user_id = "u"
    caller.user_email = "u@example.com"
    caller.usage_ledger = SENTINEL_LEDGER
    caller.llm = caller_llm
    tool = ThinkHardTool(
        project_path="/tmp",
        workspace_id="ws",
        thread_id="th",
        connection_manager=AsyncMock(spec=AgentConnectionManager),
        config=_config(),
        caller=caller,
    )
    tool.log_info = AsyncMock()
    tool.log_error = AsyncMock()
    monkeypatch.setattr(think_hard_module, "InstrumentedLLMClient", _CapturingInstrumented)

    await tool.think_hard("problem")

    assert len(_CapturingClient.captured) == 1
    assert _CapturingClient.captured[0]["usage_ledger"] is SENTINEL_LEDGER


@pytest.mark.parametrize("instrumented", [False, True])
def test_web_fetch_build_client_threads_ledger(monkeypatch, instrumented):
    tool = object.__new__(WebFetchTool)
    tool.config = _config()

    monkeypatch.setattr(web_fetch_module, "LLMClient", _CapturingClient)
    if instrumented:

        class _CapturingInstrumented(_CapturingClient):
            pass

        caller_llm = _CapturingInstrumented()
        _CapturingClient.captured = []
        caller_llm.langfuse = Mock()
        caller_llm.usage_recorder = None
        monkeypatch.setattr(web_fetch_module, "InstrumentedLLMClient", _CapturingInstrumented)
        tool.caller = SimpleNamespace(
            llm=caller_llm, usage_ledger=SENTINEL_LEDGER, agent_name="a", workspace_id="w", thread_id="t"
        )
    else:
        tool.caller = SimpleNamespace(usage_ledger=SENTINEL_LEDGER)

    tool._build_client()

    assert len(_CapturingClient.captured) == 1
    assert _CapturingClient.captured[0]["usage_ledger"] is SENTINEL_LEDGER


@pytest.mark.asyncio
async def test_terminal_security_check_threads_ledger(monkeypatch):
    tool = object.__new__(TerminalTool)
    tool.config = _config()
    tool.caller = SimpleNamespace(usage_ledger=SENTINEL_LEDGER, scratchpad_dir=None, project_path=Path("/tmp"))
    monkeypatch.setattr(terminal_tool_module, "LLMClient", _CapturingClient)

    await tool._run_command_security_check("ls")

    assert len(_CapturingClient.captured) == 1
    assert _CapturingClient.captured[0]["usage_ledger"] is SENTINEL_LEDGER


@pytest.mark.asyncio
async def test_hook_prompt_client_threads_ledger(tmp_path, monkeypatch):
    from tests.agent.compaction_helpers import build_agent

    agent, _ = build_agent(tmp_path)
    agent.usage_ledger = SENTINEL_LEDGER
    monkeypatch.setattr("kolega_code.llm.client.LLMClient", _CapturingClient)

    result = await agent._run_hook_prompt("check something", None)

    assert result == "yes"
    assert len(_CapturingClient.captured) == 1
    assert _CapturingClient.captured[0]["usage_ledger"] is SENTINEL_LEDGER
