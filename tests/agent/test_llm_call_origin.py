"""LLM call-origin attribution: the BaseAgent main-loop selector, the helper
wraps, and ContextVar semantics (task inheritance, shadowing)."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from kolega_code.llm.ledger import (
    HISTORY_ORIGIN,
    UsageLedger,
    current_llm_call_origin,
    helper_origin,
    llm_call_origin,
)

from .compaction_helpers import build_agent


def test_main_loop_origin_history_iff_recorder_present(tmp_path):
    agent, _ = build_agent(tmp_path)
    agent.session_recorder = object()
    assert agent._main_loop_origin() is HISTORY_ORIGIN

    agent.session_recorder = None
    origin = agent._main_loop_origin()
    assert origin.kind == "primary"
    assert origin.agent_name == agent.agent_name


def test_main_loop_origin_sub_agent_uses_dispatch_context(tmp_path):
    agent, _ = build_agent(tmp_path, sub_agent=True)
    cast(Any, agent).sub_agent_context = {"agent_id": "a1", "parent_tool_call_id": "tc1", "depth": 2}
    origin = agent._main_loop_origin()
    assert origin.kind == "sub_agent"
    assert (origin.agent_id, origin.parent_tool_call_id, origin.depth) == ("a1", "tc1", 2)
    assert origin.agent_name == agent.agent_name


def _current_helper() -> str:
    origin = current_llm_call_origin()
    assert origin is not None
    return origin.helper or ""


def test_origin_shadowing_and_reset():
    assert current_llm_call_origin() is None
    with llm_call_origin(helper_origin("outer")):
        assert _current_helper() == "outer"
        with llm_call_origin(helper_origin("inner")):
            assert _current_helper() == "inner"
        assert _current_helper() == "outer"
    assert current_llm_call_origin() is None


@pytest.mark.asyncio
async def test_gathered_child_tasks_inherit_origin_at_begin():
    ledger = UsageLedger()

    async def child():
        await asyncio.sleep(0)
        return ledger.begin("anthropic", "m")

    with llm_call_origin(helper_origin("web_fetch")):
        request_ids = await asyncio.gather(*(child() for _ in range(3)))

    for request_id in request_ids:
        origin = ledger._records[request_id].origin
        assert origin is not None and origin.helper == "web_fetch"


@pytest.mark.asyncio
async def test_hook_prompt_generate_runs_under_helper_origin(tmp_path, monkeypatch):
    agent, _ = build_agent(tmp_path)
    seen: list = []

    class _Capturing:
        def __init__(self, **kwargs):
            pass

        async def generate(self, **kwargs):
            seen.append(current_llm_call_origin())
            return SimpleNamespace(get_text_content=lambda: "ok")

    monkeypatch.setattr("kolega_code.llm.client.LLMClient", _Capturing)
    await agent._run_hook_prompt("check", None)
    assert seen and seen[0].kind == "helper" and seen[0].helper == "hook_prompt"


@pytest.mark.asyncio
async def test_terminal_security_check_runs_under_helper_origin(monkeypatch, tmp_path):
    from kolega_code.agent.tool_backend import terminal_tool as terminal_tool_module
    from kolega_code.agent.tool_backend.terminal_tool import TerminalTool
    from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig

    seen: list = []

    class _Capturing:
        def __init__(self, **kwargs):
            pass

        async def generate(self, **kwargs):
            seen.append(current_llm_call_origin())
            return SimpleNamespace(get_text_content=lambda: "safe")

    tool = object.__new__(TerminalTool)
    tool.config = AgentConfig(
        anthropic_api_key="k",
        fast_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-opus-5", rate_limits=RateLimitConfig()),
    )
    tool.caller = SimpleNamespace(usage_ledger=None, scratchpad_dir=None, project_path=tmp_path)
    monkeypatch.setattr(terminal_tool_module, "LLMClient", _Capturing)

    ok, _ = await tool._run_command_security_check("ls")
    assert ok is True
    assert seen and seen[0].helper == "terminal_security"


@pytest.mark.asyncio
async def test_compression_summarizer_runs_under_helper_origin(tmp_path):
    from tests.agent.compaction_helpers import FakeLLM, long_history

    seen: list = []
    llm = FakeLLM(summary_text="SUMMARY")
    original_stream = llm._stream

    async def capturing_stream(*args, **kwargs):
        seen.append(current_llm_call_origin())
        return await original_stream(*args, **kwargs)

    llm.stream = AsyncMock(side_effect=capturing_stream)

    agent, _ = build_agent(tmp_path, llm=llm)
    agent.history = long_history(6)
    result = await agent.compress_history()
    assert result.ok
    assert seen and seen[0] is not None and seen[0].helper == "compression"
