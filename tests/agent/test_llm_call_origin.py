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

from .compaction_helpers import FakeLLM, FakeStream, build_agent


class OriginProbeStream(FakeStream):
    """A stream that records the active origin at every lifecycle phase.

    Models providers that sample (and emit their trace record) inside
    ``__aenter__`` rather than at ``stream()``-call time — the shape that makes
    a too-narrow ``llm_call_origin`` scope observable.
    """

    def __init__(self, final_message=None, probes=None, text_events=()):
        super().__init__(final_message)
        self.probes = probes if probes is not None else []
        self._events = list(text_events)

    async def __aenter__(self):
        self.probes.append(("aenter", current_llm_call_origin()))
        return self

    async def __anext__(self):
        self.probes.append(("anext", current_llm_call_origin()))
        if self._events:
            return SimpleNamespace(type="text", text=self._events.pop(0), thinking=None, tool_call_delta=None)
        raise StopAsyncIteration

    async def get_final_message(self):
        self.probes.append(("final", current_llm_call_origin()))
        return self._final


class OriginProbeLLM(FakeLLM):
    """FakeLLM whose streams are OriginProbeStreams sharing one probe list."""

    def __init__(self, *args, text_events=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.probes: list = []
        self._text_events = list(text_events)
        self.stream = AsyncMock(side_effect=self._probe_stream)

    async def _probe_stream(self, *args, **kwargs):
        self.probes.append(("stream", current_llm_call_origin()))
        inner = await self._stream(*args, **kwargs)
        return OriginProbeStream(inner._final, self.probes, text_events=self._text_events)


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
async def test_compression_summarizer_origin_live_for_full_stream_lifecycle(tmp_path):
    from tests.agent.compaction_helpers import long_history

    llm = OriginProbeLLM(summary_text="SUMMARY: condensed older turns.")
    agent, _ = build_agent(tmp_path, llm=llm)
    agent.history = long_history(6)
    result = await agent.compress_history()
    assert result.ok
    assert {phase for phase, _ in llm.probes} == {"stream", "aenter", "anext", "final"}
    for phase, origin in llm.probes:
        assert origin is not None and origin.helper == "compression", phase


def _assert_phases_carry_origin(probes, expected_kind):
    assert {phase for phase, _ in probes} == {"stream", "aenter", "anext", "final"}
    for phase, origin in probes:
        assert origin is not None, f"no origin at {phase}"
        assert origin.kind == expected_kind, f"{phase}: {origin.kind}"


@pytest.mark.asyncio
async def test_main_loop_origin_live_for_full_stream_lifecycle(tmp_path):
    llm = OriginProbeLLM()
    agent, _ = build_agent(tmp_path, llm=llm)
    async for _chunk in agent.process_message_stream("hi"):
        pass
    _assert_phases_carry_origin(llm.probes, "primary")


@pytest.mark.asyncio
async def test_sub_agent_origin_live_for_full_stream_lifecycle(tmp_path):
    llm = OriginProbeLLM()
    agent, _ = build_agent(tmp_path, sub_agent=True, llm=llm)
    cast(Any, agent).sub_agent_context = {"agent_id": "a1", "parent_tool_call_id": "tc1", "depth": 1}
    async for _chunk in agent.process_message_stream("task"):
        pass
    _assert_phases_carry_origin(llm.probes, "sub_agent")


@pytest.mark.asyncio
async def test_origin_visible_between_chunks_and_cleared_by_same_task_close(tmp_path):
    # A mid-stream text event forces a yield while the origin scope is open.
    llm = OriginProbeLLM(text_events=["x" * 60])
    agent, _ = build_agent(tmp_path, llm=llm)
    gen = agent.process_message_stream("hi")
    chunk = await gen.__anext__()
    assert chunk["type"] == "response"
    # Documented property: the consumer observes the origin between chunks.
    origin = current_llm_call_origin()
    assert origin is not None and origin.kind == "primary"
    await gen.aclose()
    assert current_llm_call_origin() is None


@pytest.mark.asyncio
async def test_cross_task_generator_close_does_not_raise(tmp_path):
    llm = OriginProbeLLM(text_events=["x" * 60])
    agent, _ = build_agent(tmp_path, llm=llm)
    gen = agent.process_message_stream("hi")
    await gen.__anext__()
    # Closing from another task runs the origin scope's finally in a foreign
    # Context, where ContextVar.reset(token) raises; the resilient reset must
    # swallow it so teardown never surfaces a spurious ValueError.
    await asyncio.create_task(gen.aclose())
