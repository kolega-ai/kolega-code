"""Background child usage attribution: launch turn capture and agent ownership."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.tool_backend.agent_tool import AgentTool
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.session_usage import (
    LLM_MESSAGE_EVENT,
    LLM_REQUEST_FAILED_EVENT,
    SessionUsageSink,
)
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.llm.ledger import UsageLedger, helper_origin, llm_call_origin
from kolega_code.llm.models import Message, TextBlock
from kolega_code.llm.usage import normalize_usage
from kolega_code.session.background import BackgroundAgentManager


def _config() -> AgentConfig:
    return AgentConfig(
        anthropic_api_key="test-key",
        openai_api_key="test-key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-opus-5",
            rate_limits=RateLimitConfig(),
        ),
        fast_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-opus-5-haiku",
            rate_limits=RateLimitConfig(),
        ),
    )


def _usage(inp: int = 10, out: int = 5):
    return normalize_usage({"input_tokens": inp, "output_tokens": out}, "anthropic", "claude-opus-5")


async def _harness(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", {})
    recorder = store.recorder(session.session_id)
    ledger = UsageLedger()
    sink = SessionUsageSink(store.journal(session.session_id), recorder, ledger, mode="tui")
    ledger.observer = sink
    await sink.start()
    return project, store, session, recorder, ledger, sink


def _caller(recorder, ledger, manager: BackgroundAgentManager | None = None) -> Mock:
    caller = Mock()
    caller.agent_name = "root-agent"
    caller.current_tool_call_id = None
    caller.current_tool_execution_id = None
    caller.sub_agent = False
    caller.sub_agent_context = None
    caller.protected_files = []
    caller.workspace_env_var_descriptions = {}
    caller.workspace_memories = []
    caller.prompt_extensions = []
    caller.tool_extensions = []
    caller.usage_recorder = None
    caller.sub_agent_recorder = None
    caller.max_iterations = None
    caller.user_id = None
    caller.user_email = None
    caller.project_template_slug = None
    caller.agent_mode = None
    caller.prompt_provider = None
    caller.usage_ledger = ledger
    caller.llm_trace_sink = None
    caller.hook_dispatcher = None
    caller.fire_hook = AsyncMock()
    caller.scratchpad_dir = None
    caller.session_recorder = recorder
    caller.get_background_jobs = (lambda: manager) if manager is not None else Mock(return_value=None)
    return caller


def _settling_agent_factory(
    *,
    settle_kind: str,
    entered: asyncio.Event,
    release: asyncio.Event | None = None,
    request_ids: list[str],
):
    class SettlingAgent:
        agent_name = "general-agent"

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.usage_ledger = kwargs["usage_ledger"]
            self.session_context_message = Message(
                role="user",
                content=[TextBlock('<system-reminder source="session">\nchild context\n</system-reminder>')],
            )
            self._message_history = [self.session_context_message.to_dict()]
            self.total_tokens_used = 0
            self.parent_tool_call_id = None
            self.conversation_id = None
            self.sub_agent_context = None

        async def process_message_stream(self, task: str):
            del task
            entered.set()
            if release is not None:
                await release.wait()
            with llm_call_origin(helper_origin("compression")):
                request_id = self.usage_ledger.begin("anthropic", "claude-opus-5")
            request_ids.append(request_id)
            if settle_kind == "response":
                usage = _usage()
                self.usage_ledger.record_response(
                    request_id,
                    usage,
                    message=Message(role="assistant", content=[TextBlock(text="helper")], usage=usage),
                )
            else:
                self.usage_ledger.record_failure(request_id, "rate limited")
            self._message_history.append({"role": "assistant", "content": [{"type": "text", "text": "done"}]})
            yield {"type": "response", "content": "done", "complete": True, "uuid": str(uuid.uuid4())}

        def dump_message_history(self):
            return list(self._message_history)

        async def recap_agent_outcome(self) -> str:
            return "done"

        async def cleanup(self) -> None:
            return None

    return SettlingAgent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settle_kind", "event_type"),
    [("response", LLM_MESSAGE_EVENT), ("failure", LLM_REQUEST_FAILED_EVENT)],
)
async def test_background_child_usage_keeps_launch_turn_and_child_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env,
    settle_kind: str,
    event_type: str,
) -> None:
    del isolated_cli_env
    project, store, session, recorder, ledger, sink = await _harness(tmp_path)
    manager = BackgroundAgentManager(recorder)
    caller = _caller(recorder, ledger, manager)
    connection_manager = AsyncMock()
    tool = AgentTool(
        project_path=project,
        workspace_id="w",
        thread_id="t",
        connection_manager=connection_manager,
        config=_config(),
        caller=caller,
    )

    entered = asyncio.Event()
    release = asyncio.Event()
    request_ids: list[str] = []
    monkeypatch.setattr(
        "kolega_code.agent.generalagent.GeneralAgent",
        _settling_agent_factory(
            settle_kind=settle_kind,
            entered=entered,
            release=release,
            request_ids=request_ids,
        ),
    )

    turn_a = recorder.start_turn(Message(role="user", content=[TextBlock(text="launch A")]))
    receipt = json.loads(await tool.dispatch_general_agent("work", background=True))
    await asyncio.wait_for(entered.wait(), timeout=3)
    recorder.finish_turn("completed")
    turn_b = recorder.start_turn(Message(role="user", content=[TextBlock(text="later B")]))
    release.set()
    await manager.wait()
    recorder.finish_turn("completed")
    await sink.aclose()

    events = store.journal(session.session_id).read_events()
    settled = next(e for e in events if e.event_type == event_type and e.payload.get("request_id") == request_ids[0])
    turn_b_started = next(e for e in events if e.event_type == "turn.started" and e.turn_id == turn_b)

    assert receipt["job_id"] == settled.agent_id
    assert settled.agent_name == "general-agent"
    assert settled.depth == 1
    assert settled.turn_id == turn_a
    assert settled.seq > turn_b_started.seq
    assert settled.payload["origin"] == {"kind": "helper", "helper": "compression"}
    if event_type == LLM_REQUEST_FAILED_EVENT:
        assert settled.payload["error"] == "rate limited"


@pytest.mark.asyncio
async def test_foreground_child_helper_usage_remains_root_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env,
) -> None:
    del isolated_cli_env
    project, store, session, recorder, ledger, sink = await _harness(tmp_path)
    caller = _caller(recorder, ledger)
    connection_manager = AsyncMock()
    tool = AgentTool(
        project_path=project,
        workspace_id="w",
        thread_id="t",
        connection_manager=connection_manager,
        config=_config(),
        caller=caller,
    )

    entered = asyncio.Event()
    request_ids: list[str] = []
    monkeypatch.setattr(
        "kolega_code.agent.generalagent.GeneralAgent",
        _settling_agent_factory(
            settle_kind="response",
            entered=entered,
            request_ids=request_ids,
        ),
    )

    turn_id = recorder.start_turn(Message(role="user", content=[TextBlock(text="foreground turn")]))
    assert await tool.dispatch_general_agent("work") == "done"
    recorder.finish_turn("completed")
    await sink.aclose()

    event = next(e for e in store.journal(session.session_id).read_events() if e.event_type == LLM_MESSAGE_EVENT)
    assert request_ids == [event.payload["request_id"]]
    assert event.turn_id == turn_id
    assert event.agent_id == recorder.agent_stamp.agent_id
    assert event.agent_name == recorder.agent_stamp.agent_name
    assert event.depth == 0
