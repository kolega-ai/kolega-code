"""AgentTurnHandler end-to-end: scripted LLM through the real session stack.

No network: a ScriptedLLM drives ``BaseAgent`` inside the real
``SessionRuntime`` + ``SessionStore`` stack, and a recording adapter observes
the rendered output. This exercises session creation, restore, persistence,
commands, and eviction the way the gateway will run them.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import SettingsStore, CliSettings
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.gateway.adapters.base import AdapterCapabilities, ChatRef, GatewayAdapter, InboundMessage
from kolega_code.gateway.config import GatewayConfig
from kolega_code.gateway.sessions import AgentTurnHandler
from kolega_code.llm.models import Message, TextBlock
from kolega_code.llm.providers.models import TokenCount


class _TextEvent:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class ScriptedStream:
    """Yields one text event (optionally blocking on a second pull), then ends."""

    def __init__(self, text: str, *, block: asyncio.Event | None = None) -> None:
        self._text = text
        self._block = block
        self._sent = False

    async def __aenter__(self) -> "ScriptedStream":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def __aiter__(self) -> "ScriptedStream":
        return self

    async def __anext__(self) -> _TextEvent:
        if self._sent:
            if self._block is not None:
                await self._block.wait()
            raise StopAsyncIteration
        self._sent = True
        return _TextEvent(self._text)

    async def get_final_message(self) -> Message:
        return Message(role="assistant", content=[TextBlock(text=self._text)], stop_reason="end_turn")


class ScriptedLLM:
    def __init__(self, text: str = "hello from the scripted model", *, block: asyncio.Event | None = None) -> None:
        self._text = text
        self._block = block
        self.provider = MagicMock(base_url="https://api.test.invalid/v1")
        self.stream = AsyncMock(side_effect=self._stream)
        self.count_tokens = AsyncMock(return_value=TokenCount(input_tokens=10))
        self.generate = AsyncMock(
            return_value=Message(role="assistant", content=[TextBlock(text="summary")], stop_reason="end_turn")
        )

    async def _stream(self, **kwargs: Any) -> ScriptedStream:
        return ScriptedStream(self._text, block=self._block)


class RecordingAdapter(GatewayAdapter):
    name = "recording"

    def __init__(self) -> None:
        super().__init__()
        self.capabilities = AdapterCapabilities(supports_edits=True, supports_delete=True, text_chunk_limit=4096)
        self.sent: list[tuple[str, str]] = []

    async def send_text(self, chat_id: str, text: str, *, reply_to_message_id: str | None = None) -> str:
        self.sent.append((chat_id, text))
        return f"m-{len(self.sent)}"

    async def edit_text(self, chat_id: str, message_id: str, text: str) -> None:
        pass

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        pass


def make_agent_config() -> AgentConfig:
    return AgentConfig(
        anthropic_api_key="test-key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
        ),
        fast_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
        ),
    )


def make_builder(llm: ScriptedLLM, agent_config: AgentConfig):
    async def builder(record: Any, _config: Any, **kwargs: Any) -> BaseAgent:
        restore: bool = kwargs["restore"]
        agent = BaseAgent(
            project_path=Path(record.project_path),
            workspace_id=record.workspace_id,
            thread_id=record.thread_id,
            connection_manager=kwargs["manager"],
            config=agent_config,
            permission_mode=kwargs["permission_mode"],
            permission_callback=kwargs["permission_callback"],
            # Mirror the ACP recipe: a new session records from construction,
            # a resumed one hands the recorder over after restoring history.
            session_recorder=None if restore else kwargs["recorder"],
        )
        # The scripted LLM is not an LLMClient and BaseAgent builds no tool
        # collection; stub both the way the repo's own agent tests do.
        setattr(agent, "llm", llm)
        setattr(agent, "tool_collection", MagicMock())
        setattr(agent.tool_collection, "get_tool_list", MagicMock(return_value=[]))
        setattr(agent.tool_collection, "cleanup", AsyncMock())
        if restore:
            agent.restore_message_history(record.history)
            agent.restore_compaction_state(record.compaction)
            agent.session_recorder = kwargs["recorder"]
        return agent

    return builder


def make_handler(
    tmp_path: Path,
    *,
    llm: ScriptedLLM | None = None,
    adapter: RecordingAdapter | None = None,
    max_sessions: int = 4,
    store: SessionStore | None = None,
) -> tuple[AgentTurnHandler, RecordingAdapter, GatewayConfig]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    state_dir = tmp_path / "state"
    config = GatewayConfig(
        adapter="recording",
        project_path=workspace,
        state_dir=state_dir,
        permission_mode="ask",
        edit_throttle_seconds=0.0,
        max_sessions=max_sessions,
        session_idle_ttl_seconds=None,
    )
    settings_store = SettingsStore(root=state_dir)
    settings_store.save(
        CliSettings(
            active_provider="anthropic",
            active_model="claude-haiku-4-5-20251001",
            api_keys={"anthropic": "test-key"},
        )
    )
    adapter = adapter or RecordingAdapter()
    handler = AgentTurnHandler(
        config=config,
        adapter=adapter,
        store=store or SessionStore(root=state_dir),
        settings_store=settings_store,
        agent_builder=make_builder(llm or ScriptedLLM(), make_agent_config()),
    )
    return handler, adapter, config


def inbound(text: str, message_id: str = "m-1", chat_id: str = "42") -> InboundMessage:
    return InboundMessage(channel="recording", chat_id=chat_id, sender_id="7", message_id=message_id, text=text)


def chat_ref(chat_id: str = "42") -> ChatRef:
    return ChatRef("recording", chat_id)


def all_sent_texts(adapter: RecordingAdapter) -> str:
    return "\n".join(text for _, text in adapter.sent)


async def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


def session_map(state_dir: Path) -> dict[str, str]:
    return json.loads((state_dir / "gateway_sessions.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_turn_streams_and_persists(tmp_path: Path) -> None:
    handler, adapter, config = make_handler(tmp_path)
    await handler.handle(chat_ref(), inbound("hello", message_id="m-1"))
    assert await wait_for(lambda: "hello from the scripted model" in all_sent_texts(adapter))
    await handler.shutdown()

    # The turn is persisted through the journal and the session record.
    session_id = session_map(config.state_dir)[chat_ref().key]
    record = SessionStore(root=config.state_dir).load(session_id)
    history_texts = [block.get("text", "") for message in record.history for block in message.get("content", [])]
    assert "hello" in history_texts
    assert "hello from the scripted model" in history_texts


@pytest.mark.asyncio
async def test_resumed_session_continues_across_handler_instances(tmp_path: Path) -> None:
    # One store models one daemon's state dir; a handler "restart" in the same
    # process must release the session locks the way a process exit would.
    store = SessionStore(root=tmp_path / "state")
    first, adapter, config = make_handler(tmp_path, store=store)
    await first.handle(chat_ref(), inbound("first", message_id="m-1"))
    assert await wait_for(lambda: "hello from the scripted model" in all_sent_texts(adapter))
    await first.shutdown()
    store.release_session_locks()

    session_id = session_map(config.state_dir)[chat_ref().key]
    second, adapter2, _ = make_handler(tmp_path, store=store)
    await second.handle(chat_ref(), inbound("second", message_id="m-2"))
    assert await wait_for(lambda: len(adapter2.sent) >= 1)
    await second.shutdown()

    # Same session both times; the second turn appended to its history.
    assert session_map(config.state_dir)[chat_ref().key] == session_id
    record = store.load(session_id)
    history_texts = [block.get("text", "") for message in record.history for block in message.get("content", [])]
    assert "first" in history_texts
    assert "second" in history_texts


@pytest.mark.asyncio
async def test_status_command_reports_session_and_model(tmp_path: Path) -> None:
    handler, adapter, _ = make_handler(tmp_path)
    await handler.handle(chat_ref(), inbound("hello", message_id="m-1"))
    assert await wait_for(lambda: "hello from the scripted model" in all_sent_texts(adapter))
    await handler.handle(chat_ref(), inbound("/status", message_id="m-2"))
    assert await wait_for(lambda: "Session" in all_sent_texts(adapter))
    assert "anthropic/claude-haiku-4-5-20251001" in all_sent_texts(adapter)
    await handler.shutdown()


@pytest.mark.asyncio
async def test_new_command_starts_a_fresh_session(tmp_path: Path) -> None:
    handler, adapter, config = make_handler(tmp_path)
    await handler.handle(chat_ref(), inbound("hello", message_id="m-1"))
    assert await wait_for(lambda: "hello from the scripted model" in all_sent_texts(adapter))
    old_session_id = session_map(config.state_dir)[chat_ref().key]

    await handler.handle(chat_ref(), inbound("/new", message_id="m-2"))
    assert await wait_for(lambda: "New session started" in all_sent_texts(adapter))
    await handler.handle(chat_ref(), inbound("fresh", message_id="m-3"))
    # "fresh" is the user message — the turn's arrival shows as a second reply.
    assert await wait_for(lambda: all_sent_texts(adapter).count("hello from the scripted model") >= 2)

    new_session_id = session_map(config.state_dir)[chat_ref().key]
    assert new_session_id != old_session_id
    record = SessionStore(root=config.state_dir).load(new_session_id)
    history_texts = [block.get("text", "") for message in record.history for block in message.get("content", [])]
    assert "fresh" in history_texts
    assert "hello" not in history_texts
    await handler.shutdown()


@pytest.mark.asyncio
async def test_stop_cancels_a_blocked_turn_and_the_worker_continues(tmp_path: Path) -> None:
    blocker = asyncio.Event()
    handler, adapter, _ = make_handler(
        tmp_path, llm=ScriptedLLM(text="long answer from the scripted model", block=blocker)
    )
    await handler.handle(chat_ref(), inbound("long task", message_id="m-1"))
    assert await wait_for(lambda: "long answer from the scripted model" in all_sent_texts(adapter))

    await handler.handle(chat_ref(), inbound("/stop", message_id="m-2"))
    assert await wait_for(lambda: "Stopping the current turn" in all_sent_texts(adapter))

    # The worker survives the cancelled turn and serves the next message.
    await handler.handle(chat_ref(), inbound("quick one", message_id="m-3"))
    assert await wait_for(lambda: all_sent_texts(adapter).count("long answer from the scripted model") >= 2)
    await handler.shutdown()


@pytest.mark.asyncio
async def test_unknown_command_gets_a_hint(tmp_path: Path) -> None:
    handler, adapter, _ = make_handler(tmp_path)
    await handler.handle(chat_ref(), inbound("/frobnicate", message_id="m-1"))
    assert await wait_for(lambda: "Unknown command" in all_sent_texts(adapter))
    await handler.shutdown()


@pytest.mark.asyncio
async def test_session_cap_evicts_the_oldest_chat(tmp_path: Path) -> None:
    handler, adapter, config = make_handler(tmp_path, max_sessions=1)
    await handler.handle(chat_ref("1"), inbound("one", message_id="m-1", chat_id="1"))
    assert await wait_for(lambda: "hello from the scripted model" in all_sent_texts(adapter))

    # Opening a second chat evicts the first (persisted, then cleaned up).
    await handler.handle(chat_ref("2"), inbound("two", message_id="m-2", chat_id="2"))
    assert await wait_for(lambda: len(handler.status().keys()) >= 0)
    assert handler.status()["active_sessions"] == 1
    first_session_id = session_map(config.state_dir)[chat_ref("1").key]
    record = SessionStore(root=config.state_dir).load(first_session_id)
    assert record.history  # still durable on disk
    await handler.shutdown()


@pytest.mark.asyncio
async def test_help_command_lists_commands(tmp_path: Path) -> None:
    handler, adapter, _ = make_handler(tmp_path)
    await handler.handle(chat_ref(), inbound("/help", message_id="m-1"))
    assert await wait_for(lambda: "/new" in all_sent_texts(adapter))
    assert "/status" in all_sent_texts(adapter)
    await handler.shutdown()
