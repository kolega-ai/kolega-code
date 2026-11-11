"""Worker ownership and displayed text must survive recording checkpoints."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import create_autospec

import pytest

from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_journal import SessionJournal
from kolega_code.events import AgentConnectionManager, AgentEvent, KnownEventType
from kolega_code.services.terminal import LocalTerminalManager, TerminalEventContext, terminal_event_scope
from kolega_code.session.inmemory import InMemorySessionEventStore
from kolega_code.session.projection import replay
from kolega_code.session.recording import RecordingConnectionManager, RetentionPolicy
from kolega_code.session.store import SessionEventStore


def _recorder(
    store: SessionEventStore | None = None, *, checkpoint: int = 8_000
) -> tuple[RecordingConnectionManager, Any, SessionEventStore]:
    store = store if store is not None else InMemorySessionEventStore()
    inner = create_autospec(AgentConnectionManager, instance=True)
    manager = RecordingConnectionManager(
        inner,
        store,
        session_id="workers",
        policy=RetentionPolicy(stream_checkpoint_chars=checkpoint, stream_checkpoint_ms=10**9),
    )
    return manager, inner, store


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_type",
    [KnownEventType.ASSISTANT_DELTA, KnownEventType.TOOL_STREAMING_UPDATE, KnownEventType.TERMINAL_OUTPUT],
)
@pytest.mark.parametrize("agent_ids", [("left", "right"), ("left|nested", "left")])
async def test_same_named_workers_never_share_a_recording_checkpoint(
    event_type: str, agent_ids: tuple[str, str]
) -> None:
    manager, inner, store = _recorder()
    field = "output" if event_type == KnownEventType.TERMINAL_OUTPUT else "text"
    expected = {agent_ids[0]: "first-a first-b ", agent_ids[1]: "second-a second-b "}
    try:
        for index, text in [(0, "first-a "), (1, "second-a "), (0, "first-b "), (1, "second-b ")]:
            await manager.broadcast_event(
                AgentEvent(
                    sender="worker",
                    event_type=event_type,
                    uuid="reused-stream",
                    is_streaming=event_type == KnownEventType.ASSISTANT_DELTA,
                    sub_agent_info={"agent_id": agent_ids[index], "agent_name": "general"},
                    content={field: text, "tool_call_id": "reused-call", "terminal_id": "reused-terminal"},
                ),
                "workspace",
                "thread",
            )
    finally:
        await manager.flush()

    recorded = await store.read("workers")
    actual: defaultdict[str, str] = defaultdict(str)
    for event in recorded:
        assert event.sub_agent_info is not None
        actual[event.sub_agent_info["agent_id"]] += event.content[field]
    assert dict(actual) == expected
    assert len(recorded) == 2
    assert inner.broadcast_event.await_count == 4


@pytest.mark.asyncio
async def test_checkpoint_identity_cannot_collide_at_a_delimiter() -> None:
    manager, _, store = _recorder()
    try:
        for agent_id, call_id, output in [("a|b", "c", "left"), ("a", "b|c", "right")]:
            await manager.broadcast_event(
                AgentEvent(
                    sender="worker",
                    event_type=KnownEventType.TERMINAL_OUTPUT,
                    sub_agent_info={"agent_id": agent_id, "agent_name": "general"},
                    content={"tool_call_id": call_id, "output": output},
                ),
                "w",
                "t",
            )
    finally:
        await manager.flush()
    recorded = await store.read("workers")
    assert [(event.sub_agent_info, event.content["output"]) for event in recorded] == [
        ({"agent_id": "a|b", "agent_name": "general"}, "left"),
        ({"agent_id": "a", "agent_name": "general"}, "right"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("identifier", ["terminal_id", "session_id"])
async def test_distinct_terminal_channels_of_one_tool_call_remain_distinct_checkpoints(identifier: str) -> None:
    manager, _, store = _recorder()
    try:
        for terminal_id in ("terminal-left", "terminal-right"):
            await manager.broadcast_event(
                AgentEvent(
                    sender="worker",
                    event_type=KnownEventType.TERMINAL_OUTPUT,
                    sub_agent_info={"agent_id": "child", "agent_name": "general"},
                    content={
                        "tool_call_id": "call",
                        "terminal_id": "shared-terminal",
                        identifier: terminal_id,
                        "output": terminal_id,
                    },
                ),
                "w",
                "t",
            )
    finally:
        await manager.flush()
    recorded = await store.read("workers")
    assert [(event.content[identifier], event.content["output"]) for event in recorded] == [
        ("terminal-left", "terminal-left"),
        ("terminal-right", "terminal-right"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", [False, True])
@pytest.mark.parametrize("checkpoint", [1, 8_000])
@pytest.mark.parametrize("missing_display", [False, True])
async def test_terminal_checkpoints_preserve_both_raw_and_display_output(
    delegated: bool, checkpoint: int, missing_display: bool
) -> None:
    manager, inner, store = _recorder(checkpoint=checkpoint)
    try:
        for raw, display in [("raw-a ", "safe-a "), ("raw-b ", None if missing_display else "safe-b ")]:
            content = {"output": raw, "tool_call_id": "shell", "terminal_id": "pty"}
            if display is not None:
                content["display_output"] = display
            await manager.broadcast_event(
                AgentEvent(
                    sender="worker",
                    event_type=KnownEventType.TERMINAL_OUTPUT,
                    sub_agent_info={"agent_id": "child", "agent_name": "general"} if delegated else None,
                    content=content,
                ),
                "w",
                "t",
            )
    finally:
        await manager.flush()
    recorded = await store.read("workers")
    assert "".join(event.content["output"] for event in recorded) == "raw-a raw-b "
    assert "".join(event.content.get("display_output", event.content["output"]) for event in recorded) == (
        "safe-a raw-b " if missing_display else "safe-a safe-b "
    )
    assert inner.broadcast_event.await_count == 2


@pytest.mark.asyncio
async def test_worker_linkage_and_terminal_text_survive_a_fresh_journal_reader(tmp_path: Path) -> None:
    journal = SessionJournal("workers", tmp_path)
    journal.start_epoch("test")
    manager, _, _ = _recorder(FileSessionEventStore(journal))
    parent = {
        "agent_id": "parent",
        "agent_name": "general",
        "parent_agent_id": "root",
        "parent_tool_call_id": "dispatch-parent",
        "depth": 1,
        "background": True,
        "requested_routing": None,
        "effective_routing": {"provider": "explicit-provider", "model": "explicit-model", "thinking_effort": None},
    }
    child = {
        **parent,
        "agent_id": "nested",
        "parent_agent_id": "parent",
        "parent_tool_call_id": "dispatch-nested",
        "depth": 2,
        "background": False,
    }
    try:
        for info in (parent, child):
            await manager.broadcast_event(
                AgentEvent(
                    sender="worker",
                    event_type=KnownEventType.CHAT_MESSAGE,
                    sub_agent_info=info,
                    content={"status": "GENERATING"},
                ),
                "w",
                "t",
            )
        for info, text in [(parent, "parent-a "), (child, "child-a "), (parent, "parent-b "), (child, "child-b ")]:
            await manager.broadcast_event(
                AgentEvent(
                    sender="worker",
                    event_type=KnownEventType.TERMINAL_OUTPUT,
                    sub_agent_info=info,
                    content={"output": text, "display_output": text.upper(), "tool_call_id": "reused-shell"},
                ),
                "w",
                "t",
            )
        for info, total in [(parent, 23), (child, 0)]:
            await manager.broadcast_event(
                AgentEvent(
                    sender="worker",
                    event_type=KnownEventType.CHAT_MESSAGE,
                    sub_agent_info=info,
                    content={"status": "STOPPED", "total_tokens": total},
                ),
                "w",
                "t",
            )
    finally:
        await manager.flush()

    stored = await FileSessionEventStore(SessionJournal("workers", tmp_path)).read("workers")
    terminal_events = [event for event in stored if event.event_type == KnownEventType.TERMINAL_OUTPUT]
    assert [(event.sub_agent_info, event.content["tool_call_id"]) for event in terminal_events] == [
        (parent, "reused-shell"),
        (child, "reused-shell"),
    ]
    assert [event.content["output"] for event in terminal_events] == ["parent-a parent-b ", "child-a child-b "]
    state = replay(stored)
    assert state.terminal == ""
    assert state.conversation == []
    assert state.sub_agents["parent"].terminal == "PARENT-A PARENT-B "
    assert state.sub_agents["nested"].terminal == "CHILD-A CHILD-B "
    for key, info, total in [("parent", parent, 23), ("nested", child, 0)]:
        activity = state.to_dict()["sub_agents"][key]
        assert activity["status"] == "completed"
        assert activity["total_tokens"] == total
        for field in ("agent_id", "parent_agent_id", "parent_tool_call_id", "depth", "background", "effective_routing"):
            assert activity[field] == info[field]


@pytest.mark.asyncio
async def test_display_only_terminal_output_reaches_idle_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.session import recording

    manager, _, store = _recorder()
    monkeypatch.setattr(manager, "_ensure_ticker", lambda: None)
    await manager.broadcast_event(
        AgentEvent(
            sender="worker",
            event_type=KnownEventType.TERMINAL_OUTPUT,
            sub_agent_info={"agent_id": "child"},
            content={"output": "", "display_output": "display-only"},
        ),
        "w",
        "t",
    )
    assert await store.read("workers") == []
    monkeypatch.setattr(manager, "_elapsed_ms", lambda: 10**12)
    ticks = 0

    async def one_tick(_delay: float) -> None:
        nonlocal ticks
        ticks += 1
        if ticks > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(recording.asyncio, "sleep", one_tick)
    try:
        with pytest.raises(asyncio.CancelledError):
            await manager._checkpoint_idle_buffers()
        stored = await store.read("workers")
        assert len(stored) == 1
        assert stored[0].content == {"output": "", "display_output": "display-only"}
    finally:
        await manager.flush()


@pytest.mark.asyncio
async def test_empty_terminal_deltas_do_not_accumulate_empty_display_chunks() -> None:
    manager, _, _ = _recorder()
    try:
        for _ in range(1_024):
            await manager.broadcast_event(
                AgentEvent(
                    sender="worker",
                    event_type=KnownEventType.TERMINAL_OUTPUT,
                    sub_agent_info={"agent_id": "child"},
                    content={"output": "", "display_output": ""},
                ),
                "w",
                "t",
            )
        (state,) = manager._streams.values()
        assert state.pending == state.display_pending == []
    finally:
        await manager.flush()


@pytest.mark.asyncio
async def test_checkpoint_append_never_rescans_prior_chunks() -> None:
    from collections.abc import Iterator

    class AppendOnly(list[str]):
        def __iter__(self) -> Iterator[str]:
            raise AssertionError("Appending a delta must not scan previous chunks")

    manager, _, store = _recorder()
    state = None
    try:
        for index in range(1_025):
            await manager._record(
                AgentEvent(
                    sender="worker",
                    event_type=KnownEventType.TERMINAL_OUTPUT,
                    session_id="workers",
                    sub_agent_info={"agent_id": "child"},
                    content={"output": "r", "display_output": "d"},
                ),
            )
            if index == 0:
                (state,) = manager._streams.values()
                state.pending = AppendOnly(state.pending)
                state.display_pending = AppendOnly(state.display_pending)
    finally:
        if state is not None:
            state.pending = list(list.__iter__(state.pending))
            state.display_pending = list(list.__iter__(state.display_pending))
        await manager.flush()
    stored = await store.read("workers")
    assert len(stored) == 1
    assert stored[0].content == {"output": "r" * 1_025, "display_output": "d" * 1_025}


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True], ids=["pty", "detached"])
async def test_real_terminal_recording_replays_in_a_new_process(tmp_path: Path, background: bool) -> None:
    journal = SessionJournal("workers", tmp_path)
    journal.start_epoch("test")
    recorder, _, _ = _recorder(FileSessionEventStore(journal))
    terminals = LocalTerminalManager("w", "t", recorder, default_workdir=tmp_path)
    terminals.auto_activate_venv = False
    parent = {
        "agent_id": "parent",
        "agent_name": "general",
        "parent_agent_id": "root",
        "parent_tool_call_id": "dispatch-parent",
        "depth": 1,
        "background": True,
        "requested_routing": None,
        "effective_routing": {"provider": "explicit-provider", "model": "explicit-model", "thinking_effort": None},
    }
    nested = {
        **parent,
        "agent_id": "nested",
        "parent_agent_id": "parent",
        "parent_tool_call_id": "dispatch-nested",
        "depth": 2,
        "background": False,
    }

    async def run(info: dict[str, Any] | None, label: str) -> None:
        caller = SimpleNamespace(current_tool_execution_id="reused-shell", sub_agent_context=info)
        with terminal_event_scope(TerminalEventContext.capture(caller)):
            result = await terminals.exec_command(f"printf {label}", yield_time_ms=10_000, background=background)
        assert result.status == "exited" and result.exit_code == 0
        assert result.output == label

    try:
        for info in (parent, nested):
            await recorder.broadcast_event(
                AgentEvent(
                    event_type=KnownEventType.CHAT_MESSAGE,
                    sender="worker",
                    sub_agent_info=info,
                    content={"status": "GENERATING"},
                ),
                "w",
                "t",
            )
        await asyncio.gather(run(None, "ROOT"), run(parent, "PARENT"), run(nested, "NESTED"))
        assert not terminals.sessions
        for info, total in ((parent, 23), (nested, 0)):
            await recorder.broadcast_event(
                AgentEvent(
                    event_type=KnownEventType.CHAT_MESSAGE,
                    sender="worker",
                    sub_agent_info=info,
                    content={"status": "STOPPED", "total_tokens": total},
                ),
                "w",
                "t",
            )
    finally:
        await terminals.close_all()
        await recorder.flush()

    script = (
        "import asyncio, json, sys\n"
        "from pathlib import Path\n"
        "from kolega_code.cli.session_journal import SessionJournal\n"
        "from kolega_code.cli.session_event_store import FileSessionEventStore\n"
        "from kolega_code.session.projection import replay\n"
        "events = asyncio.run(FileSessionEventStore(SessionJournal('workers', Path(sys.argv[1]))).read('workers'))\n"
        "print(json.dumps(replay(events).to_dict()))\n"
    )
    process = await asyncio.to_thread(
        subprocess.run, [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, check=True
    )
    state = json.loads(process.stdout)
    assert state["terminal"] == "$ printf ROOT\nROOT[exited 0]\n"
    assert state["conversation"] == []
    assert set(state["sub_agents"]) == {"parent", "nested"}
    for info, label, total in ((parent, "PARENT", 23), (nested, "NESTED", 0)):
        activity = state["sub_agents"][info["agent_id"]]
        assert activity["terminal"] == f"$ printf {label}\n{label}[exited 0]\n"
        assert activity["status"] == "completed"
        assert activity["total_tokens"] == total
        assert all(activity[key] == value for key, value in info.items() if key != "agent_name")
