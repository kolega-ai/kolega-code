from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from filelock import FileLock

from kolega_code.cli.session_event_protocol import InMemorySessionJournal
from kolega_code.cli.session_journal import SessionEvent, SessionJournal, SessionJournalError, SessionRecorder
from kolega_code.cli.session_store import SessionStore
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest.fixture(params=["file", "memory", "owned"])
def journal(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SessionJournal]:
    store = None
    if request.param == "memory":
        result = InMemorySessionJournal("recovery")
        result.start_epoch("test")
    elif request.param == "owned":
        project = tmp_path / "project"
        project.mkdir()
        store = SessionStore(tmp_path / "state")
        record = store.create(project, "code", {})
        result = store.recorder(record.session_id).journal
    else:
        result = SessionJournal("recovery", tmp_path / "session")
        result.start_epoch("test")
    try:
        yield result
    finally:
        if store is not None:
            store.release_session_lock(result.session_id)


def _count_scans(journal: SessionJournal, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counts = {"reads": 0, "bytes": 0, "parses": 0, "loads": 0}
    read_bytes = journal._read_event_bytes_locked
    parse_bytes = journal._parse_event_bytes_locked
    read_events = journal._read_events_locked

    def read(*, repair_tail: bool) -> bytes:
        raw = read_bytes(repair_tail=repair_tail)
        counts["reads"] += 1
        counts["bytes"] += len(raw)
        return raw

    def parse(raw: bytes, *, strict_seq: bool = True) -> list[SessionEvent]:
        counts["parses"] += 1
        return parse_bytes(raw, strict_seq=strict_seq)

    def load(*, repair_tail: bool, **kwargs: Any) -> list[SessionEvent]:
        counts["loads"] += 1
        return read_events(repair_tail=repair_tail, **kwargs)

    monkeypatch.setattr(journal, "_read_event_bytes_locked", read)
    monkeypatch.setattr(journal, "_parse_event_bytes_locked", parse)
    monkeypatch.setattr(journal, "_read_events_locked", load)
    return counts


def _assistant(recorder: SessionRecorder, execution_id: str = "execution") -> ToolCall:
    call = ToolCall(id="provider-id", name="read", input={}, execution_id=execution_id)
    recorder.record_assistant(Message(role="assistant", content=[call], tool_calls=[call]))
    return call


def _answer(recorder: SessionRecorder, call: ToolCall) -> None:
    recorder.record_tool_results(
        [
            ToolResult(
                tool_use_id=call.id,
                name=call.name,
                execution_id=call.execution_id,
                content="original successful output",
                is_error=False,
            )
        ]
    )


def test_warm_recovery_does_not_scan_history_or_answered_tool_cycles(
    journal: SessionJournal, monkeypatch: pytest.MonkeyPatch
) -> None:
    for _ in range(2_000):
        journal.append("ui.assistant_delta", actor="assistant", payload={"text": "x" * 1024})
    recorder = SessionRecorder(journal, recover=False)
    assert recorder.repair_interrupted_tool_results() is None
    counts = _count_scans(journal, monkeypatch)
    recorder.start_turn(Message(role="user", content=[TextBlock("work")]))

    for iteration in range(5):
        call = _assistant(recorder, f"execution-{iteration}")
        _answer(recorder, call)
        recorder.record_context_message(Message(role="user", content=[TextBlock("continue")]))
        journal.append("ui.assistant_delta", actor="assistant", payload={"text": "next"})
        assert recorder.repair_interrupted_tool_results() is None
        assert SessionRecorder(journal, recover=False).repair_interrupted_tool_results() is None

    assert counts == {"reads": 0, "bytes": 0, "parses": 0, "loads": 0}


@pytest.mark.parametrize("owned", [False, True])
def test_cold_recovery_validates_only_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owned: bool) -> None:
    project = tmp_path / "project"
    project.mkdir()
    original = SessionStore(tmp_path / "state")
    record = original.create(project, "code", {})
    original.journal(record.session_id).append("ui.assistant_delta", actor="assistant", payload={"text": "x" * 1024})
    if owned:
        store = SessionStore(tmp_path / "state")
        store._acquire_session_lock(record.session_id)
        journal = store.journal(record.session_id)
    else:
        store = None
        journal = SessionJournal(record.session_id, original.session_dir_for(record.session_id))
    try:
        counts = _count_scans(journal, monkeypatch)
        recorder = SessionRecorder(journal, recover=False)
        for _ in range(5):
            assert recorder.repair_interrupted_tool_results() is None
        assert counts["reads"] == counts["parses"] == counts["loads"] == 1
        assert counts["bytes"] == journal.events_path.stat().st_size
    finally:
        if store is not None:
            store.release_session_lock(record.session_id)


@pytest.mark.parametrize("closed", [False, True])
def test_direct_semantic_appends_are_visible_to_replacement_recorders(journal: SessionJournal, closed: bool) -> None:
    journal.append("turn.started", actor="user", turn_id="direct-turn")
    payload = {
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "id": "freeform-call",
                    "name": "apply_patch",
                    "execution_id": "freeform-execution",
                    "input_kind": "freeform",
                    "input": "patch body",
                }
            ],
        }
    }
    journal.append("assistant.message", actor="assistant", turn_id="direct-turn", payload=payload)
    payload["message"]["content"][0]["id"] = "mutated-after-persistence"
    if closed:
        journal.append("turn.failed", actor="system", turn_id="direct-turn")
    recorder = SessionRecorder(journal, recover=False)
    repaired = recorder.repair_interrupted_tool_results(defer_projection=True)
    assert repaired is not None and len(repaired) == 1
    assert repaired[0].tool_use_id == "freeform-call"
    assert repaired[0].execution_id == "freeform-execution"
    assert repaired[0].name == "apply_patch"
    assert repaired[0].input_kind == "freeform"
    assert repaired[0].is_error
    assert "was not re-run" in str(repaired[0].content)
    assert recorder.current_turn_id is None
    recorder.acknowledge_tool_results()
    assert recorder.repair_interrupted_tool_results() is None


@pytest.mark.parametrize(
    ("call_execution", "result_execution", "result_provider", "missing"),
    [
        ("execution", "execution", "different-provider", False),
        ("execution", "different-execution", "provider-id", True),
        ("execution", None, "provider-id", False),
        (None, "execution", "provider-id", False),
        (None, None, "provider-id", False),
        ("execution", None, "different-provider", True),
    ],
)
def test_result_identity_matching_includes_results_before_latest_assistant(
    journal: SessionJournal,
    call_execution: str | None,
    result_execution: str | None,
    result_provider: str,
    missing: bool,
) -> None:
    journal.append("turn.started", actor="user", turn_id="turn")
    journal.append(
        "tool.results",
        actor="tool",
        turn_id="turn",
        payload={
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": result_provider, "execution_id": result_execution}]
            }
        },
    )
    journal.append(
        "assistant.message",
        actor="assistant",
        turn_id="turn",
        payload={
            "message": {
                "content": [
                    {
                        "type": "tool_call",
                        "id": "provider-id",
                        "name": "read",
                        "execution_id": call_execution,
                    }
                ]
            }
        },
    )
    recorder = SessionRecorder(journal, recover=False)
    repaired = recorder.repair_interrupted_tool_results()
    if missing:
        assert repaired is not None and len(repaired) == 1
        assert repaired[0].tool_use_id == "provider-id"
    else:
        assert repaired is None


def test_child_and_other_epoch_events_cannot_answer_or_replace_root_tools(journal: SessionJournal) -> None:
    root = SessionRecorder(journal, recover=False)
    turn_id = root.start_turn(Message(role="user", content=[TextBlock("root")]))
    call = _assistant(root)
    child = root.scoped_child(agent_id="child", agent_name="child", parent_tool_call_id=call.id, depth=1)
    child.start_turn(Message(role="user", content=[TextBlock("child")]))
    _answer(child, call)
    child.record_assistant(Message(role="assistant", content=[TextBlock("child finished")]))
    journal.append(
        "assistant.message",
        actor="assistant",
        epoch_id="inactive-epoch",
        turn_id=turn_id,
        payload={"message": {"content": [{"type": "text", "text": "not the root response"}]}},
    )
    repaired = root.repair_interrupted_tool_results()
    assert repaired is not None and [result.execution_id for result in repaired] == ["execution"]
    assert root.current_turn_id == turn_id


@pytest.mark.parametrize("discard_all", [False, True])
@pytest.mark.parametrize("ambiguous", [False, True])
def test_rewind_rebuilds_only_latest_surviving_turn(
    journal: SessionJournal, discard_all: bool, ambiguous: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = SessionRecorder(journal, recover=False)
    first = recorder.start_turn(Message(role="user", content=[TextBlock("first")]))
    _assistant(recorder, "first-execution")
    recorder.finish_turn("failed")
    second = recorder.start_turn(Message(role="user", content=[TextBlock("second")]))
    _answer(recorder, _assistant(recorder, "second-execution"))
    recorder.finish_turn("completed")
    assert recorder.repair_interrupted_tool_results() is None

    original_write = journal._write_event_locked

    def ambiguous_write(event: SessionEvent) -> None:
        original_write(event)
        if event.event_type == "context.rewound":
            raise SessionJournalError("ambiguous rewind")

    with monkeypatch.context() as faults:
        if ambiguous:
            faults.setattr(journal, "_write_event_locked", ambiguous_write)
            with pytest.raises(SessionJournalError, match="ambiguous rewind"):
                recorder.record_rewind(first if discard_all else second)
        else:
            recorder.record_rewind(first if discard_all else second)
    repaired = recorder.repair_interrupted_tool_results()
    if discard_all:
        assert repaired is None
    else:
        assert repaired is not None and [result.execution_id for result in repaired] == ["first-execution"]
    assert recorder.repair_interrupted_tool_results() is None


def test_new_epoch_drops_old_recovery_state(journal: SessionJournal, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = SessionRecorder(journal, recover=False)
    recorder.start_turn(Message(role="user", content=[TextBlock("first")]))
    _assistant(recorder)
    recorder.finish_turn("failed")
    counts = _count_scans(journal, monkeypatch)
    recorder.start_epoch("reset")
    assert recorder.repair_interrupted_tool_results() is None
    recorder.start_continuation_turn()
    _answer(recorder, _assistant(recorder, "new-execution"))
    assert recorder.repair_interrupted_tool_results() is None
    assert counts == {"reads": 0, "bytes": 0, "parses": 0, "loads": 0}


@pytest.mark.parametrize("assistant", ["text", "absent", "invalid", "idless-tool"])
def test_startup_recovery_preserves_terminal_decision(journal: SessionJournal, assistant: str) -> None:
    journal.append("turn.started", actor="user", turn_id="turn")
    if assistant != "absent":
        content = [{"type": "tool_call"}] if assistant == "idless-tool" else [{"type": "text", "text": "done"}]
        journal.append(
            "assistant.message",
            actor="assistant",
            turn_id="turn",
            payload={"message": "invalid" if assistant == "invalid" else {"role": "assistant", "content": content}},
        )
    recovery = SessionRecorder(journal, recover=False)
    assert recovery.recover_interrupted_turn()
    assert recovery.current_turn_id is None
    terminal = journal.read_events()[-1]
    assert terminal.event_type == ("turn.completed" if assistant == "text" else "turn.failed")
    assert recovery.recover_interrupted_turn() is False


def test_recovery_snapshot_is_immutable_and_retains_no_payload_bodies(journal: SessionJournal) -> None:
    recorder = SessionRecorder(journal, recover=False)
    recorder.start_turn(Message(role="user", content=[TextBlock("large input body " * 10_000)]))
    call = ToolCall(id="call", name="read", input={"large": "tool input body " * 10_000}, execution_id="execution")
    recorder.record_assistant(
        Message(role="assistant", content=[TextBlock("assistant prose " * 10_000), call], tool_calls=[call])
    )
    snapshot = journal._root_recovery_snapshot()
    assert snapshot is not None
    recorder.record_tool_results(
        [
            ToolResult(
                tool_use_id=call.id,
                name=call.name,
                execution_id=call.execution_id,
                content="tool output body " * 20_000,
                is_error=False,
            )
        ]
    )
    journal.append("ui.assistant_delta", actor="assistant", payload={"text": "presentation body " * 10_000})
    assert [item.tool_use_id for item in snapshot.missing] == ["call"]
    with pytest.raises(FrozenInstanceError):
        setattr(snapshot, "status", "changed")
    with pytest.raises(FrozenInstanceError):
        setattr(snapshot.missing[0], "name", "changed")
    retained = repr(journal._root_turn_recovery)
    for text in ("large input body", "tool input body", "assistant prose", "tool output body", "presentation body"):
        assert text not in retained
    assert recorder.repair_interrupted_tool_results() is None


@pytest.mark.parametrize("operation", ["read", "raw"])
def test_failed_validation_cannot_leave_trusted_recovery(tmp_path: Path, operation: str) -> None:
    journal = SessionJournal("recovery", tmp_path)
    journal.start_epoch("test")
    recorder = SessionRecorder(journal, recover=False)
    assert recorder.repair_interrupted_tool_results() is None
    with journal.events_path.open("ab") as stream:
        stream.write(b"invalid json\n")
    with pytest.raises(SessionJournalError, match="Invalid JSON"):
        journal.read_events() if operation == "read" else journal.raw_events()
    with pytest.raises(SessionJournalError, match="Invalid JSON"):
        recorder.repair_interrupted_tool_results()


@pytest.mark.parametrize("failure", ["before", "after", "interrupt", "close"])
def test_uncertain_direct_assistant_append_reloads_durable_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    journal = SessionJournal("recovery", tmp_path)
    journal.start_epoch("test")
    journal.append("turn.started", actor="user", turn_id="turn")
    original_write = journal._write_event_locked
    original_close = os.close
    failed_close = False

    def write(event: SessionEvent) -> None:
        if failure == "before":
            raise SessionJournalError("injected failure")
        original_write(event)
        if failure == "interrupt":
            raise KeyboardInterrupt("injected failure")
        if failure == "after":
            raise SessionJournalError("injected failure")

    def close(fd: int) -> None:
        nonlocal failed_close
        original_close(fd)
        if not failed_close:
            failed_close = True
            raise OSError("injected failure")

    with monkeypatch.context() as fault:
        fault.setattr(journal, "_write_event_locked", write)
        if failure == "close":
            fault.setattr(os, "close", close)
        error = KeyboardInterrupt if failure == "interrupt" else OSError if failure == "close" else SessionJournalError
        with pytest.raises(error, match="injected failure"):
            journal.append(
                "assistant.message",
                actor="assistant",
                turn_id="turn",
                payload={
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "tool_call", "id": "call", "name": "read", "input": {}}],
                    }
                },
            )
    recorder = SessionRecorder(journal, recover=False)
    repaired = recorder.repair_interrupted_tool_results()
    if failure == "before":
        assert repaired is None
    else:
        assert repaired is not None and [result.tool_use_id for result in repaired] == ["call"]
    assert recorder.repair_interrupted_tool_results() is None


@pytest.mark.parametrize("version", [1, 2])
def test_cold_recovery_handles_legacy_lineage_and_tool_ids(tmp_path: Path, version: int) -> None:
    original = SessionJournal("recovery", tmp_path)
    original.start_epoch("test")
    original.append("turn.started", actor="user", turn_id="turn")
    original.append(
        "assistant.message",
        actor="assistant",
        turn_id="turn",
        payload={"message": {"content": [{"type": "tool_call", "id": "legacy", "name": "read"}]}},
    )
    records = [event.to_dict() for event in original.read_events()]
    for record in records:
        record["version"] = version
        if version == 1:
            for key in ("schema", "agent_id", "agent_name", "parent_agent_id", "parent_tool_call_id", "depth"):
                record.pop(key, None)
    original.events_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    recorder = SessionRecorder(SessionJournal("recovery", tmp_path), recover=False)
    repaired = recorder.repair_interrupted_tool_results()
    assert repaired is not None and len(repaired) == 1
    assert repaired[0].tool_use_id == "legacy" and repaired[0].execution_id is None
    assert recorder.repair_interrupted_tool_results() is None


def test_non_owner_refreshes_after_external_appends_and_ownership_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / ".lock"
    owns = False
    journal = SessionJournal(
        "recovery",
        tmp_path / "session",
        cross_process_lock=FileLock(lock_path, timeout=0, thread_local=False),
        owns_cross_process_lock=lambda: owns,
    )
    journal.start_epoch("test")
    recorder = SessionRecorder(journal, recover=False)
    assert recorder.repair_interrupted_tool_results() is None
    writer = SessionJournal(
        "recovery",
        journal.session_dir,
        cross_process_lock=FileLock(lock_path, timeout=0, thread_local=False),
    )
    writer.append("turn.started", actor="user", turn_id="external")
    writer.append(
        "assistant.message",
        actor="assistant",
        turn_id="external",
        payload={"message": {"content": [{"type": "tool_call", "id": "external-call", "name": "read"}]}},
    )
    journal.append("ui.assistant_delta", actor="assistant", payload={"text": "resync"})
    assert journal._recovery_valid is False
    assert journal._cross_process_lock is not None
    with journal._cross_process_lock:
        owns = True
        repaired = recorder.repair_interrupted_tool_results()
        assert repaired is not None and [result.tool_use_id for result in repaired] == ["external-call"]
        counts = _count_scans(journal, monkeypatch)
        assert recorder.repair_interrupted_tool_results() is None
        assert counts["loads"] == 0


@pytest.mark.parametrize("read_prefix", [False, True])
def test_partial_result_with_failed_rollback_does_not_poison_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_prefix: bool
) -> None:
    journal = SessionJournal("recovery", tmp_path)
    journal.start_epoch("test")
    recorder = SessionRecorder(journal, recover=False)
    recorder.start_turn(Message(role="user", content=[TextBlock("work")]))
    call = _assistant(recorder)
    write = os.write

    def partial(fd: int, data: bytes) -> int:
        return write(fd, data[:27])

    def fail_rollback(fd: int, length: int) -> None:
        raise OSError("rollback unavailable")

    with monkeypatch.context() as faults:
        faults.setattr(os, "write", partial)
        faults.setattr(os, "ftruncate", fail_rollback)
        with pytest.raises(SessionJournalError, match="Short event write"):
            journal.append(
                "tool.results",
                actor="tool",
                turn_id=recorder.current_turn_id,
                payload={"message": {"content": [ToolResult(call.id, "not committed", call.name, False).to_dict()]}},
            )
    if read_prefix:
        journal.read_events(repair_tail=False)
        assert journal.tail_requires_recovery
    repaired = recorder.repair_interrupted_tool_results()
    assert repaired is not None and [result.execution_id for result in repaired] == [call.execution_id]
    assert journal.events_path.read_bytes().endswith(b"\n")
    assert [event.seq for event in journal.read_events()] == [1, 2, 3, 4]
    assert recorder.repair_interrupted_tool_results() is None


@pytest.mark.parametrize("operation", ["read", "raw"])
def test_explicit_reads_refresh_recovery_from_validated_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    journal = SessionJournal("recovery", tmp_path)
    journal.start_epoch("test")
    recorder = SessionRecorder(journal, recover=False)
    assert recorder.repair_interrupted_tool_results() is None
    writer = SessionJournal("recovery", tmp_path)
    writer.append("turn.started", actor="user", turn_id="external")
    writer.append(
        "assistant.message",
        actor="assistant",
        turn_id="external",
        payload={"message": {"content": [{"type": "tool_call", "id": "external-call", "name": "read"}]}},
    )
    counts = _count_scans(journal, monkeypatch)
    journal.read_events() if operation == "read" else journal.raw_events()
    assert counts["reads"] == counts["parses"] == 1
    snapshot = journal._root_recovery_snapshot()
    assert snapshot is not None and [item.tool_use_id for item in snapshot.missing] == ["external-call"]
    assert counts["reads"] == counts["parses"] == 1


def test_sequence_repair_rebuilds_from_canonical_boundaries(tmp_path: Path) -> None:
    journal = SessionJournal("recovery", tmp_path)
    journal.start_epoch("test")
    recorder = SessionRecorder(journal, recover=False)
    recorder.start_turn(Message(role="user", content=[TextBlock("work")]))
    _assistant(recorder)
    records = [event.to_dict() for event in journal.read_events()]
    records[1]["seq"] = 22
    records[2]["seq"] = 33
    journal.events_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert journal.repair_sequence() == 3
    assert journal._root_turn_recovery is not None and journal._root_turn_recovery.boundary_seq == 2
    assert journal.append("ui.assistant_delta", actor="assistant").seq == 4
    repaired = recorder.repair_interrupted_tool_results()
    assert repaired is not None and [result.execution_id for result in repaired] == ["execution"]
    assert [event.seq for event in journal.read_events()] == [1, 2, 3, 4, 5]
    assert recorder.repair_interrupted_tool_results() is None


def test_recovery_requires_an_epoch() -> None:
    journal = InMemorySessionJournal("no-epoch")
    with pytest.raises(SessionJournalError, match="no context epoch"):
        SessionRecorder(journal, recover=False).repair_interrupted_tool_results()


@pytest.mark.parametrize("operation", ["recover", "append"])
def test_reacquired_ownership_revalidates_without_an_intervening_operation(tmp_path: Path, operation: str) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    recorder = store.recorder(record.session_id, recover=False)
    journal = recorder.journal
    assert recorder.repair_interrupted_tool_results() is None
    store.release_session_lock(record.session_id)
    writer_store = SessionStore(tmp_path / "state")
    writer_store._acquire_session_lock(record.session_id)
    writer = writer_store.journal(record.session_id)
    try:
        epoch_id = writer.start_epoch("other owner")
        writer.append("turn.started", actor="user", turn_id="external")
        writer.append(
            "assistant.message",
            actor="assistant",
            turn_id="external",
            payload={"message": {"content": [{"type": "tool_call", "id": "external-call", "name": "read"}]}},
        )
    finally:
        writer_store.release_session_lock(record.session_id)
    expected_next_seq = len(journal.events_path.read_bytes().splitlines()) + 1
    store._acquire_session_lock(record.session_id)
    try:
        if operation == "recover":
            repaired = recorder.repair_interrupted_tool_results()
            assert repaired is not None and [result.tool_use_id for result in repaired] == ["external-call"]
        else:
            appended = journal.append("ui.assistant_delta", actor="assistant", payload={"text": "resumed"})
            assert appended.seq == expected_next_seq
        assert journal.epoch_id == epoch_id
        events = journal.read_events()
        assert [event.seq for event in events] == list(range(1, len(events) + 1))
    finally:
        store.release_session_lock(record.session_id)


@pytest.mark.parametrize("phase", ["before", "after"])
def test_interrupted_projection_advancement_invalidates_durable_state(
    journal: SessionJournal, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    journal.append("turn.started", actor="user", turn_id="turn")
    advance = journal._advance_root_recovery_locked

    def interrupt(event: SessionEvent) -> None:
        if phase == "after":
            advance(event)
        raise KeyboardInterrupt("interrupted projection")

    with monkeypatch.context() as fault:
        fault.setattr(journal, "_advance_root_recovery_locked", interrupt)
        with pytest.raises(KeyboardInterrupt, match="interrupted projection"):
            journal.append(
                "assistant.message",
                actor="assistant",
                turn_id="turn",
                payload={"message": {"content": [{"type": "tool_call", "id": "call", "name": "read"}]}},
            )
    assert not journal._loaded and not journal._recovery_valid
    repaired = SessionRecorder(journal, recover=False).repair_interrupted_tool_results()
    assert repaired is not None and [result.tool_use_id for result in repaired] == ["call"]


@pytest.mark.parametrize("operation", ["read", "raw"])
def test_epoch_reset_cannot_hide_failed_validation(tmp_path: Path, operation: str) -> None:
    journal = SessionJournal("recovery", tmp_path)
    journal.start_epoch("test")
    with journal.events_path.open("ab") as stream:
        stream.write(b"invalid json\n")
    with pytest.raises(SessionJournalError, match="Invalid JSON"):
        journal.read_events() if operation == "read" else journal.raw_events()
    with pytest.raises(SessionJournalError, match="Invalid JSON"):
        journal.start_epoch("reset")
    with pytest.raises(SessionJournalError, match="Invalid JSON"):
        SessionRecorder(journal, recover=False).repair_interrupted_tool_results()


@pytest.mark.parametrize("event_type", ["assistant.message", "tool.results"])
def test_tuple_content_matches_persisted_json_arrays(journal: SessionJournal, event_type: str) -> None:
    recorder = SessionRecorder(journal, recover=False)
    turn_id = recorder.start_turn(Message(role="user", content=[TextBlock("work")]))
    call = _assistant(recorder)
    block = (
        ToolCall(id="next", name="read", input={}, execution_id="next-execution").to_dict()
        if event_type == "assistant.message"
        else ToolResult(call.id, "successful output", call.name, False, execution_id=call.execution_id).to_dict()
    )
    journal.append(
        event_type,
        actor="assistant" if event_type == "assistant.message" else "tool",
        turn_id=turn_id,
        payload={"message": {"content": (block,)}},
    )
    snapshot = journal._root_recovery_snapshot()
    assert snapshot is not None
    assert [item.tool_use_id for item in snapshot.missing] == (["next"] if event_type == "assistant.message" else [])
    journal.read_events()
    assert journal._root_recovery_snapshot() == snapshot
