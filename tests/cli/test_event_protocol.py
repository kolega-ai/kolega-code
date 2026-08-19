"""Semantic event protocol v2: envelope, public projection, sinks, lineage."""

import json
from io import StringIO
from pathlib import Path

import pytest

from kolega_code.cli.session_event_protocol import (
    InMemorySessionJournal,
    SemanticStdoutPrinter,
    to_public_event,
)
from kolega_code.cli.session_journal import (
    DEFAULT_ROOT_AGENT_NAME,
    EVENT_SCHEMA_VERSION,
    SessionEvent,
    SessionJournal,
    SessionJournalError,
    SessionRecorder,
    collect_epoch_turns,
    derive_root_agent_id,
)
from kolega_code.llm.models import (
    ContentBlock,
    Message,
    RedactedThinkingBlock,
    ResponsesReasoningBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
)

ENVELOPE_KEYS = {
    "schema",
    "version",
    "id",
    "session_id",
    "seq",
    "epoch_id",
    "turn_id",
    "timestamp",
    "actor",
    "agent_id",
    "agent_name",
    "parent_agent_id",
    "parent_tool_call_id",
    "depth",
    "type",
    "payload",
    "artifacts",
}


def make_file_journal(tmp_path: Path, session_id: str = "s1") -> SessionJournal:
    return SessionJournal(session_id, tmp_path / "session")


def bootstrap(journal: SessionJournal) -> None:
    journal.append("session.created", actor="system", payload={"metadata": {}}, epoch_id="e1")
    journal.append("context.epoch_started", actor="system", payload={"reason": "session_created"}, epoch_id="e1")


def drive_turn(recorder: SessionRecorder) -> None:
    recorder.start_turn(Message(role="user", content=[TextBlock("hello")]))
    recorder.record_assistant(Message(role="assistant", content=[TextBlock("hi")], stop_reason="end_turn"))
    recorder.finish_turn("completed")


def test_every_public_event_has_the_complete_v2_envelope(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    recorder = SessionRecorder(journal, recover=False)
    drive_turn(recorder)

    for event in journal.read_events():
        public = to_public_event(event)
        assert public is not None
        assert set(public) == ENVELOPE_KEYS
        assert public["schema"] == "kolega.session.event"
        assert public["version"] == 2
        assert public["seq"] >= 1
        assert public["agent_id"] == derive_root_agent_id("s1")
        assert public["agent_name"] == DEFAULT_ROOT_AGENT_NAME
        assert public["depth"] == 0
        assert public["parent_agent_id"] is None
        assert public["parent_tool_call_id"] is None


def test_v1_events_project_with_derived_root_identity() -> None:
    event = SessionEvent(
        version=1,
        event_id="ev-1",
        session_id="legacy",
        seq=1,
        epoch_id="e1",
        turn_id=None,
        timestamp="2026-01-01T00:00:00+00:00",
        actor="system",
        event_type="session.created",
        payload={"metadata": {}},
        artifacts=[],
    )
    public = to_public_event(event)
    assert public is not None
    assert set(public) == ENVELOPE_KEYS
    assert public["agent_id"] == derive_root_agent_id("legacy")
    assert public["agent_name"] == DEFAULT_ROOT_AGENT_NAME
    assert public["depth"] == 0


def test_ui_events_are_excluded_from_the_public_projection(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    event = journal.append("ui.chat_message", actor="agent", payload={"event": {"content": "x"}})
    assert to_public_event(event) is None


def test_opaque_provider_state_is_stripped_and_readable_reasoning_kept(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    recorder = SessionRecorder(journal, recover=False)
    recorder.start_turn(Message(role="user", content=[TextBlock("q")]))
    blocks_in: list[ContentBlock] = [
        ThinkingBlock("visible thinking", signature="OPAQUE-SIGNATURE"),
        RedactedThinkingBlock(data="T1BBUVVF"),
        ResponsesReasoningBlock(content=["readable"], encrypted_content="ENC", item_id="rs_1"),
        ResponsesReasoningBlock(content=None, encrypted_content="ENC-ONLY", item_id="rs_2"),
        TextBlock("answer"),
    ]
    recorder.record_assistant(Message(role="assistant", content=blocks_in, stop_reason="end_turn"))
    recorder.finish_turn("completed")

    event = next(e for e in journal.read_events() if e.event_type == "assistant.message")
    public = to_public_event(event)
    assert public is not None
    rendered = json.dumps(public)
    blocks = public["payload"]["message"]["content"]
    types = [b["type"] for b in blocks]
    assert types == ["thinking", "responses_reasoning", "text"]
    assert "OPAQUE-SIGNATURE" not in rendered
    assert "ENC" not in rendered.replace("ENC-ONLY", "")
    assert "ENC-ONLY" not in rendered
    assert "signature" not in blocks[0]
    assert "artifact_fields" not in json.dumps(blocks)
    assert public["payload"]["origin_type"] == "llm"
    assert public["payload"]["llm_call_count"] == 1
    assert "usage_metadata" not in public["payload"]["message"]


def test_tool_blocks_use_canonical_execution_ids(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    recorder = SessionRecorder(journal, recover=False)
    recorder.start_turn(Message(role="user", content=[TextBlock("q")]))
    recorder.record_assistant(
        Message(
            role="assistant",
            content=[
                ToolCall(id="prov-1", name="read_file", input={"path": "a.py"}, execution_id="tool_exec_a"),
                ToolCall(
                    id="prov-2",
                    name="apply_patch",
                    input="*** Begin Patch",
                    execution_id="tool_exec_b",
                    input_kind="freeform",
                ),
            ],
            stop_reason="tool_use",
        )
    )
    recorder.record_tool_results(
        [
            ToolResult(
                tool_use_id="prov-2",
                content="patched",
                name="apply_patch",
                is_error=False,
                execution_id="tool_exec_b",
                input_kind="freeform",
            ),
            ToolResult(
                tool_use_id="prov-1", content="text", name="read_file", is_error=False, execution_id="tool_exec_a"
            ),
        ]
    )
    recorder.finish_turn("completed")

    events = journal.read_events()
    assistant = to_public_event(next(e for e in events if e.event_type == "assistant.message"))
    assert assistant is not None
    calls = [b for b in assistant["payload"]["message"]["content"] if b["type"] == "tool_call"]
    assert [c["tool_call_id"] for c in calls] == ["tool_exec_a", "tool_exec_b"]
    assert [c["provider_call_id"] for c in calls] == ["prov-1", "prov-2"]
    assert calls[0]["arguments"] == {"path": "a.py"}
    assert calls[1]["input_kind"] == "freeform"
    assert calls[1]["arguments"] == {"input": "*** Begin Patch"}
    assert calls[1]["input"] == "*** Begin Patch"

    results_event = to_public_event(next(e for e in events if e.event_type == "tool.results"))
    assert results_event is not None
    results = results_event["payload"]["message"]["content"]
    # Results arrived out of order; correlation is by id, never order.
    assert [r["tool_call_id"] for r in results] == ["tool_exec_b", "tool_exec_a"]
    assert all(r["provider_call_id"] for r in results)


def test_v1_tool_result_without_execution_id_falls_back_to_provider_id() -> None:
    event = SessionEvent(
        version=1,
        event_id="ev-1",
        session_id="legacy",
        seq=5,
        epoch_id="e1",
        turn_id="t1",
        timestamp="2026-01-01T00:00:00+00:00",
        actor="tool",
        event_type="tool.results",
        payload={
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "prov-9", "content": "x", "name": "bash", "is_error": False}
                ],
            }
        },
        artifacts=[],
    )
    public = to_public_event(event)
    assert public is not None
    result = public["payload"]["message"]["content"][0]
    assert result["tool_call_id"] == "prov-9"
    assert result["provider_call_id"] == "prov-9"


def test_secrets_and_home_paths_are_scrubbed(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    recorder = SessionRecorder(journal, recover=False)
    home = str(Path.home())
    recorder.start_turn(Message(role="user", content=[TextBlock(f"key sk-VERYSECRET123 lives at {home}/creds.txt")]))
    recorder.finish_turn("completed")

    event = next(e for e in journal.read_events() if e.event_type == "turn.started")
    public = to_public_event(event, secret_values=["sk-VERYSECRET123"])
    rendered = json.dumps(public)
    assert "sk-VERYSECRET123" not in rendered
    assert home not in rendered
    assert "~/creds.txt" in rendered


def test_artifact_refs_are_filtered_and_path_stripped(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    recorder = SessionRecorder(journal, recover=False)
    recorder.start_turn(Message(role="user", content=[TextBlock("q")]))
    recorder.record_assistant(
        Message(
            role="assistant",
            content=[ThinkingBlock("t", signature="SIG-ARTIFACT"), TextBlock("a")],
            stop_reason="end_turn",
        )
    )
    big = "x" * 200_000
    recorder.record_tool_results(
        [ToolResult(tool_use_id="p1", content=big, name="bash", is_error=False, execution_id="tool_exec_1")]
    )
    recorder.finish_turn("completed")

    events = journal.read_events()
    assistant_public = to_public_event(next(e for e in events if e.event_type == "assistant.message"))
    assert assistant_public is not None
    # The provider-signature artifact is not shareable: filtered from the envelope.
    assert assistant_public["artifacts"] == []

    results_public = to_public_event(next(e for e in events if e.event_type == "tool.results"))
    assert results_public is not None
    assert len(results_public["artifacts"]) == 1
    ref = results_public["artifacts"][0]
    assert ref["purpose"] == "tool_result"
    assert "path" not in ref
    block_ref = results_public["payload"]["message"]["content"][0]["content_artifact"]
    assert "path" not in block_ref


def test_mixed_v1_and_v2_journals_read_and_append(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    v1_lines = [
        {
            "version": 1,
            "id": "ev-1",
            "session_id": "mix",
            "seq": 1,
            "epoch_id": "e1",
            "turn_id": None,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "actor": "system",
            "type": "session.created",
            "payload": {"metadata": {}},
            "artifacts": [],
        },
        {
            "version": 1,
            "id": "ev-2",
            "session_id": "mix",
            "seq": 2,
            "epoch_id": "e1",
            "turn_id": None,
            "timestamp": "2026-01-01T00:00:01+00:00",
            "actor": "system",
            "type": "context.epoch_started",
            "payload": {"reason": "session_created"},
            "artifacts": [],
        },
    ]
    (session_dir / "events.jsonl").write_text("".join(json.dumps(line) + "\n" for line in v1_lines), encoding="utf-8")
    journal = SessionJournal("mix", session_dir)
    events = journal.read_events()
    assert [e.version for e in events] == [1, 1]

    appended = journal.append(
        "turn.started", actor="user", payload={"message": {"role": "user", "content": []}}, turn_id="t1"
    )
    assert appended.version == EVENT_SCHEMA_VERSION == 2
    events = journal.read_events()
    assert [e.version for e in events] == [1, 1, 2]
    assert [e.seq for e in events] == [1, 2, 3]


def test_listener_receives_persisted_event_and_failures_do_not_break_append(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    seen: list[SessionEvent] = []

    def bad_listener(event: SessionEvent) -> None:
        raise RuntimeError("listener boom")

    journal.add_listener(bad_listener)
    journal.add_listener(seen.append)
    bootstrap(journal)

    assert len(seen) == 2
    persisted = journal.read_events()
    assert seen[0].event_id == persisted[0].event_id
    assert seen[0].seq == persisted[0].seq
    assert seen[0].timestamp == persisted[0].timestamp


def test_printer_streams_the_same_records_as_the_saved_projection(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    buffer = StringIO()
    printer = SemanticStdoutPrinter(stream=buffer)
    journal.add_listener(printer)
    bootstrap(journal)
    recorder = SessionRecorder(journal, recover=False)
    drive_turn(recorder)

    live = [json.loads(line) for line in buffer.getvalue().splitlines()]
    saved = [p for p in (to_public_event(e) for e in journal.read_events()) if p is not None]
    assert [(r["id"], r["seq"], r["timestamp"]) for r in live] == [(r["id"], r["seq"], r["timestamp"]) for r in saved]
    assert printer.emitted_total == len(saved)


def test_printer_backlog_does_not_duplicate_live_events(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    buffer = StringIO()
    printer = SemanticStdoutPrinter(stream=buffer)
    journal.add_listener(printer)
    bootstrap(journal)
    printer.emit_backlog(journal.read_events())
    assert printer.emitted_total == 2


def test_in_memory_journal_matches_file_journal_and_touches_no_disk(tmp_path: Path) -> None:
    def run(journal: SessionJournal) -> list[str]:
        bootstrap(journal)
        recorder = SessionRecorder(journal, recover=False)
        drive_turn(recorder)
        return [e.event_type for e in journal.read_events()]

    file_types = run(make_file_journal(tmp_path))
    memory_journal = InMemorySessionJournal("s1")
    memory_types = run(memory_journal)
    assert memory_types == file_types
    assert not Path("<in-memory>").exists()

    big = "y" * 200_000
    recorder = SessionRecorder(memory_journal, recover=False)
    recorder.start_turn(Message(role="user", content=[TextBlock("more")]))
    stored = recorder.record_tool_results(
        [ToolResult(tool_use_id="p", content=big, name="bash", is_error=False, execution_id="tool_exec_z")]
    )
    recorder.finish_turn("completed")
    assert len(stored) == 1
    event = next(e for e in memory_journal.read_events() if e.event_type == "tool.results")
    ref = event.payload["message"]["content"][0]["content_artifact"]
    assert "path" not in ref
    assert memory_journal.read_artifact(ref).decode() == big


def test_llm_call_stamp_is_lifted_into_the_assistant_payload(tmp_path: Path) -> None:
    from kolega_code.llm.ledger import UsageLedger

    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    recorder = SessionRecorder(journal, recover=False)
    ledger = UsageLedger()
    message = Message(role="assistant", content=[TextBlock("hi")], stop_reason="end_turn")
    request_id = ledger.begin("anthropic", "claude-opus-5")
    ledger.record_response(request_id, None, message)

    recorder.start_turn(Message(role="user", content=[TextBlock("q")]))
    recorder.record_assistant(message, reasoning_effort="high")
    recorder.finish_turn("completed")

    event = next(e for e in journal.read_events() if e.event_type == "assistant.message")
    assert event.payload["llm_call_id"] == request_id
    assert event.payload["run_id"] == ledger.run_id
    assert event.payload["provider"] == "anthropic"
    assert event.payload["model"] == "claude-opus-5"
    assert event.payload["reasoning_effort"] == "high"
    assert event.payload["llm_call_count"] == 1
    # The stamp is payload metadata, not part of the replayable message.
    assert "llm_call" not in json.dumps(event.payload["message"])


def test_synthetic_assistant_notice_shape(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    recorder = SessionRecorder(journal, recover=False)
    recorder.record_synthetic_assistant("Prompt blocked by policy.", notice_code="hook_blocked")

    event = next(e for e in journal.read_events() if e.event_type == "assistant.message")
    assert event.payload["origin_type"] == "synthetic"
    assert event.payload["llm_call_id"] is None
    assert event.payload["llm_call_count"] == 0
    assert event.payload["notice_code"] == "hook_blocked"
    assert "provider" not in event.payload
    assert event.turn_id is None


def test_system_context_is_fingerprinted_and_reset_by_epoch(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    recorder = SessionRecorder(journal, recover=False)
    assert recorder.record_system_context("You are a helpful agent.") is True
    assert recorder.record_system_context("You are a helpful agent.") is False
    assert recorder.record_system_context("Changed prompt.") is True
    recorder.start_epoch("agent_clear_command")
    assert recorder.record_system_context("Changed prompt.") is True
    assert sum(1 for e in journal.read_events() if e.event_type == "context.system") == 3


def test_tool_definitions_are_fingerprinted_and_reset_by_epoch(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    recorder = SessionRecorder(journal, recover=False)
    tools = [{"name": "read_file", "description": "Read a file.", "input_schema": {"type": "object"}}]
    assert recorder.record_tool_definitions([]) is False
    assert recorder.record_tool_definitions(tools) is True
    assert recorder.record_tool_definitions(tools) is False
    assert recorder.record_tool_definitions(tools + [{"name": "bash", "description": "", "input_schema": {}}]) is True
    recorder.start_epoch("agent_clear_command")
    assert recorder.record_tool_definitions(tools) is True
    events = [e for e in journal.read_events() if e.event_type == "context.tools"]
    assert len(events) == 3
    assert events[0].payload["tools"][0]["name"] == "read_file"
    public = to_public_event(events[0])
    assert public is not None
    assert public["payload"]["tools"] == tools


def test_scoped_child_recorder_stamps_lineage_and_interleaves(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    root = SessionRecorder(journal, recover=False)
    root.start_turn(Message(role="user", content=[TextBlock("dispatch two agents")]))

    child_a = root.scoped_child(agent_id="agent-a", agent_name="explorer", parent_tool_call_id="tool_exec_a", depth=1)
    child_b = root.scoped_child(agent_id="agent-b", agent_name="explorer", parent_tool_call_id="tool_exec_b", depth=1)
    child_a.record_agent_started({"task": "a"}, turn_id=root.current_turn_id)
    child_b.record_agent_started({"task": "b"}, turn_id=root.current_turn_id)
    child_a.start_turn(Message(role="user", content=[TextBlock("task a")]))
    child_b.start_turn(Message(role="user", content=[TextBlock("task b")]))
    child_a.record_assistant(Message(role="assistant", content=[TextBlock("done a")], stop_reason="end_turn"))
    child_a.finish_turn("completed")
    child_b.finish_turn("failed", error="boom")
    child_a.record_agent_terminal("completed", {"summary": "done a"})
    child_b.record_agent_terminal("failed", {"error": "boom"})
    root.finish_turn("completed")

    events = journal.read_events()
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    a_events = [e for e in events if e.agent_id == "agent-a"]
    assert {e.parent_tool_call_id for e in a_events} == {"tool_exec_a"}
    assert {e.parent_agent_id for e in a_events} == {derive_root_agent_id("s1")}
    assert {e.depth for e in a_events} == {1}
    assert [e.event_type for e in a_events] == [
        "agent.started",
        "turn.started",
        "assistant.message",
        "turn.completed",
        "agent.completed",
    ]

    # Sibling agents may share a name; identity is the id.
    b_events = [e for e in events if e.agent_id == "agent-b"]
    assert {e.agent_name for e in a_events} == {e.agent_name for e in b_events} == {"explorer"}

    # Subagent turns never appear in the root's rewindable history.
    turns = collect_epoch_turns(events, journal.epoch_id)
    assert [t.user_text for t in turns] == ["dispatch two agents"]


def test_scoped_recorders_cannot_reset_rewind_or_terminate_the_run(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    root = SessionRecorder(journal, recover=False)
    child = root.scoped_child(agent_id="a", agent_name="x", parent_tool_call_id="t", depth=1)
    with pytest.raises(SessionJournalError):
        child.start_epoch("agent_clear_command")
    with pytest.raises(SessionJournalError):
        child.record_rewind("some-turn")
    with pytest.raises(SessionJournalError):
        child.record_run_terminal("completed", {})


def test_run_terminal_records_once_and_only_once(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    recorder = SessionRecorder(journal, recover=False)
    recorder.record_run_terminal("failed", {"error": {"code": "billing_error"}})
    recorder.record_run_terminal("completed", {})
    events = [e for e in journal.read_events() if e.event_type.startswith("run.")]
    assert [e.event_type for e in events] == ["run.failed"]


def test_recovery_ignores_open_subagent_turns(tmp_path: Path) -> None:
    journal = make_file_journal(tmp_path)
    bootstrap(journal)
    root = SessionRecorder(journal, recover=False)
    root.start_turn(Message(role="user", content=[TextBlock("go")]))
    child = root.scoped_child(agent_id="a", agent_name="x", parent_tool_call_id="t", depth=1)
    child.start_turn(Message(role="user", content=[TextBlock("subtask")]))
    root.finish_turn("completed")
    # The child's turn is left open (interrupted dispatch). Recovery must not
    # reopen or close it as if it were the root's.
    recovered = SessionRecorder(journal, recover=True)
    assert recovered.current_turn_id is None
    types = [e.event_type for e in journal.read_events()]
    assert types.count("turn.completed") == 1


def test_continuation_turn_started_passes_through_without_message() -> None:
    journal = InMemorySessionJournal("proto-continuation")
    bootstrap(journal)
    recorder = SessionRecorder(journal, recover=False)
    recorder.start_continuation_turn()
    recorder.finish_turn("completed")

    started = next(event for event in journal.read_events() if event.event_type == "turn.started")
    public = to_public_event(started)
    assert public is not None
    assert public["type"] == "turn.started"
    assert public["actor"] == "system"
    assert public["payload"] == {"continuation": True}
