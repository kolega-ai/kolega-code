import json
import os
import stat
from pathlib import Path

import pytest

from kolega_code.cli import session_journal as session_journal_module
from kolega_code.cli.session_journal import SessionJournalError, SessionRecorder, TOOL_RESULT_PREVIEW_CHARS
from kolega_code.cli.session_store import SessionRecord, SessionStore, SessionStoreError, default_state_dir
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    ResponsesReasoningBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
)


def test_default_state_dir_honors_env() -> None:
    assert default_state_dir({"KOLEGA_CODE_STATE_DIR": "/tmp/kolega-test"}) == Path("/tmp/kolega-test")


@pytest.mark.parametrize("session_id", ["", ".", "..", "../escape", "nested/session", "nested\\session"])
def test_session_ids_cannot_escape_the_sessions_directory(tmp_path: Path, session_id: str) -> None:
    store = SessionStore(tmp_path / "state")
    with pytest.raises(SessionStoreError, match="single non-empty path component"):
        store.session_dir_for(session_id)


def test_session_store_writes_private_directory_layout(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")

    old_umask = os.umask(0)
    try:
        record = store.create(project, "code", {"api_key": "secret"})
    finally:
        os.umask(old_umask)

    assert store.path_for(record.session_id).name == "metadata.json"
    assert store.events_path_for(record.session_id).name == "events.jsonl"
    assert (store.session_dir_for(record.session_id) / "artifacts").is_dir()
    if os.name != "nt":
        assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
        assert stat.S_IMODE(store.sessions_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(store.session_dir_for(record.session_id).stat().st_mode) == 0o700
        assert stat.S_IMODE(store.path_for(record.session_id).stat().st_mode) == 0o600
        assert stat.S_IMODE(store.events_path_for(record.session_id).stat().st_mode) == 0o600


def test_session_store_create_load_list_export_delete(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")

    record = store.create(project, "code", {"long_model": "claude-opus-4-7"}, title="Project")
    recorder = store.recorder(record.session_id)
    recorder.start_turn(Message(role="user", content=[TextBlock("inspect")]))
    recorder.record_assistant(Message(role="assistant", content=[TextBlock("done")], stop_reason="end_turn"))
    recorder.finish_turn("completed")
    record.task_list_markdown = "- [ ] inspect\n- [x] plan"
    record.latest_plan_markdown = "# Plan\n\nImplement it."
    record.plan_pending = True
    record.plan_reofferable = True
    record.interaction_mode = "plan"
    record.permission_mode = "auto"
    record.gigacode_enabled = True
    store.save(record)

    loaded = store.load(record.session_id)
    assert loaded.project_path == str(project.resolve())
    assert [message["role"] for message in loaded.history] == ["user", "assistant"]
    assert loaded.task_list_markdown == "- [ ] inspect\n- [x] plan"
    assert loaded.latest_plan_markdown == "# Plan\n\nImplement it."
    assert loaded.plan_pending is True
    assert loaded.plan_reofferable is True
    assert loaded.interaction_mode == "plan"
    assert loaded.permission_mode == "auto"
    assert loaded.gigacode_enabled is True
    latest = store.latest_for_project(project)
    assert latest is not None
    assert latest.session_id == record.session_id
    exported = store.export(record.session_id)
    assert record.session_id in exported
    assert "task_list_markdown" in exported
    assert "latest_plan_markdown" in exported
    assert "interaction_mode" in exported

    store.delete(record.session_id)
    assert not store.session_dir_for(record.session_id).exists()
    with pytest.raises(SessionStoreError):
        store.load(record.session_id)


def test_save_is_metadata_only(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    events_before = store.events_path_for(record.session_id).read_bytes()

    record.history = [Message(role="user", content=[TextBlock("must not be snapshotted")]).to_dict()]
    record.compaction = {"summary": "must not be snapshotted"}
    store.save(record)

    loaded = store.load(record.session_id)
    assert loaded.history == []
    assert loaded.compaction == {}
    metadata = json.loads(store.path_for(record.session_id).read_text(encoding="utf-8"))
    assert "history" not in metadata
    assert "compaction" not in metadata
    assert store.events_path_for(record.session_id).read_bytes().startswith(events_before)


def test_event_records_are_ordered_and_have_stable_envelope(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    recorder = store.recorder(record.session_id)
    recorder.start_turn(Message(role="user", content=[TextBlock("hello")]))
    recorder.record_assistant(Message(role="assistant", content=[TextBlock("hi")], stop_reason="end_turn"))
    recorder.finish_turn("completed")

    events = store.journal(record.session_id).read_events()
    assert [event.seq for event in events] == list(range(1, len(events) + 1))
    assert [event.event_type for event in events] == [
        "session.created",
        "context.epoch_started",
        "turn.started",
        "assistant.message",
        "turn.completed",
    ]
    assert all(event.version == 1 for event in events)
    assert all(event.session_id == record.session_id for event in events)
    assert all(event.event_id and event.timestamp and event.epoch_id for event in events)


def test_incomplete_final_jsonl_fragment_is_repaired(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    path = store.events_path_for(record.session_id)
    original = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"version":1,"id":"partial"')

    events = store.journal(record.session_id).read_events(repair_tail=True)

    assert len(events) == 2
    assert path.read_bytes() == original


def test_short_append_is_repaired_before_the_next_sequence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    journal = store.journal(record.session_id)
    real_write = session_journal_module.os.write

    def short_write(fd, data):
        prefix = data[: max(1, len(data) // 2)]
        return real_write(fd, prefix)

    monkeypatch.setattr(session_journal_module.os, "write", short_write)
    with pytest.raises(SessionJournalError, match="Short event write"):
        journal.append("context.message", actor="user", payload={"message": {"role": "user", "content": []}})
    monkeypatch.setattr(session_journal_module.os, "write", real_write)

    journal.append("context.message", actor="user", payload={"message": {"role": "user", "content": []}})
    events = journal.read_events()
    assert [event.seq for event in events] == [1, 2, 3]
    assert events[-1].event_type == "context.message"


def test_mid_file_corruption_and_sequence_gaps_are_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    corrupt = store.create(project, "code", {})
    corrupt_path = store.events_path_for(corrupt.session_id)
    lines = corrupt_path.read_text(encoding="utf-8").splitlines()
    corrupt_path.write_text(lines[0] + "\n{not-json}\n" + lines[1] + "\n", encoding="utf-8")
    with pytest.raises(SessionStoreError, match="Invalid JSON"):
        store.load(corrupt.session_id)

    gap = store.create(project, "code", {})
    gap_path = store.events_path_for(gap.session_id)
    rows = [json.loads(line) for line in gap_path.read_text(encoding="utf-8").splitlines()]
    rows[1]["seq"] = 3
    gap_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(SessionStoreError, match="sequence gap"):
        store.load(gap.session_id)


def test_context_epochs_replay_only_the_current_epoch(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    recorder = store.recorder(record.session_id)
    recorder.record_context_message(Message(role="user", content=[TextBlock("old context")]))
    old_epoch = recorder.journal.epoch_id

    new_epoch = recorder.start_epoch("thread_reset")
    recorder.record_context_message(Message(role="user", content=[TextBlock("new context")]))

    loaded = store.load(record.session_id)
    assert [Message.from_dict(item).get_text_content() for item in loaded.history] == ["new context"]
    assert old_epoch != new_epoch
    events = store.journal(record.session_id).read_events()
    assert any(event.epoch_id == old_epoch and event.event_type == "context.message" for event in events)


def test_compaction_is_an_incremental_event(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    compaction = {"summary": "## Goal\nShip it", "compacted_through": 7, "compacted_history_length": 9}

    store.recorder(record.session_id).record_compaction(compaction)

    assert store.load(record.session_id).compaction == compaction
    assert store.journal(record.session_id).read_events()[-1].event_type == "context.compacted"


def test_large_tool_result_uses_artifact_and_bounded_preview(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    recorder = store.recorder(record.session_id)
    full = "begin\n" + ("x" * (TOOL_RESULT_PREVIEW_CHARS + 50_000)) + "\nend"
    recorder.start_turn(Message(role="user", content=[TextBlock("read it")]))
    recorder.record_assistant(
        Message(
            role="assistant",
            content=[ToolCall(id="call-1", name="read_file", input={"path": "large.txt"})],
            stop_reason="tool_use",
        )
    )
    prepared = recorder.record_tool_results(
        [ToolResult(tool_use_id="call-1", content=full, name="read_file", is_error=False)]
    )
    recorder.finish_turn("failed", error="test stopped after tool")

    assert isinstance(prepared[0].content, str)
    assert len(prepared[0].content) <= TOOL_RESULT_PREVIEW_CHARS
    assert "Full output:" in prepared[0].content
    event = next(
        event for event in store.journal(record.session_id).read_events() if event.event_type == "tool.results"
    )
    ref = event.artifacts[0]
    assert store.journal(record.session_id).read_artifact(ref).decode() == full
    loaded_result = Message.from_dict(store.load(record.session_id).history[2]).content[0]
    assert isinstance(loaded_result, ToolResult)
    assert loaded_result.content == prepared[0].content
    bug_export = store.bug_export(record.session_id)
    assert full not in bug_export.session_json
    assert full not in bug_export.events_jsonl
    assert ref["sha256"] in bug_export.artifact_manifest_json


def test_images_and_provider_opaque_fields_are_hydrated_from_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    recorder = store.recorder(record.session_id)
    image_data = base64_png = "iVBORw0KGgo="
    recorder.record_context_message(
        Message(
            role="user",
            content=[ImageBlock(image_type="base64", media_type="image/png", data=image_data)],
        )
    )
    recorder.record_context_message(
        Message(
            role="assistant",
            content=[
                ThinkingBlock("private", signature="signed-provider-value"),
                ResponsesReasoningBlock("encrypted-provider-value", summary=["summary"]),
                ToolCall(id="call", name="noop", input={}, thought_signature=b"thought-bytes"),
            ],
        ),
        actor="assistant",
    )

    loaded = [Message.from_dict(item) for item in store.load(record.session_id).history]
    assert isinstance(loaded[0].content, list)
    assert isinstance(loaded[0].content[0], ImageBlock)
    assert loaded[0].content[0].data == base64_png
    assert isinstance(loaded[1].content, list)
    assert isinstance(loaded[1].content[0], ThinkingBlock)
    assert loaded[1].content[0].signature == "signed-provider-value"
    assert isinstance(loaded[1].content[1], ResponsesReasoningBlock)
    assert loaded[1].content[1].encrypted_content == "encrypted-provider-value"
    assert isinstance(loaded[1].content[2], ToolCall)
    assert loaded[1].content[2].thought_signature == b"thought-bytes"
    raw_events = store.events_path_for(record.session_id).read_text(encoding="utf-8")
    assert image_data not in raw_events
    assert "signed-provider-value" not in raw_events
    assert "encrypted-provider-value" not in raw_events
    bug_export = store.bug_export(record.session_id)
    assert image_data not in bug_export.session_json
    assert "signed-provider-value" not in bug_export.session_json
    assert "encrypted-provider-value" not in bug_export.session_json

    # Bug reporting remains available even if a referenced binary artifact is
    # damaged; the bundle carries the event-safe projection and manifest.
    image_event = next(event for event in store.journal(record.session_id).read_events() if event.artifacts)
    image_ref = image_event.artifacts[0]
    (store.journal(record.session_id).artifacts_dir / image_ref["sha256"]).unlink()
    assert store.bug_export(record.session_id).session_json
    with pytest.raises(SessionStoreError, match="Missing session artifact"):
        store.load(record.session_id)


def test_interrupted_tool_call_gets_synthetic_error_and_is_not_rerun(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    interrupted = SessionRecorder(store.journal(record.session_id), recover=False)
    interrupted.start_turn(Message(role="user", content=[TextBlock("change it")]))
    interrupted.record_assistant(
        Message(
            role="assistant",
            content=[ToolCall(id="write-1", name="write", input={"path": "a.txt", "content": "x"})],
            stop_reason="tool_use",
        )
    )

    assert SessionRecorder(store.journal(record.session_id), recover=True).current_turn_id is None

    loaded = [Message.from_dict(item) for item in store.load(record.session_id).history]
    result = loaded[-1].content[0]
    assert isinstance(result, ToolResult)
    assert result.tool_use_id == "write-1"
    assert result.is_error is True
    assert "was not re-run" in result.content
    assert store.journal(record.session_id).read_events()[-1].event_type == "turn.failed"


def test_terminal_assistant_message_is_recovered_when_only_marker_is_missing(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    interrupted = SessionRecorder(store.journal(record.session_id), recover=False)
    interrupted.start_turn(Message(role="user", content=[TextBlock("hello")]))
    interrupted.record_assistant(Message(role="assistant", content=[TextBlock("done")], stop_reason="end_turn"))

    SessionRecorder(store.journal(record.session_id), recover=True)

    events = store.journal(record.session_id).read_events()
    assert events[-1].event_type == "turn.completed"
    assert Message.from_dict(store.load(record.session_id).history[-1]).get_text_content() == "done"


def test_metadata_event_survives_failure_before_projection_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    original_write = store._write_metadata
    record.title = "new durable title"

    def fail_write(metadata):
        raise OSError("projection write failed")

    monkeypatch.setattr(store, "_write_metadata", fail_write)
    with pytest.raises(OSError, match="projection write failed"):
        store.save(record)

    assert store.load(record.session_id).title == "new durable title"
    monkeypatch.setattr(store, "_write_metadata", original_write)


@pytest.mark.parametrize("damage", ["missing", "invalid"])
def test_metadata_projection_is_rebuilt_from_canonical_events(tmp_path: Path, damage: str) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {}, title="durable title")
    path = store.path_for(record.session_id)
    if damage == "missing":
        path.unlink()
    else:
        path.write_text("{not-json", encoding="utf-8")

    loaded = store.load(record.session_id)

    assert loaded.title == "durable title"
    assert json.loads(path.read_text(encoding="utf-8"))["title"] == "durable title"


def test_lazy_migration_preserves_history_compaction_and_deletes_legacy_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    store.ensure_dirs()
    record = SessionRecord.create(project, "code", {}, session_id="legacy")
    record.history = [
        Message(role="user", content=[TextBlock("old request")]).to_dict(),
        Message(role="assistant", content=[TextBlock("old answer")], stop_reason="end_turn").to_dict(),
    ]
    record.compaction = {"summary": "old summary", "compacted_through": 1}
    legacy = store.legacy_path_for(record.session_id)
    legacy.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    loaded = store.load(record.session_id)

    assert loaded.history == record.history
    assert loaded.compaction == record.compaction
    assert store.path_for(record.session_id).exists()
    assert store.events_path_for(record.session_id).exists()
    assert not legacy.exists()


def test_lazy_migration_repairs_incomplete_tool_call_and_uses_stable_artifact_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    store.ensure_dirs()
    record = SessionRecord.create(project, "code", {}, session_id="legacy-incomplete")
    full = "z" * (TOOL_RESULT_PREVIEW_CHARS + 10_000)
    record.history = [
        Message(role="user", content=[TextBlock("read")]).to_dict(),
        Message(
            role="assistant",
            content=[ToolCall(id="complete", name="read_file", input={"path": "a.txt"})],
            stop_reason="tool_use",
        ).to_dict(),
        Message(
            role="user",
            content=[ToolResult(tool_use_id="complete", content=full, name="read_file", is_error=False)],
        ).to_dict(),
        Message(
            role="assistant",
            content=[ToolCall(id="missing", name="write", input={"path": "b.txt", "content": "x"})],
            stop_reason="tool_use",
        ).to_dict(),
    ]
    legacy = store.legacy_path_for(record.session_id)
    legacy.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    loaded = store.load(record.session_id)

    messages = [Message.from_dict(item) for item in loaded.history]
    synthetic = messages[-1].content[0]
    assert isinstance(synthetic, ToolResult)
    assert synthetic.tool_use_id == "missing"
    assert synthetic.is_error is True
    persisted_large_result = loaded.history[2]["content"][0]
    artifact_path = Path(persisted_large_result["content_artifact"]["path"])
    assert artifact_path.exists()
    assert ".migrating-" not in str(artifact_path)
    assert store.journal(record.session_id).read_events()[-1].event_type == "turn.failed"


def test_new_metadata_loads_when_optional_planning_fields_are_absent(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    payload = json.loads(store.path_for(record.session_id).read_text(encoding="utf-8"))
    for field in (
        "task_list_markdown",
        "latest_plan_markdown",
        "plan_pending",
        "plan_reofferable",
        "interaction_mode",
        "permission_mode",
        "gigacode_enabled",
    ):
        payload.pop(field)
    payload["latest_plan_markdown"] = "# Plan\n\nImplement it."
    payload["plan_pending"] = True
    store.path_for(record.session_id).write_text(json.dumps(payload), encoding="utf-8")

    loaded = store.load(record.session_id)
    assert loaded.task_list_markdown == ""
    assert loaded.latest_plan_markdown == "# Plan\n\nImplement it."
    assert loaded.plan_pending is True
    assert loaded.plan_reofferable is True
    assert loaded.interaction_mode == "build"
    assert loaded.permission_mode == "ask"
    assert loaded.gigacode_enabled is False


def test_session_store_ignores_corrupt_legacy_files_when_listing(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "state")
    store.ensure_dirs()
    (store.sessions_dir / "bad.json").write_text("{not json", encoding="utf-8")

    assert store.list() == []
