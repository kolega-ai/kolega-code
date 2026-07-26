"""Presentation projection: fidelity, determinism, and tolerance.

These are the properties the replay player and the web client depend on. The
seek property in particular is what makes scrubbing correct: folding a prefix of
the log must give exactly the state the viewer would have reached by watching.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kolega_code.events import AgentEvent, ArtifactRef, KnownEventType
from kolega_code.session.projection import (
    ProjectionError,
    PresentationState,
    fold,
    replay,
)


def _event(event_type: str, seq: int, *, elapsed_ms: int = 0, uuid: str | None = None, **content: Any) -> AgentEvent:
    event = AgentEvent(
        session_id="s",
        sender="agent",
        event_type=event_type,
        content=dict(content),
        seq=seq,
        elapsed_ms=elapsed_ms,
    )
    if uuid is not None:
        event.uuid = uuid
    return event


def _session_events() -> list[AgentEvent]:
    """A representative session: a turn, reasoning, prose, a tool, and output."""
    return [
        _event(KnownEventType.TURN_STARTED, 1, elapsed_ms=0, turn_id="t1", user_text="fix the bug"),
        _event(KnownEventType.THINKING_DELTA, 2, elapsed_ms=100, uuid="think-1", text="Let me ", complete=False),
        _event(KnownEventType.THINKING_DELTA, 3, elapsed_ms=200, uuid="think-1", text="look.", complete=True),
        _event(KnownEventType.ASSISTANT_DELTA, 4, elapsed_ms=300, uuid="say-1", text="I will ", complete=False),
        _event(KnownEventType.ASSISTANT_DELTA, 5, elapsed_ms=400, uuid="say-1", text="edit it.", complete=True),
        _event(
            KnownEventType.CHAT_MESSAGE,
            6,
            elapsed_ms=500,
            message_type="tool_call",
            text="Editing app.py",
            tool_description="edit",
            tool_call_id="call-1",
        ),
        _event(KnownEventType.TERMINAL_COMMAND, 7, elapsed_ms=600, command="pytest -q"),
        _event(KnownEventType.TERMINAL_OUTPUT, 8, elapsed_ms=700, output="1 passed\n"),
        _event(
            KnownEventType.CHAT_MESSAGE,
            9,
            elapsed_ms=800,
            message_type="tool_result",
            text="Edited 1 file",
            tool_call_id="call-1",
        ),
        _event(KnownEventType.TURN_ENDED, 10, elapsed_ms=900, turn_id="t1", status="completed"),
    ]


def test_folds_a_session_into_a_transcript() -> None:
    state = replay(_session_events())

    kinds = [item.kind for item in state.conversation]
    assert kinds == ["user", "thinking", "assistant", "tool"], f"unexpected transcript shape: {kinds}"
    assert state.conversation[0].text == "fix the bug"
    assert state.conversation[1].text == "Let me look.", "reasoning deltas must concatenate in order"
    assert state.conversation[2].text == "I will edit it.", "streamed deltas must concatenate in order"
    assert state.conversation[2].complete is True
    assert state.terminal == "$ pytest -q\n1 passed\n"
    assert state.activity == "idle"


def test_streamed_segments_accumulate_not_replace() -> None:
    state = replay(
        [
            _event(KnownEventType.ASSISTANT_DELTA, 1, uuid="a", text="one ", complete=False),
            _event(KnownEventType.ASSISTANT_DELTA, 2, uuid="a", text="two ", complete=False),
            _event(KnownEventType.ASSISTANT_DELTA, 3, uuid="a", text="three", complete=True),
        ]
    )
    assert [item.text for item in state.conversation] == ["one two three"]


def test_interleaved_streams_stay_in_separate_blocks() -> None:
    """Reasoning and prose arrive interleaved and must not merge."""
    state = replay(
        [
            _event(KnownEventType.THINKING_DELTA, 1, uuid="t", text="hmm", complete=False),
            _event(KnownEventType.ASSISTANT_DELTA, 2, uuid="a", text="hello", complete=False),
            _event(KnownEventType.THINKING_DELTA, 3, uuid="t", text=" more", complete=True),
            _event(KnownEventType.ASSISTANT_DELTA, 4, uuid="a", text=" there", complete=True),
        ]
    )
    assert [(item.kind, item.text) for item in state.conversation] == [
        ("thinking", "hmm more"),
        ("assistant", "hello there"),
    ]


def test_tool_result_updates_the_call_in_place() -> None:
    state = replay(_session_events())
    tools = [item for item in state.conversation if item.kind == "tool"]
    assert len(tools) == 1, "a call and its result are one transcript entry, not two"
    assert tools[0].status == "done"
    assert tools[0].text == "Edited 1 file"


def test_tool_error_marks_failure() -> None:
    state = replay(
        [
            _event(
                KnownEventType.CHAT_MESSAGE,
                1,
                message_type="tool_call",
                tool_call_id="c",
                tool_description="build",
                text="Building",
            ),
            _event(KnownEventType.CHAT_MESSAGE, 2, message_type="tool_error", tool_call_id="c", text="boom"),
        ]
    )
    (tool,) = [item for item in state.conversation if item.kind == "tool"]
    assert tool.status == "failed" and tool.text == "boom"


def test_sub_agent_activity_is_routed_away_from_the_main_transcript() -> None:
    event = _event(KnownEventType.ASSISTANT_DELTA, 1, uuid="s", text="delegated work", complete=True)
    event.sub_agent_info = {"dispatch_id": "d1", "agent_name": "investigator", "task": "find it"}

    state = replay([event])

    assert state.conversation == [], "sub-agent output must not appear inline in the main transcript"
    assert "d1" in state.sub_agents
    activity = state.sub_agents["d1"]
    assert activity.name == "investigator"
    assert activity.task == "find it"
    assert [step.text for step in activity.steps] == ["delegated work"]


def test_sub_agent_compaction_does_not_touch_main_indicator() -> None:
    main = _event(KnownEventType.COMPACTION_STATUS, 1, phase="started", message="main")
    delegate = _event(KnownEventType.COMPACTION_STATUS, 2, phase="finished", message="delegate")
    delegate.sub_agent_info = {"dispatch_id": "d1"}

    state = replay([main, delegate])

    assert state.compaction is not None
    assert state.compaction["message"] == "main", "a delegate's compaction stomped the main indicator"


def test_tool_streaming_replace_and_append_modes() -> None:
    base = _event(
        KnownEventType.CHAT_MESSAGE,
        1,
        message_type="tool_call",
        tool_call_id="c",
        tool_description="run",
        text="",
    )
    replace = replay(
        [
            base,
            _event(KnownEventType.TOOL_STREAMING_UPDATE, 2, tool_call_id="c", text="abc", stream_mode="replace"),
            _event(KnownEventType.TOOL_STREAMING_UPDATE, 3, tool_call_id="c", text="abcdef", stream_mode="replace"),
        ]
    )
    (tool,) = [item for item in replace.conversation if item.kind == "tool"]
    assert tool.text == "abcdef", "replace mode must overwrite, not concatenate"

    appended = replay(
        [
            base.model_copy(deep=True),
            _event(KnownEventType.TOOL_STREAMING_UPDATE, 2, tool_call_id="c", text="abc", stream_mode="append"),
            _event(KnownEventType.TOOL_STREAMING_UPDATE, 3, tool_call_id="c", text="def", stream_mode="append"),
        ]
    )
    (tool,) = [item for item in appended.conversation if item.kind == "tool"]
    assert tool.text == "abcdef", "append mode must concatenate"


def test_turn_markers_support_seeking_by_turn() -> None:
    state = replay(_session_events())
    assert len(state.turns) == 1
    marker = state.turns[0]
    assert marker.turn_id == "t1"
    assert marker.status == "completed"
    assert marker.started_seq == 1
    assert marker.ended_ms == 900


def test_context_and_edit_previews_are_captured() -> None:
    state = replay(
        [
            _event(
                KnownEventType.LLM_CONTEXT_UPDATE,
                1,
                input_tokens=1000,
                max_tokens=8000,
                usage_percentage=12.5,
                alert_level="ok",
                message=None,
                will_compress_at=6400,
            ),
            _event(KnownEventType.FILE_EDIT_PREVIEW, 2, path="app.py", diff="@@ -1 +1 @@", tool_call_id="c"),
        ]
    )
    assert state.context is not None and state.context.usage_percentage == 12.5
    assert len(state.edit_previews) == 1 and state.edit_previews[0]["path"] == "app.py"


def test_seek_equals_watching_up_to_that_point() -> None:
    """Scrubbing correctness: a prefix fold must match incremental folding."""
    events = _session_events()
    for cut in range(1, len(events) + 1):
        incremental = PresentationState()
        for event in events[:cut]:
            fold(incremental, event)
        assert replay(events[:cut]).to_dict() == incremental.to_dict(), f"seek to {cut} diverged from watching"


def test_unknown_event_types_are_inert_but_reported() -> None:
    """A client running against a newer agent degrades instead of crashing."""
    state = replay(
        [
            _event("some_future_event", 1, text="from the future"),
            _event(KnownEventType.ASSISTANT_DELTA, 2, uuid="a", text="still works", complete=True),
        ]
    )
    assert state.unknown_event_types == {"some_future_event"}
    assert [item.text for item in state.conversation] == ["still works"]


def test_out_of_order_events_raise_rather_than_corrupt() -> None:
    state = PresentationState()
    fold(state, _event(KnownEventType.ASSISTANT_DELTA, 5, uuid="a", text="x", complete=True))
    with pytest.raises(ProjectionError):
        fold(state, _event(KnownEventType.ASSISTANT_DELTA, 4, uuid="b", text="y", complete=True))


def test_shuffled_log_raises() -> None:
    events = _session_events()
    shuffled = [events[3], events[0], events[7]]
    state = PresentationState()
    with pytest.raises(ProjectionError):
        for event in shuffled:
            fold(state, event)


def test_sparse_sequence_numbers_are_accepted() -> None:
    """The filesystem store shares a sequence space, so gaps are normal."""
    state = replay(
        [
            _event(KnownEventType.ASSISTANT_DELTA, 3, uuid="a", text="one", complete=True),
            _event(KnownEventType.ASSISTANT_DELTA, 17, uuid="b", text="two", complete=True),
            _event(KnownEventType.ASSISTANT_DELTA, 402, uuid="c", text="three", complete=True),
        ]
    )
    assert [item.text for item in state.conversation] == ["one", "two", "three"]
    assert state.last_seq == 402


def test_terminal_buffer_is_bounded_and_says_so() -> None:
    from kolega_code.session import projection

    events = [
        _event(KnownEventType.TERMINAL_OUTPUT, index + 1, output="z" * 10_000)
        for index in range(projection.TERMINAL_BUFFER_CHARS // 10_000 + 5)
    ]
    state = replay(events)
    assert len(state.terminal) <= projection.TERMINAL_BUFFER_CHARS
    assert state.terminal_truncated is True


def test_state_is_json_serializable() -> None:
    events = _session_events()
    events[5].artifacts = [
        ArtifactRef(
            sha256="a" * 64,
            bytes=10,
            media_type="text/plain",
            purpose="tool_result",
            encoding="utf-8",
            chars=10,
        )
    ]
    payload = replay(events).to_dict()

    encoded = json.dumps(payload)
    assert '"conversation"' in encoded
    assert "_streams" not in encoded and "_tools" not in encoded, "internal fold indices must not be exposed to clients"
    tool = next(item for item in payload["conversation"] if item["kind"] == "tool")
    assert tool["artifacts"][0]["sha256"] == "a" * 64


def test_pending_prompt_is_visible_and_blocks_activity() -> None:
    """A replay must show where the agent stopped to ask, not just skip the gap."""
    state = replay(
        [
            _event(
                KnownEventType.CONTROL_REQUESTED,
                1,
                elapsed_ms=100,
                request_id="r1",
                kind="permission",
                payload={"command": "rm -rf build"},
            ),
        ]
    )
    assert len(state.prompts) == 1
    prompt = state.prompts[0]
    assert prompt.kind == "permission"
    assert prompt.payload == {"command": "rm -rf build"}
    assert prompt.resolved is False
    assert state.activity == "waiting_for_user"


def test_resolved_prompt_records_the_answer_and_how_it_settled() -> None:
    state = replay(
        [
            _event(KnownEventType.CONTROL_REQUESTED, 1, request_id="r1", kind="permission", payload={}),
            _event(
                KnownEventType.CONTROL_RESOLVED,
                2,
                request_id="r1",
                response={"allowed": False},
                reason="timeout",
            ),
        ]
    )
    (prompt,) = state.prompts
    assert prompt.resolved is True
    assert prompt.response == {"allowed": False}
    assert prompt.reason == "timeout", "how a prompt settled is part of the record"
    assert state.activity == "generating"


def test_a_second_open_prompt_keeps_the_session_waiting() -> None:
    state = replay(
        [
            _event(KnownEventType.CONTROL_REQUESTED, 1, request_id="r1", kind="permission", payload={}),
            _event(KnownEventType.CONTROL_REQUESTED, 2, request_id="r2", kind="question", payload={}),
            _event(KnownEventType.CONTROL_RESOLVED, 3, request_id="r1", response={}, reason="answered"),
        ]
    )
    assert state.activity == "waiting_for_user", "one answer must not clear a still-open prompt"
    assert [prompt.resolved for prompt in state.prompts] == [True, False]


def test_duplicate_request_announcement_is_idempotent() -> None:
    """A client catching up may see the same request twice; it is one prompt."""
    state = replay(
        [
            _event(KnownEventType.CONTROL_REQUESTED, 1, request_id="r1", kind="permission", payload={}),
            _event(KnownEventType.CONTROL_REQUESTED, 2, request_id="r1", kind="permission", payload={}),
        ]
    )
    assert len(state.prompts) == 1


def test_retention_marker_is_visible_in_the_transcript() -> None:
    state = replay(
        [
            _event(KnownEventType.ASSISTANT_DELTA, 1, uuid="a", text="before", complete=True),
            _event(KnownEventType.STREAM_TRUNCATED, 2, reason="retention_limit"),
        ]
    )
    assert state.recording_truncated is True
    assert state.conversation[-1].kind == "system"
    assert "retention limit" in state.conversation[-1].text
