from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from kolega_code.events import AgentEvent, ArtifactRef
from kolega_code.session.projection import PresentationState, fold, replay
from tests.web.test_fold_parity import ASSET_DIR, _run_js_fold

EMPTY_STATE = {
    "conversation": [],
    "sub_agents": {},
    "turns": [],
    "terminal": "",
    "terminal_truncated": False,
    "logs": [],
    "edit_previews": [],
    "context": None,
    "compaction": None,
    "task_list": None,
    "usage": None,
    "queue": None,
    "prompts": [],
    "status": "",
    "activity": "idle",
    "open_browsers": [],
    "last_seq": 0,
    "elapsed_ms": 0,
    "recording_truncated": False,
    "unknown_event_types": [],
}
EMPTY_QUEUE = {"queue_id": "publisher-a", "revision": 0, "source": "tui", "reason": "snapshot", "items": []}
LOCAL_ITEM = {"queue_id": "entry-a", "text": "Next request", "image_count": 2, "origin": None}
PEER_ITEM = {
    "queue_id": "entry-b",
    "text": "Peer request",
    "image_count": 0,
    "origin": {"kind": "peer", "session_id": "peer-session", "title": "Peer title"},
}


def _event(
    seq: int,
    content: Any,
    *,
    event_type: str = "queue_updated",
    info: Any = None,
) -> AgentEvent:
    return AgentEvent.model_construct(
        session_id="queue-session",
        sender="agent",
        uuid=f"event-{seq}",
        seq=seq,
        elapsed_ms=seq * 10,
        event_type=event_type,
        content=deepcopy(content),
        sub_agent_info=deepcopy(info),
    )


def _wire(event: AgentEvent) -> dict[str, Any]:
    return {
        "uuid": event.uuid,
        "event_type": event.event_type,
        "content": event.content,
        "sub_agent_info": event.sub_agent_info,
        "seq": event.seq,
        "elapsed_ms": event.elapsed_ms,
    }


def _node(tmp_path: Path, payload: Any, script: str) -> Any:
    if shutil.which("node") is None:
        pytest.skip("node is required for JavaScript queue fold checks")
    data = tmp_path / "queue-input.json"
    harness = tmp_path / "queue-check.mjs"
    data.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    harness.write_text(
        f"""
import assert from "node:assert/strict";
import {{ readFileSync }} from "node:fs";
import {{ emptyState, fold, replay, toDict }} from {json.dumps(str(ASSET_DIR / "fold.js"))};
const input = JSON.parse(readFileSync(process.argv[2], "utf8"));
{script}
""",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(harness), str(data)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout or "null")


def _fixture() -> tuple[list[AgentEvent], list[dict[str, Any] | None]]:
    enqueued = {
        "queue_id": "publisher-a",
        "revision": 1,
        "source": "tui",
        "reason": "enqueued",
        "items": [LOCAL_ITEM, PEER_ITEM],
    }
    dequeued = {
        "queue_id": "publisher-a",
        "revision": 7,
        "source": "tui",
        "reason": "dequeued",
        "items": [PEER_ITEM],
    }
    removed = {"queue_id": "publisher-a", "revision": 8, "source": "tui", "reason": "removed", "items": []}
    cleared = {"queue_id": "publisher-a", "revision": 9, "source": "tui", "reason": "cleared", "items": []}
    second = {"queue_id": "publisher-b", "revision": 0, "source": "ask", "reason": "snapshot", "items": [LOCAL_ITEM]}
    third = {"queue_id": "publisher-c", "revision": 0, "source": "tui", "reason": "snapshot", "items": []}
    cases = [
        (_event(1, {"text": "legacy"}, event_type="unrelated_future_event"), None),
        (_event(2, EMPTY_QUEUE), EMPTY_QUEUE),
        (_event(3, enqueued), enqueued),
        (_event(4, {**enqueued, "items": [], "source": "ask"}), enqueued),
        (_event(5, EMPTY_QUEUE), enqueued),
        (_event(6, dequeued), dequeued),
        (_event(7, removed), removed),
        (_event(8, cleared), cleared),
        (_event(9, second, info={}), cleared),
        (_event(10, {**second, "items": [LOCAL_ITEM, LOCAL_ITEM]}), cleared),
        (_event(11, {**cleared, "revision": 10, "reason": "unknown"}), cleared),
        (_event(12, second), second),
        (_event(13, {**cleared, "revision": 9_007_199_254_740_991}), second),
        (_event(14, {**second, "items": []}), second),
        (_event(15, third), third),
        (_event(16, {**second, "revision": 100}), third),
        (_event(17, {**enqueued, "revision": 100}), third),
        (_event(18, {**third, "revision": 12, "reason": "cleared"}), {**third, "revision": 12, "reason": "cleared"}),
    ]
    return [event for event, _ in cases], [None, *(deepcopy(expected) for _, expected in cases)]


@pytest.mark.parametrize("cut", range(19))
def test_literal_state_at_every_fixture_prefix_and_seek(tmp_path: Path, cut: int) -> None:
    events, queues = _fixture()
    prefix = events[:cut]
    expected = {
        **deepcopy(EMPTY_STATE),
        "queue": queues[cut],
        "last_seq": cut,
        "elapsed_ms": cut * 10,
        "unknown_event_types": ["unrelated_future_event"] if cut else [],
    }
    incremental = PresentationState()
    for event in prefix:
        fold(incremental, event)
    assert incremental.to_dict() == expected
    assert replay(prefix).to_dict() == expected
    assert _run_js_fold([_wire(event) for event in prefix], tmp_path) == expected
    _node(
        tmp_path,
        {"events": [_wire(event) for event in prefix], "expected": expected},
        """
const state = emptyState();
for (const event of input.events) fold(state, event);
assert.deepEqual(toDict(state), input.expected);
""",
    )


def _malformed_contents() -> list[Any]:
    base = {**EMPTY_QUEUE, "queue_id": "publisher-b", "revision": 100, "items": [LOCAL_ITEM, PEER_ITEM]}
    missing = object()
    cases: list[Any] = [None, [], "", 0, True]

    def variants(path: tuple[str | int, ...], values: tuple[Any, ...], *, include_missing: bool = True) -> None:
        for value in (missing, *values) if include_missing else values:
            candidate = deepcopy(base)
            target: Any = candidate
            for key in path[:-1]:
                target = target[key]
            if value is missing:
                del target[path[-1]]
            else:
                target[path[-1]] = value
            cases.append(candidate)

    wrong_counts = (None, True, False, -1, 0.5, "1", [], {}, 9_007_199_254_740_992, 10**100)
    variants(("queue_id",), (None, "", 7, False, [], {}))
    variants(("revision",), wrong_counts)
    variants(("source",), (None, "", "gateway", "TUI", 1, [], {}))
    variants(("reason",), (None, "", "updated", "enqueue", 1, [], {}))
    variants(("items",), (None, "", {}, 1, True))
    for index in (0, 1):
        variants(("items", index), (None, [], "", 0, True), include_missing=False)
        variants(("items", index, "queue_id"), (None, "", 1, False, [], {}))
        variants(("items", index, "text"), (None, 1, False, [], {}))
        variants(("items", index, "image_count"), wrong_counts)
        variants(("items", index, "origin"), ([], "", 1, False, {}, {"kind": "local"}))
    variants(("items", 1, "origin", "kind"), (None, "", "local", "PEER", 1, [], {}))
    for name in ("session_id", "title"):
        variants(("items", 1, "origin", name), (None, 1, False, [], {}))
    cases.append({**base, "items": [LOCAL_ITEM, {**PEER_ITEM, "queue_id": "entry-a"}]})
    return cases


@pytest.mark.parametrize("recorded", [False, True])
def test_malformed_snapshots_are_atomic_and_do_not_retire_or_reserve_ids(tmp_path: Path, recorded: bool) -> None:
    malformed = _malformed_contents()
    events = [_event(1, EMPTY_QUEUE)] if recorded else []
    events.extend(_event(index, content) for index, content in enumerate(malformed, len(events) + 1))
    next_revision = {**EMPTY_QUEUE, "revision": 1}
    new_publisher = {**EMPTY_QUEUE, "queue_id": "publisher-b"}
    events.extend([_event(len(events) + 1, next_revision), _event(len(events) + 2, new_publisher)])
    expected_queues = [EMPTY_QUEUE if recorded else None] * (len(malformed) + int(recorded)) + [
        next_revision,
        new_publisher,
    ]
    state = PresentationState()
    for event, queue in zip(events, expected_queues):
        fold(state, event)
        assert state.to_dict() == {
            **EMPTY_STATE,
            "queue": queue,
            "last_seq": event.seq,
            "elapsed_ms": event.elapsed_ms,
        }
    _node(
        tmp_path,
        {"events": [_wire(event) for event in events], "queues": expected_queues, "empty": EMPTY_STATE},
        """
const state = emptyState();
for (let index = 0; index < input.events.length; index++) {
  const event = input.events[index];
  fold(state, event);
  assert.deepEqual(toDict(state), {
    ...input.empty, queue: input.queues[index], last_seq: event.seq, elapsed_ms: event.elapsed_ms,
  }, `validation prefix ${index + 1}`);
}
""",
    )


@pytest.mark.parametrize("info", [{}, {"dispatch_id": "worker", "background": True}, [], "", False, 0])
def test_only_null_root_envelopes_can_publish(tmp_path: Path, info: Any) -> None:
    child = _event(1, EMPTY_QUEUE, info=info)
    valid = _event(2, {**EMPTY_QUEUE, "source": "ask"})
    for prefix, expected_queue in (([child], None), ([child, valid], {**EMPTY_QUEUE, "source": "ask"})):
        expected = {
            **EMPTY_STATE,
            "queue": expected_queue,
            "last_seq": len(prefix),
            "elapsed_ms": len(prefix) * 10,
        }
        assert replay(prefix).to_dict() == expected
        assert _run_js_fold([_wire(event) for event in prefix], tmp_path) == expected


@pytest.mark.parametrize("value", [0.0, -0.0, 1.0, 1e3, 9_007_199_254_740_991.0])
def test_integral_json_numbers_have_the_same_canonical_queue(tmp_path: Path, value: float) -> None:
    content = {**EMPTY_QUEUE, "revision": value, "items": [{**LOCAL_ITEM, "image_count": value}]}
    expected_queue = {
        **EMPTY_QUEUE,
        "revision": int(value),
        "items": [{**LOCAL_ITEM, "image_count": int(value)}],
    }
    event = _event(1, content)
    expected = {**EMPTY_STATE, "queue": expected_queue, "last_seq": 1, "elapsed_ms": 10}
    assert _run_js_fold([_wire(event)], tmp_path) == expected
    assert replay([event]).to_dict() == expected
    queue = replay([event]).queue
    assert queue is not None
    assert type(queue["revision"]) is int and type(queue["items"][0]["image_count"]) is int


@pytest.mark.parametrize("field", ["revision", "image_count"])
def test_nonfinite_numbers_are_inert_in_both_folds(tmp_path: Path, field: str) -> None:
    baseline = {**EMPTY_QUEUE, "items": [LOCAL_ITEM]}
    state = replay([_event(1, baseline)])
    for seq, value in enumerate((float("nan"), float("inf"), float("-inf")), 2):
        content = deepcopy(baseline)
        if field == "revision":
            content[field] = value
        else:
            content["items"][0][field] = value
        fold(state, _event(seq, content))
        assert state.queue == baseline
        assert state.conversation == [] and state.sub_agents == {} and state.activity == "idle"
    _node(
        tmp_path,
        {"baseline": baseline, "field": field},
        """
const state = replay([{ event_type: "queue_updated", content: input.baseline }]);
for (const value of [NaN, Infinity, -Infinity]) {
  const content = structuredClone(input.baseline);
  if (input.field === "revision") content.revision = value;
  else content.items[0].image_count = value;
  fold(state, { event_type: "queue_updated", content });
  assert.deepEqual(state.queue, input.baseline);
  assert.deepEqual(toDict(state).conversation, []);
  assert.equal(state.subAgents.size, 0);
  assert.equal(state.activity, "idle");
}
""",
    )


def test_canonical_snapshot_is_defensively_copied_and_historical_text_is_preserved(tmp_path: Path) -> None:
    expected = {
        "queue_id": "publisher-a",
        "revision": 9_007_199_254_740_991,
        "source": "ask",
        "reason": "enqueued",
        "items": [
            {
                "queue_id": "entry-a",
                "text": "  Historical\npreview " + "x" * 200,
                "image_count": 9_007_199_254_740_991,
                "origin": {"kind": "peer", "session_id": "", "title": ""},
            },
            {"queue_id": "entry-b", "text": "", "image_count": 0, "origin": None},
        ],
    }
    content = deepcopy(expected)
    content["extra"] = {"private": ["not presentation"]}
    content["items"][0]["extra"] = {"image": "not hydrated"}
    content["items"][0]["origin"]["extra"] = {"private": True}
    event = _event(1, content)
    state = replay([event])
    assert state.queue == expected
    assert state.queue is not None
    assert state.queue is not event.content
    assert state.queue["items"] is not event.content["items"]
    assert state.queue["items"][0]["origin"] is not event.content["items"][0]["origin"]
    event.content["items"][0]["origin"]["title"] = "mutated event"
    event.content["items"][0]["text"] = "mutated event"
    event.content["items"].clear()
    assert state.queue == expected
    exported = state.to_dict()
    exported["queue"]["items"][0]["origin"]["title"] = "mutated export"
    exported["queue"]["items"].clear()
    assert state.to_dict()["queue"] == expected
    _node(
        tmp_path,
        {"event": _wire(_event(1, content)), "expected": expected},
        """
const state = replay([input.event]);
assert.deepEqual(state.queue, input.expected);
assert.notEqual(state.queue, input.event.content);
assert.notEqual(state.queue.items, input.event.content.items);
assert.notEqual(state.queue.items[0].origin, input.event.content.items[0].origin);
input.event.content.items[0].origin.title = "mutated event";
input.event.content.items.length = 0;
assert.deepEqual(state.queue, input.expected);
const exported = toDict(state);
exported.queue.items[0].origin.title = "mutated export";
exported.queue.items.length = 0;
assert.deepEqual(toDict(state).queue, input.expected);
""",
    )


def test_queue_events_leave_existing_turns_workers_activity_and_accounting_unchanged(tmp_path: Path) -> None:
    prefix = [
        _event(1, {"turn_id": "turn-a", "user_text": "Actual user"}, event_type="turn_started"),
        _event(
            2,
            {"text": "Worker answer", "complete": True},
            event_type="assistant_delta",
            info={"dispatch_id": "worker", "agent_name": "general"},
        ),
        _event(3, {"status": "STOPPED"}, event_type="chat_message", info={"dispatch_id": "worker"}),
        _event(4, {"text": "Actual answer", "complete": False}, event_type="thinking_delta"),
    ]
    event = _event(5, {**EMPTY_QUEUE, "items": [LOCAL_ITEM, PEER_ITEM]})
    event.artifacts = [
        ArtifactRef(sha256="a" * 64, bytes=500, media_type="image/png", purpose="image", encoding="base64")
    ]
    state = replay(prefix)
    state.usage = {"sentinel": "unchanged accounting"}
    baseline = deepcopy(state.to_dict())
    assert baseline["activity"] == "generating"
    assert len(baseline["turns"]) == 1
    assert [item["kind"] for item in baseline["conversation"]] == ["user", "thinking"]
    assert baseline["sub_agents"]["worker"]["status"] == "completed"
    fold(state, event)
    expected = {**baseline, "queue": {**EMPTY_QUEUE, "items": [LOCAL_ITEM, PEER_ITEM]}, "last_seq": 5, "elapsed_ms": 50}
    assert state.to_dict() == expected
    _node(
        tmp_path,
        {"prefix": [_wire(item) for item in prefix], "event": event.model_dump(mode="json"), "expected": expected},
        """
const state = replay(input.prefix);
state.usage = { sentinel: "unchanged accounting" };
fold(state, input.event);
assert.deepEqual(toDict(state), input.expected);
""",
    )


def test_legacy_recording_does_not_infer_queue_from_user_messages(tmp_path: Path) -> None:
    events = [
        _event(1, {"turn_id": "turn-a", "user_text": "queued?"}, event_type="turn_started"),
        _event(2, {"turn_id": "turn-a", "status": "completed"}, event_type="turn_ended"),
        _event(3, {"queue_id": "old-format", "items": []}, event_type="future_queue_event"),
    ]
    for cut in range(len(events) + 1):
        for state in (replay(events[:cut]).to_dict(), _run_js_fold([_wire(event) for event in events[:cut]], tmp_path)):
            assert state["queue"] is None
            assert state["unknown_event_types"] == (["future_queue_event"] if cut == 3 else [])
            assert state["activity"] == ("generating" if cut == 1 else "idle")
