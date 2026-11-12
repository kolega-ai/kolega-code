from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from kolega_code.events import AgentEvent, ArtifactRef
from kolega_code.session.projection import PresentationState, fold, replay
from tests.web.test_fold_parity import _run_js_fold
from tests.web.test_queue_fold import _node, _wire

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
    "goal": None,
    "prompts": [],
    "status": "",
    "activity": "idle",
    "open_browsers": [],
    "last_seq": 0,
    "elapsed_ms": 0,
    "recording_truncated": False,
    "unknown_event_types": [],
}
GOAL = {
    "condition": "Verify the full change\n  Preserve literal 測試 e\u0301 **text**",
    "started_at": "2026-11-12T10:00:00+00:00",
    "turns_evaluated": 0,
    "tokens_spent": 0,
    "last_reason": "",
    "last_evaluated_at": None,
    "max_turns": 10,
    "run_to_completion": False,
    "paused": False,
    "met": False,
    "status_note": "",
}
ABSENT = {
    "generation_id": "generation-a",
    "revision": 0,
    "source": "tui",
    "reason": "snapshot",
    "goal": None,
    "evaluation_met": None,
}
STARTED = {**ABSENT, "revision": 1, "reason": "started", "goal": GOAL}
MAX_SAFE = 9_007_199_254_740_991


def _event(seq: int, content: Any, *, info: Any = None, event_type: str = "goal_updated") -> AgentEvent:
    return AgentEvent.model_construct(
        session_id="goal-session",
        sender="agent",
        uuid=f"goal-event-{seq}",
        seq=seq,
        elapsed_ms=seq * 10,
        event_type=event_type,
        content=deepcopy(content),
        sub_agent_info=deepcopy(info),
    )


def test_literal_complete_state_at_every_prefix_incremental_and_seek(tmp_path: Path) -> None:
    evaluated = {
        **STARTED,
        "revision": 2,
        "reason": "evaluated",
        "goal": {
            **GOAL,
            "turns_evaluated": 1,
            "tokens_spent": 87,
            "last_reason": "One check remains.",
            "last_evaluated_at": "2026-11-12T10:00:10+00:00",
        },
        "evaluation_met": False,
    }
    paused = {**evaluated, "revision": 3, "reason": "paused", "goal": {**evaluated["goal"], "paused": True}}
    resumed = {**evaluated, "revision": 4, "reason": "resumed"}
    verdict = {
        **evaluated,
        "revision": 5,
        "goal": {**evaluated["goal"], "turns_evaluated": 2, "run_to_completion": True},
        "evaluation_met": True,
    }
    completed = {**verdict, "revision": 6, "reason": "completed", "goal": {**verdict["goal"], "met": True}}
    cleared = {**ABSENT, "revision": 7, "reason": "cleared"}
    replaced = {**STARTED, "revision": 8, "source": "ask", "reason": "replaced", "goal": {**GOAL, "condition": ""}}
    ended_unmet = {**replaced, "revision": 9, "reason": "completed", "evaluation_met": False}
    second = {**ABSENT, "generation_id": "generation-b", "source": "ask"}
    third = {**STARTED, "generation_id": "generation-c", "revision": 0, "reason": "snapshot"}
    cases = [
        (ABSENT, ABSENT, None),
        (STARTED, STARTED, None),
        ({**STARTED, "goal": None}, STARTED, None),
        ({**ABSENT, "revision": 0}, STARTED, None),
        (evaluated, evaluated, None),
        (paused, paused, None),
        (resumed, resumed, None),
        (verdict, verdict, None),
        (completed, completed, None),
        (cleared, cleared, None),
        (replaced, replaced, None),
        (ended_unmet, ended_unmet, None),
        (second, ended_unmet, {}),
        ({**second, "goal": {}}, ended_unmet, None),
        (second, second, None),
        ({**ended_unmet, "revision": MAX_SAFE}, second, None),
        (third, third, None),
        ({**second, "revision": 100}, third, None),
        ({**completed, "revision": MAX_SAFE}, third, None),
        (
            {**third, "revision": 1, "reason": "resumed", "goal": {**GOAL, "paused": True}},
            {**third, "revision": 1, "reason": "resumed", "goal": {**GOAL, "paused": True}},
            None,
        ),
    ]
    events = [_event(seq, content, info=info) for seq, (content, _, info) in enumerate(cases, 1)]
    expected = [
        deepcopy(EMPTY_STATE),
        *[
            {**deepcopy(EMPTY_STATE), "goal": deepcopy(goal), "last_seq": seq, "elapsed_ms": seq * 10}
            for seq, (_, goal, _) in enumerate(cases, 1)
        ],
    ]
    state = PresentationState()
    for cut, wanted in enumerate(expected):
        if cut:
            fold(state, events[cut - 1])
        assert state.to_dict() == wanted, f"Python incremental prefix {cut}"
        assert replay(events[:cut]).to_dict() == wanted, f"Python seek prefix {cut}"
    _node(
        tmp_path,
        {"events": [_wire(event) for event in events], "expected": expected},
        """
const state = emptyState();
for (let cut = 0; cut <= input.events.length; cut++) {
  if (cut) fold(state, input.events[cut - 1]);
  assert.deepEqual(toDict(state), input.expected[cut], `incremental prefix ${cut}`);
  assert.deepEqual(toDict(replay(input.events.slice(0, cut))), input.expected[cut], `seek prefix ${cut}`);
}
""",
    )


def _malformed() -> list[Any]:
    base = {**STARTED, "generation_id": "generation-b", "revision": 100}
    missing = object()
    cases: list[Any] = [None, [], "", 0, True]

    def variants(path: tuple[str, ...], values: tuple[Any, ...]) -> None:
        for value in (missing, *values):
            candidate = deepcopy(base)
            target = candidate if len(path) == 1 else candidate["goal"]
            if value is missing:
                del target[path[-1]]
            else:
                target[path[-1]] = value
            cases.append(candidate)

    wrong_counts = (None, True, False, -1, 0.5, "1", [], {}, MAX_SAFE + 1, 10**100)
    variants(("generation_id",), (None, "", 7, False, [], {}))
    variants(("revision",), wrong_counts)
    variants(("source",), (None, "", "gateway", "TUI", 1, [], {}))
    variants(("reason",), (None, "", "updated", "complete", 1, [], {}))
    variants(("evaluation_met",), (0, 1, "", "false", [], {}))
    variants(("goal",), (None, [], "", 0, True))
    for field in ("condition", "started_at", "last_reason", "status_note"):
        variants(("goal", field), (None, 1, False, [], {}))
    for field in ("turns_evaluated", "tokens_spent", "max_turns"):
        variants(("goal", field), wrong_counts)
    for field in ("run_to_completion", "paused", "met"):
        variants(("goal", field), (None, 0, 1, "", "false", [], {}))
    variants(("goal", "last_evaluated_at"), (0, True, [], {}))
    cases.extend(
        {**base, "reason": reason, "goal": None}
        for reason in ("started", "replaced", "evaluated", "paused", "resumed", "completed")
    )
    cases.append({**base, "reason": "cleared"})
    return cases


@pytest.mark.parametrize("recorded", [False, True])
@pytest.mark.parametrize("generation", ["generation-a", "generation-b"])
def test_malformed_updates_neither_retire_generations_nor_reserve_revisions(
    tmp_path: Path, recorded: bool, generation: str
) -> None:
    malformed = _malformed()
    for content in malformed:
        if isinstance(content, dict) and content.get("generation_id") == "generation-b":
            content["generation_id"] = generation
    events = [_event(1, ABSENT)] if recorded else []
    events.extend(_event(seq, content) for seq, content in enumerate(malformed, len(events) + 1))
    second = {**ABSENT, "generation_id": "generation-b"}
    events.extend([_event(len(events) + 1, STARTED), _event(len(events) + 2, second)])
    goals = [ABSENT if recorded else None] * (len(malformed) + int(recorded)) + [STARTED, second]
    state = PresentationState()
    for event, goal in zip(events, goals):
        fold(state, event)
        assert state.to_dict() == {
            **EMPTY_STATE,
            "goal": goal,
            "last_seq": event.seq,
            "elapsed_ms": event.elapsed_ms,
        }
    _node(
        tmp_path,
        {"events": [_wire(event) for event in events], "goals": goals, "empty": EMPTY_STATE},
        """
const state = emptyState();
for (let index = 0; index < input.events.length; index++) {
  const event = input.events[index];
  fold(state, event);
  assert.deepEqual(toDict(state), {
    ...input.empty, goal: input.goals[index], last_seq: event.seq, elapsed_ms: event.elapsed_ms,
  }, `malformed prefix ${index}`);
}
""",
    )


@pytest.mark.parametrize("info", [{}, {"dispatch_id": "worker", "background": True}, [], "", False, 0])
def test_only_null_root_envelopes_publish_without_creating_children(tmp_path: Path, info: Any) -> None:
    events = [_event(1, STARTED, info=info), _event(2, ABSENT)]
    for cut, goal in ((1, None), (2, ABSENT)):
        expected = {**EMPTY_STATE, "goal": goal, "last_seq": cut, "elapsed_ms": cut * 10}
        assert replay(events[:cut]).to_dict() == expected
        assert _run_js_fold([_wire(event) for event in events[:cut]], tmp_path) == expected


@pytest.mark.parametrize("value", [0.0, -0.0, 1.0, 1e3, float(MAX_SAFE)])
def test_safe_integral_json_numbers_are_canonical_in_every_counter(tmp_path: Path, value: float) -> None:
    goal = {**GOAL, "turns_evaluated": value, "tokens_spent": value, "max_turns": value}
    event = _event(1, {**STARTED, "revision": value, "goal": goal})
    canonical = {**GOAL, "turns_evaluated": int(value), "tokens_spent": int(value), "max_turns": int(value)}
    expected_goal = {**STARTED, "revision": int(value), "goal": canonical}
    expected = {**EMPTY_STATE, "goal": expected_goal, "last_seq": 1, "elapsed_ms": 10}
    state = replay([event])
    assert state.to_dict() == expected
    assert state.goal is not None and type(state.goal["revision"]) is int
    assert all(type(state.goal["goal"][field]) is int for field in ("turns_evaluated", "tokens_spent", "max_turns"))
    assert _run_js_fold([_wire(event)], tmp_path) == expected


@pytest.mark.parametrize("field", ["revision", "turns_evaluated", "tokens_spent", "max_turns"])
def test_nonfinite_numbers_are_inert_in_both_folds(tmp_path: Path, field: str) -> None:
    state = replay([_event(1, STARTED)])
    for seq, value in enumerate((float("nan"), float("inf"), float("-inf")), 2):
        content = deepcopy(STARTED)
        if field != "revision":
            content["revision"] = seq
        target = content if field == "revision" else content["goal"]
        target[field] = value
        fold(state, _event(seq, content))
        assert state.to_dict() == {**EMPTY_STATE, "goal": STARTED, "last_seq": seq, "elapsed_ms": seq * 10}
    _node(
        tmp_path,
        {"baseline": STARTED, "field": field},
        """
const state = replay([{ event_type: "goal_updated", content: input.baseline }]);
for (const value of [NaN, Infinity, -Infinity]) {
  const content = structuredClone(input.baseline);
  if (input.field !== "revision") content.revision = state.goal.revision + 1;
  const target = input.field === "revision" ? content : content.goal;
  target[input.field] = value;
  fold(state, { event_type: "goal_updated", content });
  assert.deepEqual(state.goal, input.baseline);
  assert.equal(state.subAgents.size, 0);
  assert.deepEqual(state.conversation, []);
}
""",
    )


def test_defined_fields_only_copy_isolation_and_literal_historical_text(tmp_path: Path) -> None:
    expected = {
        **STARTED,
        "revision": MAX_SAFE,
        "goal": {**GOAL, "condition": "  " + "full goal\n" * 200, "last_evaluated_at": "", "status_note": "\n"},
    }
    content = deepcopy(expected)
    content["extra"] = {"private": ["ignore"]}
    content["goal"]["extra"] = {"private": ["ignore"]}
    event = _event(1, content)
    state = replay([event])
    assert state.goal == expected
    assert state.goal is not None
    assert state.goal is not event.content and state.goal["goal"] is not event.content["goal"]
    event.content["goal"]["condition"] = "mutated event"
    event.content.clear()
    assert state.goal == expected
    exported = state.to_dict()
    exported["goal"]["goal"]["condition"] = "mutated export"
    exported["goal"].clear()
    assert state.to_dict()["goal"] == expected
    _node(
        tmp_path,
        {"event": _wire(_event(1, content)), "expected": expected},
        """
const state = replay([input.event]);
assert.deepEqual(state.goal, input.expected);
assert.notEqual(state.goal, input.event.content);
assert.notEqual(state.goal.goal, input.event.content.goal);
input.event.content.goal.condition = "mutated event";
delete input.event.content.goal;
assert.deepEqual(state.goal, input.expected);
const exported = toDict(state);
exported.goal.goal.condition = "mutated export";
delete exported.goal.goal;
assert.deepEqual(toDict(state).goal, input.expected);
assert.equal("retiredGoalGenerationIds" in toDict(state), false);
""",
    )
    assert "_retired_goal_generation_ids" not in state.to_dict()


def test_goal_observations_leave_turns_workers_activity_and_other_rails_unchanged(tmp_path: Path) -> None:
    prefix = [
        _event(1, {"turn_id": "turn-a", "user_text": "Actual user"}, event_type="turn_started"),
        _event(
            2,
            {"text": "Worker answer", "complete": True},
            event_type="assistant_delta",
            info={"dispatch_id": "worker", "agent_name": "general"},
        ),
        _event(3, {"status": "STOPPED"}, event_type="chat_message", info={"dispatch_id": "worker"}),
        _event(4, {"text": "Actual thinking", "complete": False}, event_type="thinking_delta"),
    ]
    event = _event(5, STARTED)
    event.artifacts = [
        ArtifactRef(sha256="a" * 64, bytes=500, media_type="image/png", purpose="image", encoding="base64")
    ]
    state = replay(prefix)
    state.usage = {"sentinel": "unchanged accounting"}
    state.queue = {"sentinel": "unchanged queue"}
    state.task_list = {"sentinel": "unchanged tasks"}
    baseline = deepcopy(state.to_dict())
    assert baseline["activity"] == "generating"
    assert len(baseline["turns"]) == 1
    assert [item["kind"] for item in baseline["conversation"]] == ["user", "thinking"]
    assert baseline["sub_agents"]["worker"]["status"] == "completed"
    fold(state, event)
    expected = {**baseline, "goal": STARTED, "last_seq": 5, "elapsed_ms": 50}
    assert state.to_dict() == expected
    _node(
        tmp_path,
        {"prefix": [_wire(item) for item in prefix], "event": event.model_dump(mode="json"), "expected": expected},
        """
const state = replay(input.prefix);
state.usage = { sentinel: "unchanged accounting" };
state.queue = { sentinel: "unchanged queue" };
state.taskList = { sentinel: "unchanged tasks" };
fold(state, input.event);
assert.deepEqual(toDict(state), input.expected);
""",
    )


def test_no_goal_inference_from_legacy_chat_semantic_tool_status_or_children(tmp_path: Path) -> None:
    events = [
        _event(1, {"turn_id": "turn-a", "user_text": "/goal ship"}, event_type="turn_started"),
        _event(2, {"message_type": "goal_updated", **STARTED}, event_type="chat_message"),
        _event(3, {"goal": GOAL, "evaluation_met": True, "text": "Goal met"}, event_type="assistant_delta"),
        _event(
            4, {"message_type": "tool_result", "text": "Goal met", "tool_call_id": "call"}, event_type="chat_message"
        ),
        _event(5, {"message": "Goal paused", "goal": GOAL}, event_type="status_update"),
        _event(6, STARTED, info={"dispatch_id": "not-a-worker"}),
        _event(7, {"turn_id": "turn-a", "status": "completed"}, event_type="turn_ended"),
    ]
    for cut in range(len(events) + 1):
        for state in (replay(events[:cut]).to_dict(), _run_js_fold([_wire(e) for e in events[:cut]], tmp_path)):
            assert state["goal"] is None
            assert state["sub_agents"] == {}
            assert len(state["turns"]) == int(cut > 0)
