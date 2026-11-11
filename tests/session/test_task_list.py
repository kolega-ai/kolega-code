from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.session.projection import PresentationState, ProjectionError, fold, replay
from kolega_code.session.task_list import task_list_entries

FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures" / "task_list.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", FIXTURE["parser_cases"], ids=lambda case: case["name"])
def test_task_list_entries(case: dict) -> None:
    assert task_list_entries(case["markdown"]) == case["entries"]


@pytest.mark.parametrize(
    "prefix,status",
    [("- [x]", "completed"), ("- [X]", "completed"), ("- [ ]", "pending"), ("-", "pending"), ("", "pending")],
)
@pytest.mark.parametrize("legacy", [False, True])
def test_task_list_text_is_never_truncated(prefix: str, status: str, legacy: bool) -> None:
    text = '完成 🐍 <tag> & "quoted" é ' * 100
    text = text.strip()
    markdown = f" \t{prefix} {text}\n "
    expected = {"markdown": markdown, "entries": [{"text": text, "status": status}]}
    assert task_list_entries(markdown) == expected["entries"]
    event = AgentEvent(
        sender="agent",
        event_type=KnownEventType.CHAT_MESSAGE if legacy else KnownEventType.TASK_LIST_UPDATED,
        content={"message_type": "task_list_update", "text": markdown} if legacy else {"markdown": markdown},
    )
    state = replay([event])
    assert state.task_list == expected
    assert state.to_dict()["task_list"] == expected
    assert state.conversation == []


def test_task_list_independent_expected_state_at_every_prefix() -> None:
    expected = deepcopy(FIXTURE["initial"])
    state = PresentationState()
    events = []
    assert state.task_list is None
    assert state.to_dict() == expected
    assert replay([]).to_dict() == expected
    for cut, step in enumerate(FIXTURE["steps"], 1):
        event = AgentEvent(session_id="s", sender="agent", **step["event"])
        events.append(event)
        expected.update(deepcopy(step["expect"]))
        assert fold(state, event) is state
        assert state.to_dict() == expected, f"incremental prefix {cut}"
        assert replay(events).to_dict() == expected, f"seek prefix {cut}"


@pytest.mark.parametrize(
    "event_type,content",
    [
        ("task_list_updated", {"markdown": "- replacement"}),
        ("task_list_updated", {"markdown": None}),
        ("chat_message", {"message_type": "task_list_update", "text": ""}),
        ("chat_message", {"message_type": "task_list_update", "text": False}),
    ],
)
@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("seq", [1, 3])
def test_task_updates_reject_repeated_and_out_of_order_seq(
    event_type: str, content: dict, worker: bool, seq: int
) -> None:
    state = replay(
        [
            AgentEvent(
                sender="agent", event_type=KnownEventType.TASK_LIST_UPDATED, content={"markdown": "- known"}, seq=3
            )
        ]
    )
    expected = deepcopy(state.to_dict())
    event = AgentEvent(
        sender="agent",
        event_type=event_type,
        content=content,
        sub_agent_info={"agent_id": "ignored-worker"} if worker else None,
        seq=seq,
        elapsed_ms=100,
    )
    with pytest.raises(ProjectionError, match="ascending order"):
        fold(state, event)
    assert state.to_dict() == expected
