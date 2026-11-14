from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from kolega_code.events import AgentEvent, ArtifactRef
from kolega_code.session.projection import replay
from tests.web.test_fold_parity import _run_js_fold

DELIVERY_EVENT = "background_job_delivered"
WORKER = {
    "dispatch_id": "background-worker",
    "agent_id": "source-agent",
    "agent_name": "general-agent",
    "parent_agent_id": "root",
    "parent_tool_call_id": "launch-call",
    "task": "Inspect the build",
    "background": True,
    "effective_routing": {"provider": "offline", "model": "worker-model"},
}
METADATA = {
    "parent_tool_call_id": "launch-call",
    "turn_id": "launch-turn",
    "effective_routing": {"provider": "offline", "model": "worker-model"},
    "workspace_id": "workspace-a",
    "thread_id": "thread-a",
    "project_path": "/example/project",
    "durable_extra": {"keep": ["existing", "metadata"]},
}
DELIVERY = {
    "job_id": "job-a",
    "completion_id": "completion-a",
    "agent_type": "general",
    "status": "completed",
    "task": "Inspect the build",
    "text": "Build verified.\nComplete report.",
    "disposition": "context",
    "epoch_id": "epoch-a",
    "metadata": METADATA,
}
EXPECTED_DELIVERY = {
    "kind": "background_delivery",
    "text": "Build verified.\nComplete report.",
    "complete": True,
    "seq": 6,
    "elapsed_ms": 500,
    "status": "completed",
    "background_delivery": {
        "job_id": "job-a",
        "completion_id": "completion-a",
        "agent_type": "general",
        "status": "completed",
        "task": "Inspect the build",
        "disposition": "context",
        "epoch_id": "epoch-a",
        "metadata": METADATA,
    },
}


def _event(
    event_type: str,
    seq: int,
    *,
    info: dict[str, Any] | None = None,
    **content: Any,
) -> AgentEvent:
    return AgentEvent(
        session_id="background-session",
        sender="agent",
        uuid=f"event-{seq}",
        seq=seq,
        elapsed_ms=(seq - 1) * 100,
        event_type=event_type,
        content=deepcopy(content),
        sub_agent_info=deepcopy(info),
    )


def _prefix_events() -> list[AgentEvent]:
    return [
        _event("turn_started", 1, turn_id="launch-turn", user_text="Start the worker"),
        _event("assistant_delta", 2, info=WORKER, text="Worker-only answer", complete=True),
        _event("chat_message", 3, info=WORKER, status="STOPPED", message="Worker finished"),
        _event("turn_ended", 4, turn_id="launch-turn", status="completed"),
        _event("turn_started", 5, turn_id="later-turn", user_text="A later request"),
        _event(DELIVERY_EVENT, 6, **DELIVERY),
        _event(DELIVERY_EVENT, 7, **{**DELIVERY, "text": "Duplicate must not replace the report"}),
        _event(
            DELIVERY_EVENT,
            8,
            **{
                **DELIVERY,
                "job_id": "job-b",
                "completion_id": "completion-b",
                "status": "interrupted",
                "text": "Archived report",
                "disposition": "archived",
                "epoch_id": None,
            },
        ),
        _event(
            DELIVERY_EVENT,
            9,
            job_id="legacy-job",
            completion_id="legacy-completion",
            agent_type="investigation",
            status="failed",
            task="Legacy job",
            text="Legacy report",
            disposition="context",
        ),
        _event(DELIVERY_EVENT, 10, info={"dispatch_id": "must-not-exist"}, **DELIVERY),
        _event(DELIVERY_EVENT, 11, **{**DELIVERY, "completion_id": "invalid", "status": "running"}),
    ]


@pytest.mark.parametrize("cut", range(12))
def test_literal_prefixes_and_javascript_parity(tmp_path: Path, cut: int) -> None:
    events = _prefix_events()[:cut]
    python = replay(events).to_dict()
    javascript = _run_js_fold([event.model_dump(mode="json") for event in events], tmp_path)
    assert javascript == python
    for state in (python, javascript):
        deliveries = [item for item in state["conversation"] if item["kind"] == "background_delivery"]
        assert len(deliveries) == (0 if cut < 6 else 1 if cut < 8 else 2 if cut < 9 else 3)
        assert list(state["sub_agents"]) == ([] if cut < 2 else ["background-worker"])
        assert state["unknown_event_types"] == []
        if cut >= 3:
            worker = state["sub_agents"]["background-worker"]
            assert worker["status"] == "completed"
            assert [item["text"] for item in worker["steps"]] == ["Worker-only answer"]
            assert worker["effective_routing"] == {"provider": "offline", "model": "worker-model"}
        if cut >= 6:
            assert deliveries[0] == EXPECTED_DELIVERY
            assert [turn["turn_id"] for turn in state["turns"]] == ["launch-turn", "later-turn"]
            assert state["activity"] == "generating"
            assert "sub_agent" not in deliveries[0]
        if cut >= 8:
            assert deliveries[1] == {
                **EXPECTED_DELIVERY,
                "text": "Archived report",
                "seq": 8,
                "elapsed_ms": 700,
                "status": "interrupted",
                "background_delivery": {
                    **EXPECTED_DELIVERY["background_delivery"],
                    "job_id": "job-b",
                    "completion_id": "completion-b",
                    "status": "interrupted",
                    "disposition": "archived",
                    "epoch_id": None,
                },
            }
        if cut >= 9:
            assert deliveries[2]["background_delivery"] == {
                "job_id": "legacy-job",
                "completion_id": "legacy-completion",
                "agent_type": "investigation",
                "status": "failed",
                "task": "Legacy job",
                "disposition": "context",
                "epoch_id": None,
                "metadata": {},
            }


@pytest.mark.parametrize(
    "changes",
    [
        {"job_id": ""},
        {"completion_id": ""},
        {"agent_type": ""},
        {"job_id": None},
        {"completion_id": 7},
        {"agent_type": []},
        {"status": "running"},
        {"status": "STOPPED"},
        {"task": {}},
        {"text": None},
        {"disposition": "delivered"},
        {"disposition": False},
        {"epoch_id": 42},
        {"metadata": []},
        {"metadata": None},
    ],
)
def test_invalid_delivery_does_not_reserve_completion_or_create_workers(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    invalid = _event(DELIVERY_EVENT, 1, **{**DELIVERY, **changes})
    valid = _event(DELIVERY_EVENT, 2, **DELIVERY)
    for events, count in (([invalid], 0), ([invalid, valid], 1)):
        python = replay(events).to_dict()
        javascript = _run_js_fold([event.model_dump(mode="json") for event in events], tmp_path)
        assert javascript == python
        for state in (python, javascript):
            assert len(state["conversation"]) == count
            assert state["sub_agents"] == {}


@pytest.mark.parametrize("field", [key for key in DELIVERY if key not in {"epoch_id", "metadata"}])
def test_missing_required_field_is_not_a_report(tmp_path: Path, field: str) -> None:
    content = deepcopy(DELIVERY)
    del content[field]
    event = _event(DELIVERY_EVENT, 1, **content)
    for state in (replay([event]).to_dict(), _run_js_fold([event.model_dump(mode="json")], tmp_path)):
        assert state["conversation"] == []
        assert state["sub_agents"] == {}


@pytest.mark.parametrize("info", [{}, {"dispatch_id": "unseen-worker"}])
def test_non_root_delivery_envelope_is_ignored(tmp_path: Path, info: dict[str, Any]) -> None:
    events = [_event(DELIVERY_EVENT, 1, info=info, **DELIVERY), _event(DELIVERY_EVENT, 2, **DELIVERY)]
    for cut, count in ((1, 0), (2, 1)):
        prefix = events[:cut]
        for state in (
            replay(prefix).to_dict(),
            _run_js_fold([event.model_dump(mode="json") for event in prefix], tmp_path),
        ):
            assert len(state["conversation"]) == count
            assert state["sub_agents"] == {}


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "interrupted"])
def test_all_terminal_statuses_are_delivery_not_worker_lifecycle(tmp_path: Path, status: str) -> None:
    event = _event(DELIVERY_EVENT, 1, **{**DELIVERY, "status": status, "text": "", "task": ""})
    ref = ArtifactRef(
        sha256="a" * 64, bytes=100, media_type="text/plain", purpose="tool_result", encoding="utf-8", chars=100
    )
    event.artifacts = [ref]
    python = replay([event]).to_dict()
    javascript = _run_js_fold([event.model_dump(mode="json")], tmp_path)
    assert javascript == python
    for state in (python, javascript):
        assert state["sub_agents"] == {}
        assert state["activity"] == "idle"
        assert state["conversation"][0]["status"] == status
        assert state["conversation"][0]["artifacts"] == [ref.model_dump(exclude_none=True)]


def test_delivery_metadata_is_a_durable_snapshot() -> None:
    event = _event(DELIVERY_EVENT, 1, **DELIVERY)
    state = replay([event])
    event.content["metadata"]["effective_routing"]["model"] = "mutated"
    assert state.to_dict()["conversation"][0]["background_delivery"]["metadata"]["effective_routing"] == {
        "provider": "offline",
        "model": "worker-model",
    }
