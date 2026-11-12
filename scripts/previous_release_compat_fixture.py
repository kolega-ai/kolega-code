"""Current-checkout fixture for the previous-release settings/session round trip.

The producer guard scans Python ASTs for journal append calls and the recorder's
status f-strings, plus session_usage's LLM constants. It cannot discover event
types constructed indirectly outside those patterns or in non-Python producers.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import kolega_code
from kolega_code.cli import session_usage
from kolega_code.cli.session_event_store import FileArtifactStore, FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import SettingsStore
from kolega_code.events import AgentEvent, ArtifactPurpose, ArtifactRef, KnownEventType
from kolega_code.llm.models import ImageBlock, Message, TextBlock
from kolega_code.session.projection import replay


_SEMANTIC_TYPES = {
    "session.created",
    "session.metadata_updated",
    "session.workspace_switched",
    "context.epoch_started",
    "context.system",
    "context.tools",
    "context.message",
    "context.message_archived",
    "context.compacted",
    "context.rewound",
    "turn.started",
    "turn.completed",
    "turn.failed",
    "turn.cancelled",
    "assistant.message",
    "tool.results",
    "background.job.submitted",
    "background.job.terminal",
    "run.completed",
    "run.failed",
    "run.cancelled",
    "skill.activated",
    "goal.evaluated",
    "goal.completed",
    "loop.iteration_started",
    "loop.sleeping",
    "loop.completed",
    "agent.started",
    "agent.completed",
    "agent.failed",
    "llm.message",
    "llm.request_failed",
    "llm.run_started",
    "llm.request_started",
    "llm.request_settled",
    "llm.usage_observed",
    "llm.run_finished",
}
_DYNAMIC_CALLS = {
    ("cli/session_journal.py", "event_type"),
    ("cli/session_journal.py", "pending.event_type"),
    ("cli/session_journal.py", "f'run.{status}'"),
    ("cli/session_journal.py", "f'agent.{status}'"),
    ("cli/session_usage.py", "item.kind"),
    ("cli/session_event_store.py", "UI_EVENT_PREFIX + event.event_type"),
}
_STATUSES = {
    "finish_turn": {"completed", "failed", "cancelled"},
    "record_run_terminal": {"completed", "failed", "cancelled"},
    "record_agent_terminal": {"completed", "failed"},
}
_UNKNOWN_SEMANTIC = "compat.future_semantic"
_UNKNOWN_UI = "compat_future_ui"
_SNAPSHOT = "compat-original-events.jsonl"
_TASK_LIST_INITIAL: dict[str, Any] = {
    "markdown": "- [ ] Original task list",
    "compat_payload": {"preserve": [1, False, None, {"text": "future café 🐍 <literal>"}]},
}
_TASK_LIST_TEXT = "Preserve the full task text — café 🐍 " + "without truncation " * 40 + "COMPAT END"
_TASK_LIST: dict[str, Any] = {
    "markdown": f"## Session tasks\nKeep this source note.\n\n- [x] Read the requirements\n- [ ] {_TASK_LIST_TEXT}\n",
    "entries": [
        {"text": "Read the requirements", "status": "completed"},
        {"text": _TASK_LIST_TEXT, "status": "pending"},
    ],
}


def _usage_payload(*, populated: bool) -> dict[str, Any]:
    usage = session_usage.derive_session_usage(())
    if populated:
        usage.update(requests=2, responses=2, reported=2)
        usage["known_requests"] = {name: 2 for name in usage["known_requests"]}
        usage["known_fields"] = copy.deepcopy(usage["known_requests"])
        usage["costs"].update(
            reported=1,
            estimated=1,
            currencies={
                "USD": {
                    "reported": "0.000000000000000000123456789",
                    "estimated": "0",
                    "total": "0.000000000000000000123456789",
                    "requests": 1,
                    "reported_requests": 1,
                    "estimated_requests": 0,
                }
            },
            non_monetary={
                "api_credits": {
                    "reported": "0",
                    "estimated": "0.002",
                    "total": "0.002",
                    "requests": 1,
                    "reported_requests": 0,
                    "estimated_requests": 1,
                }
            },
        )
    return {
        "scope": "session",
        "semantics": "snapshot",
        "source_seq": int(populated),
        "usage": usage,
        "compat_payload": {"preserve": [False, None, "future café 🐍 <literal>"]},
    }


_IMAGE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_SUB_AGENTS = (
    {
        "info": {
            "agent_id": "compat-parent",
            "agent_name": "general",
            "parent_agent_id": "root",
            "parent_tool_call_id": "compat-dispatch-parent",
            "depth": 1,
            "background": True,
            "requested_routing": None,
            "effective_routing": {"provider": "openai", "model": "compat-child-model", "thinking_effort": "high"},
        },
        "command": "printf parent",
        "output": "parent output\n",
        "tool_call_id": "compat-parent-shell",
        "total_tokens": 37,
    },
    {
        "info": {
            "agent_id": "compat-nested",
            "agent_name": "general",
            "parent_agent_id": "compat-parent",
            "parent_tool_call_id": "compat-dispatch-nested",
            "depth": 2,
            "background": False,
            "requested_routing": {"provider": "openai", "model": "compat-nested-model", "thinking_effort": None},
            "effective_routing": {"provider": "openai", "model": "compat-nested-model", "thinking_effort": None},
        },
        "command": "printf nested",
        "output": "nested output\n",
        "tool_call_id": "compat-nested-shell",
        "total_tokens": 0,
    },
)


def _background_deliveries(epoch_id: str, turn_id: str, root: Path) -> list[dict[str, Any]]:
    return [
        {
            "job_id": job_id,
            "completion_id": f"compat-completion-{status}",
            "agent_type": "general",
            "status": status,
            "task": f"Inspect the {status} background task — café 🐍",
            "text": f"{status} report — café 🐍 <literal>\n" + "Preserve the complete report. " * 40 + f"{status} END",
            "disposition": disposition,
            "epoch_id": epoch_id,
            "metadata": (
                {
                    "parent_tool_call_id": "compat-dispatch-parent",
                    "turn_id": turn_id,
                    "effective_routing": copy.deepcopy(_SUB_AGENTS[0]["info"]["effective_routing"]),
                    "project_path": str(root.resolve()),
                    "workspace_id": "compat-workspace",
                    "thread_id": "compat-thread",
                    "compat_future_metadata": {"preserve": [False, None, "future café 🐍"]},
                }
                if status == "completed"
                else {}
            ),
            "compat_payload": {"preserve": [False, None, {"text": "future café 🐍 <literal>"}]},
        }
        for job_id, status, disposition in (
            ("compat-parent", "completed", "context"),
            ("compat-failed", "failed", "context"),
            ("compat-cancelled", "cancelled", "context"),
            ("compat-archived", "interrupted", "archived"),
        )
    ]


def _producer_inventory() -> set[str]:
    package = Path(kolega_code.__file__).resolve().parent
    found: set[str] = set()
    dynamics: set[tuple[str, str]] = set()
    recorder_tree: ast.Module | None = None
    for path in package.rglob("*.py"):
        relative = path.relative_to(package).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if relative == "cli/session_journal.py":
            recorder_tree = tree
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            receiver = ast.unparse(node.func.value)
            is_journal = node.func.attr == "append" and (
                "journal" in receiver or (relative == "cli/session_journal.py" and receiver == "self")
            )
            is_recorder = relative == "cli/session_journal.py" and receiver == "self" and node.func.attr == "_append"
            if not (is_journal or is_recorder):
                continue
            arg = (
                node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg == "event_type"), None)
            )
            assert arg is not None, f"Unreviewed journal call in {relative}:{node.lineno}"
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.add(arg.value)
            elif isinstance(arg, ast.Name) and arg.id.startswith("LLM_") and arg.id.endswith("_EVENT"):
                found.add(getattr(session_usage, arg.id))
            else:
                dynamics.add((relative, ast.unparse(arg)))
    assert dynamics == _DYNAMIC_CALLS, f"Unreviewed dynamic journal producers: {dynamics ^ _DYNAMIC_CALLS}"
    assert recorder_tree is not None
    funcs = {n.name: n for n in ast.walk(recorder_tree) if isinstance(n, ast.FunctionDef)}
    for name, expected in _STATUSES.items():
        func = funcs[name]
        status_sets = [
            {element.value for element in n.comparators[0].elts if isinstance(element, ast.Constant)}
            for n in ast.walk(func)
            if isinstance(n, ast.Compare)
            and isinstance(n.left, ast.Name)
            and n.left.id == "status"
            and len(n.comparators) == 1
            and isinstance(n.comparators[0], (ast.Set, ast.Tuple))
        ]
        assert status_sets == [expected], f"Review {name} status cases: {status_sets}"
        prefix = {"finish_turn": "turn", "record_run_terminal": "run", "record_agent_terminal": "agent"}[name]
        found.update(f"{prefix}.{status}" for status in expected)
    turn_formats = {
        ast.unparse(n)
        for n in ast.walk(funcs["finish_turn"])
        if isinstance(n, ast.JoinedStr) and ast.unparse(n).startswith("f'turn.")
    }
    assert turn_formats == {"f'turn.{status}'"}, f"Review turn event formats: {turn_formats}"
    usage_types = {
        value
        for name, value in vars(session_usage).items()
        if name.startswith("LLM_") and name.endswith("_EVENT") and isinstance(value, str)
    }
    usage_tree = ast.parse((package / "cli/session_usage.py").read_text(encoding="utf-8"))
    usage_ast = {
        target.id: node.value.value
        for node in ast.walk(usage_tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.startswith("LLM_") and target.id.endswith("_EVENT")
    }
    runtime_usage = {
        name: value
        for name, value in vars(session_usage).items()
        if name.startswith("LLM_") and name.endswith("_EVENT")
    }
    assert usage_ast == runtime_usage, "Review nonliteral or changed session_usage event constants"
    found.update(usage_types)
    assert found == _SEMANTIC_TYPES, f"Review semantic event inventory: {found ^ _SEMANTIC_TYPES}"
    return found


def _ui_types() -> set[str]:
    return {value for name, value in vars(KnownEventType).items() if name.isupper() and isinstance(value, str)}


async def _write_ui(
    store: FileSessionEventStore, session_id: str, image: ArtifactRef, background_payloads: list[dict[str, Any]]
) -> None:
    for event_type in sorted((_ui_types() | {_UNKNOWN_UI}) - {KnownEventType.BACKGROUND_JOB_DELIVERED}):
        content: dict[str, Any] = {"compat_payload": {"preserve": [1, "future"]}}
        if event_type == KnownEventType.TURN_STARTED:
            content.update(turn_id="compat-mixed", user_text="current-release user", image_count=1)
        elif event_type == KnownEventType.TASK_LIST_UPDATED:
            content = copy.deepcopy(_TASK_LIST_INITIAL)
        elif event_type == KnownEventType.USAGE_UPDATED:
            content = _usage_payload(populated=False)
        await store.append(
            AgentEvent(
                event_type=event_type,
                sender="compat-fixture",
                session_id=session_id,
                content=content,
                artifacts=[image] if event_type == KnownEventType.TURN_STARTED else [],
            )
        )
    for phase in (KnownEventType.TURN_STARTED, KnownEventType.TURN_ENDED):
        await store.append(
            AgentEvent(
                event_type=phase,
                sender="compat-fixture",
                session_id=session_id,
                content=(
                    {"turn_id": "compat-image-only", "user_text": "", "image_count": 1}
                    if phase == KnownEventType.TURN_STARTED
                    else {"turn_id": "compat-image-only", "status": "completed"}
                ),
                artifacts=[image] if phase == KnownEventType.TURN_STARTED else [],
            )
        )
    for event_type, content in (
        (KnownEventType.TERMINAL_COMMAND, {"command": "printf root", "tool_call_id": "compat-root-shell"}),
        (KnownEventType.TERMINAL_OUTPUT, {"output": "root output\n", "tool_call_id": "compat-root-shell"}),
        (
            KnownEventType.CHAT_MESSAGE,
            {"message_type": "task_list_update", "text": "- [x] Legacy task list"},
        ),
        (KnownEventType.TASK_LIST_UPDATED, {"markdown": _TASK_LIST["markdown"]}),
        (KnownEventType.USAGE_UPDATED, _usage_payload(populated=True)),
    ):
        await store.append(
            AgentEvent(event_type=event_type, sender="compat-fixture", session_id=session_id, content=content)
        )
    for worker in _SUB_AGENTS:
        for event_type, content in (
            (KnownEventType.CHAT_MESSAGE, {"status": "GENERATING"}),
            (
                KnownEventType.TERMINAL_COMMAND,
                {"command": worker["command"], "tool_call_id": worker["tool_call_id"]},
            ),
            (
                KnownEventType.TERMINAL_OUTPUT,
                {"output": worker["output"], "tool_call_id": worker["tool_call_id"]},
            ),
            (KnownEventType.CHAT_MESSAGE, {"status": "STOPPED", "total_tokens": worker["total_tokens"]}),
        ):
            await store.append(
                AgentEvent(
                    event_type=event_type,
                    sender="compat-fixture",
                    session_id=session_id,
                    sub_agent_info=worker["info"],
                    content=content,
                )
            )
    for content in background_payloads:
        await store.append(
            AgentEvent(
                event_type=KnownEventType.BACKGROUND_JOB_DELIVERED,
                sender="compat-fixture",
                session_id=session_id,
                content=copy.deepcopy(content),
            )
        )


def _message(role: str, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def create_fixture(root: Path) -> dict:
    semantic_types = _producer_inventory()
    root = Path(root)
    if any((root / name).exists() for name in ("settings.json", "sessions", _SNAPSHOT)):
        raise FileExistsError(f"Fixture root already contains settings, sessions or snapshot: {root}")
    root.mkdir(parents=True, exist_ok=True)
    settings_store = SettingsStore(root)
    settings = settings_store.load()
    settings.active_provider = "openai"
    settings.active_model = "compat-current-model"
    settings.active_theme = "compat-current-theme"
    settings.set_agent_model("building", "openai", "compat-current-model", thinking_effort="high")
    settings.set_api_key("openai", "compatibility-fixture-not-a-real-key")
    # Added after 0.47.1: unlike the synthetic sentinels, this is real current state.
    settings.usage_pricing = {"openai/compat-current-model": False}
    settings_store.save(settings)
    document = json.loads(settings_store.path.read_text(encoding="utf-8"))
    document["compat_future_top"] = {"nested": [1, {"keep": True}]}
    document["agent_models"]["building"]["compat_future_nested"] = {"value": "keep"}
    settings_store.path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    sessions = SessionStore(root)
    record = sessions.create(project_path=root, mode="build", config={"compat": True}, name="compat fixture")
    session_id = record.session_id
    try:
        record.task_list_markdown = _TASK_LIST["markdown"]
        sessions.save(record)
        recorder = sessions.recorder(session_id)
        journal = recorder.journal
        image = ImageBlock("base64", "image/png", base64.b64encode(_IMAGE).decode("ascii"))
        launch_turn = recorder.start_turn(Message(role="user", content=[TextBlock("current-release user"), image]))
        background_deliveries = _background_deliveries(journal.epoch_id, launch_turn, root)
        background_jobs: list[dict[str, Any]] = []
        for delivery in background_deliveries:
            job = {
                key: copy.deepcopy(delivery[key])
                for key in ("job_id", "completion_id", "agent_type", "task", "metadata", "epoch_id")
            }
            job.update(status="queued", result="", delivered=False)
            recorder.record_background_job_submitted(job, epoch_id=journal.epoch_id)
            background_jobs.append(job)
        recorder.record_assistant(_message("assistant", "current-release assistant"))
        recorder.finish_turn("completed")
        recorder.start_turn(Message(role="user", content=[image]))
        recorder.record_assistant(_message("assistant", "current-release image response"))
        recorder.finish_turn("completed")
        for job, delivery in zip(background_jobs, background_deliveries):
            job.update(status=delivery["status"], result=delivery["text"])
            recorder.record_background_job_terminal(job, epoch_id=journal.epoch_id)
            job["archived"] = delivery["disposition"] == "archived"
            inserted = recorder.record_background_completion(
                job, _message("user", delivery["text"]), epoch_id=journal.epoch_id
            )
            assert inserted
        existing = {event.event_type for event in journal.read_events(repair_tail=False)}
        message = _message("user", "compat context").to_dict()
        for event_type in sorted(semantic_types - existing):
            payload: dict[str, Any] = {"compat_payload": {"preserve": True}}
            if event_type in {"context.message", "tool.results", "assistant.message", "turn.started"}:
                payload["message"] = message
            elif event_type == "context.compacted":
                payload["compaction"] = {"compat": True}
            elif event_type == "context.rewound":
                payload["boundary_seq"] = 10**9
            elif event_type == "session.workspace_switched":
                payload["new_root"] = str(root.resolve())
            elif event_type in {"background.job.submitted", "context.message_archived"}:
                payload["completion_id"] = "compat-completion"
            journal.append(event_type, actor="system", payload=payload)
        journal.append(_UNKNOWN_SEMANTIC, actor="system", payload={"compat_unknown_payload": {"keep": [1, 2]}})
        image_ref = ArtifactRef(
            **journal.put_artifact(_IMAGE, media_type="image/png", purpose=ArtifactPurpose.IMAGE, encoding="base64")
        )
        background_payloads = [*background_deliveries, copy.deepcopy(background_deliveries[0])]
        asyncio.run(_write_ui(FileSessionEventStore(journal), session_id, image_ref, background_payloads))
        path = sessions.events_path_for(session_id)
        lines = path.read_bytes().splitlines()
        for index, line in enumerate(lines):
            item = json.loads(line)
            if item["type"] == _UNKNOWN_SEMANTIC:
                item["compat_unknown_envelope"] = {"keep": "envelope"}
                lines[index] = json.dumps(item, separators=(",", ":")).encode()
            elif item["type"] == f"ui.{_UNKNOWN_UI}":
                item["payload"]["event"]["compat_runtime_extension"] = {"keep": "runtime"}
                lines[index] = json.dumps(item, separators=(",", ":")).encode()
        raw = b"\n".join(lines) + b"\n"
        path.write_bytes(raw)
        (root / _SNAPSHOT).write_bytes(raw)
        events = journal.read_events(repair_tail=False)
        assert semantic_types | {_UNKNOWN_SEMANTIC} <= {event.event_type for event in events}
        assert {f"ui.{value}" for value in _ui_types() | {_UNKNOWN_UI}} <= {event.event_type for event in events}
        loaded = sessions.load(session_id)
        history = loaded.history
        assert any(_has_message(m, "user", "current-release user") for m in history)
        assert any(_has_message(m, "assistant", "current-release assistant") for m in history)
        expected = copy.deepcopy(document)
        expected["active_theme"] = "compat-previous-theme"
        expected["agent_models"]["building"]["model"] = "compat-previous-model"
        return {
            "session_id": session_id,
            "snapshot": _SNAPSHOT,
            "expected_settings": expected,
            "event_types": [event.event_type for event in events],
            "original_types": sorted({event.event_type for event in events}),
            "original_history": history,
            "original_metadata": {
                key: value
                for key, value in loaded.to_metadata_dict().items()
                if key
                in {
                    "schema_version",
                    "session_id",
                    "project_path",
                    "mode",
                    "name",
                    "title",
                    "config",
                    "task_list_markdown",
                }
            },
            "task_list_payloads": [copy.deepcopy(_TASK_LIST_INITIAL), {"markdown": _TASK_LIST["markdown"]}],
            "task_list": copy.deepcopy(_TASK_LIST),
            "usage_payloads": [_usage_payload(populated=False), _usage_payload(populated=True)],
            "usage": {"source_seq": 1, **_usage_payload(populated=True)["usage"]},
            "image_sha256": hashlib.sha256(_IMAGE).hexdigest(),
            "sub_agents": copy.deepcopy(_SUB_AGENTS),
            "background_delivery_payloads": copy.deepcopy(background_payloads),
            "background_deliveries": copy.deepcopy(background_deliveries),
        }
    finally:
        sessions.release_session_locks()


def _has_message(message: dict[str, Any], role: str, text: str) -> bool:
    content = message.get("content")
    return message.get("role") == role and (
        content == text
        or isinstance(content, list)
        and any(isinstance(block, dict) and block.get("text") == text for block in content)
    )


def verify_after_previous(root: Path, manifest: dict) -> None:
    root = Path(root)
    document = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    assert document == manifest["expected_settings"], "Settings changed beyond the expected previous-writer mutations"
    sessions = SessionStore(root)
    session_id = manifest["session_id"]
    original = (root / manifest["snapshot"]).read_bytes()
    actual = sessions.events_path_for(session_id).read_bytes()
    assert actual.startswith(original), "Original journal bytes were rewritten or truncated"
    assert len(actual) > len(original), "Previous writer did not append to the journal"
    events = sessions.journal(session_id).read_events(repair_tail=False)
    assert set(manifest["original_types"]) <= {event.event_type for event in events}
    assert len(events) == len(actual.splitlines()), "Journal contains unreadable records"
    ui_events = asyncio.run(FileSessionEventStore(sessions.journal(session_id)).read(session_id))
    assert _ui_types() | {_UNKNOWN_UI} <= {event.event_type for event in ui_events}
    task_events = [event for event in ui_events if event.event_type == KnownEventType.TASK_LIST_UPDATED]
    assert [event.content for event in task_events] == manifest["task_list_payloads"], (
        "Previous writer lost canonical task-list payloads"
    )
    assert all(event.sub_agent_info is None and event.schema_version == 2 for event in task_events), (
        "Task-list events changed scope or schema version"
    )
    usage_events = [event for event in ui_events if event.event_type == KnownEventType.USAGE_UPDATED]
    assert [event.content for event in usage_events] == manifest["usage_payloads"], (
        "Previous writer lost canonical usage payloads"
    )
    assert all(event.sub_agent_info is None and event.schema_version == 2 for event in usage_events), (
        "Usage events changed scope or schema version"
    )
    background_events = [event for event in ui_events if event.event_type == KnownEventType.BACKGROUND_JOB_DELIVERED]
    assert [event.content for event in background_events] == manifest["background_delivery_payloads"], (
        "Previous writer lost canonical background-delivery payloads"
    )
    assert all(event.sub_agent_info is None and event.schema_version == 2 for event in background_events), (
        "Background-delivery events changed scope or schema version"
    )
    image_turns = [
        event
        for event in ui_events
        if event.event_type == KnownEventType.TURN_STARTED
        and event.content.get("turn_id") in {"compat-mixed", "compat-image-only"}
    ]
    assert len(image_turns) == 2, "Previous writer lost user-image presentation events"
    for event in image_turns:
        assert event.content["image_count"] == 1
        assert "images" not in event.content, "Raw user-image payload escaped offloading"
        assert len(event.artifacts) == 1
        ref = event.artifacts[0]
        assert ref.sha256 == manifest["image_sha256"] == hashlib.sha256(_IMAGE).hexdigest()
        assert ref.purpose == ArtifactPurpose.IMAGE and ref.media_type == "image/png"
        assert asyncio.run(FileArtifactStore(sessions.journal(session_id)).open(ref)) == _IMAGE
    state = replay(ui_events)
    assert state.task_list == manifest["task_list"], "Previous writer lost the complete latest task-list projection"
    assert state.usage == manifest["usage"], "Previous writer lost the latest usage projection"
    for delivery in manifest["background_deliveries"]:
        rows = [item for item in state.conversation if delivery["text"] in item.text]
        assert len(rows) == 1, "Background-delivery projection lost, truncated or duplicated a full report"
        row = rows[0]
        first = next(
            event for event in background_events if event.content["completion_id"] == delivery["completion_id"]
        )
        assert row.sub_agent is None and row.seq == first.seq, "Background-delivery projection lost root attribution"
        assert (
            row.kind == "background_delivery" and row.text == delivery["text"] and row.status == delivery["status"]
        ), "Background-delivery projection changed report text or status"
        assert row.background_delivery == {
            key: value for key, value in delivery.items() if key not in {"text", "compat_payload"}
        }, "Background-delivery projection lost identity, disposition or admission metadata"
    background_sequences = {event.seq for event in background_events}
    assert all(
        item.seq not in background_sequences for activity in state.sub_agents.values() for item in activity.steps
    ), "Background-delivery projection duplicated reports into worker steps"
    assert set(state.sub_agents) == {worker["info"]["agent_id"] for worker in manifest["sub_agents"]}, (
        "Background-delivery projection invented a worker from missing metadata"
    )
    assert KnownEventType.BACKGROUND_JOB_DELIVERED not in state.unknown_event_types, (
        "Background-delivery projection treated a canonical event as unknown"
    )
    image_rows = [item for item in state.conversation if item.kind == "user"]
    assert [(item.text, item.image_count) for item in image_rows] == [("current-release user", 1), ("", 1)]
    assert all([ref.sha256 for ref in item.artifacts] == [manifest["image_sha256"]] for item in image_rows)
    task_sequences = {
        event.seq
        for event in ui_events
        if event.event_type == KnownEventType.TASK_LIST_UPDATED
        or event.event_type == KnownEventType.CHAT_MESSAGE
        and event.content.get("message_type") == "task_list_update"
    }
    assert all(item.seq not in task_sequences for item in state.conversation), (
        "Task-list updates duplicated into conversation prose"
    )
    assert state.terminal == "$ printf root\nroot output\n", "Child terminal output escaped into the root terminal"
    for worker in manifest["sub_agents"]:
        info = worker["info"]
        activity = state.to_dict()["sub_agents"].get(info["agent_id"])
        assert activity is not None, "Previous writer lost a sub-agent activity"
        assert all(activity[key] == value for key, value in info.items() if key != "agent_name"), (
            "Previous writer lost sub-agent linkage or routing"
        )
        assert activity["status"] == "completed" and activity["total_tokens"] == worker["total_tokens"], (
            "Previous writer lost sub-agent completion or token totals"
        )
        assert activity["terminal"] == f"$ {worker['command']}\n{worker['output']}", (
            "Previous writer lost attributed sub-agent terminal output"
        )
        terminal_events = [
            event
            for event in ui_events
            if event.sub_agent_info == info
            and event.event_type in {KnownEventType.TERMINAL_COMMAND, KnownEventType.TERMINAL_OUTPUT}
        ]
        assert len(terminal_events) == 2 and all(
            event.content.get("tool_call_id") == worker["tool_call_id"] for event in terminal_events
        ), "Previous writer lost a child command's tool-call identity"
    loaded = sessions.load(session_id)
    metadata = loaded.to_metadata_dict()
    assert {key: metadata[key] for key in manifest["original_metadata"]} == manifest["original_metadata"], (
        "Previous writer changed original session metadata"
    )
    history = loaded.history
    assert history[: len(manifest["original_history"])] == manifest["original_history"]
    appended_history = history[len(manifest["original_history"]) :]
    assert any(
        _has_message(user, "user", "previous-release user")
        and _has_message(assistant, "assistant", "previous-release assistant")
        for user, assistant in zip(appended_history, appended_history[1:])
    ), "Previous turn was not replayed in order"
    appended_events = events[len(original.splitlines()) :]
    started: set[tuple[str, str]] = set()
    answered: set[tuple[str, str]] = set()
    completed: set[tuple[str, str]] = set()
    for event in appended_events:
        if not event.turn_id:
            continue
        turn = (event.epoch_id, event.turn_id)
        if event.event_type == "turn.started" and _has_message(
            event.payload.get("message", {}), "user", "previous-release user"
        ):
            started.add(turn)
        elif (
            event.event_type == "assistant.message"
            and turn in started
            and _has_message(event.payload.get("message", {}), "assistant", "previous-release assistant")
        ):
            answered.add(turn)
        elif event.event_type == "turn.completed" and turn in answered:
            completed.add(turn)
    assert completed, "Previous turn must contain an ordered user, assistant and completion in the same epoch"
