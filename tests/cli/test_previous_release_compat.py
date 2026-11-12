"""Offline failure-detection tests for the previous-release compatibility check."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest
from packaging.version import Version

from kolega_code.cli.goal import GoalState
from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import SettingsStore
from kolega_code.events import AGENT_EVENT_SCHEMA_VERSION, AgentEvent, KnownEventType
from kolega_code.llm.models import ImageBlock, Message
from kolega_code.session.projection import ConversationItem, PresentationState, replay
from kolega_code.session.store import SessionStoreError

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture_module = _load("previous_release_compat_fixture")
check = _load("check_previous_release_compat")
worker = _load("previous_release_compat_worker")


def _wheel(version: str, *, yanked: bool = False) -> dict[str, Any]:
    filename = f"kolega_code-{version}-py3-none-any.whl"
    return {
        "filename": filename,
        "packagetype": "bdist_wheel",
        "yanked": yanked,
        "url": f"https://files.pythonhosted.org/packages/{filename}",
        "digests": {"sha256": "a" * 64},
    }


def _pypi(monkeypatch: pytest.MonkeyPatch, releases: dict[str, list[dict[str, Any]]]) -> None:
    data = json.dumps({"releases": releases}).encode()
    monkeypatch.setattr(check, "urlopen", lambda *args, **kwargs: io.BytesIO(data))


def test_discovery_uses_latest_non_yanked_stable_release(monkeypatch: pytest.MonkeyPatch) -> None:
    _pypi(
        monkeypatch,
        {
            "1.9": [_wheel("1.9")],
            "1.10": [_wheel("1.10")],
            "2.0rc1": [_wheel("2.0rc1")],
            "3.0": [_wheel("3.0", yanked=True)],
            "not-a-version": [_wheel("not-a-version")],
        },
    )
    version, wheel = check.discover_release()
    assert version == "1.10"
    assert wheel["filename"] == "kolega_code-1.10-py3-none-any.whl"


def test_release_discovery_excludes_candidate_and_newer_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    _pypi(
        monkeypatch,
        {
            "1.8": [_wheel("1.8")],
            "1.9": [_wheel("1.9")],
            "1.10": [_wheel("1.10")],
            "1.11": [_wheel("1.11")],
        },
    )
    # The candidate may already be on PyPI when a preflight is rerun.
    assert check.discover_release(before=Version("1.10"))[0] == "1.9"
    assert check.discover_release()[0] == "1.11"


def test_release_discovery_fails_when_no_older_release_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    _pypi(monkeypatch, {"1.0": [_wheel("1.0")], "2.0": [_wheel("2.0")]})
    with pytest.raises(ValueError, match="non-yanked stable release"):
        check.discover_release(before=Version("1.0"))


@pytest.mark.parametrize("version", ["1.0", "2.0"])
def test_release_preflight_rejects_not_older_installed_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, version: str
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.0"\n', encoding="utf-8")
    monkeypatch.setenv("KOLEGA_SCRATCHPAD", str(tmp_path))
    monkeypatch.setattr(check, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["check", "--previous-python", str(tmp_path / "python"), "--release-preflight"])
    monkeypatch.setattr(check, "installed_version", lambda *args: version)
    with pytest.raises(ValueError, match="must be older than candidate"):
        check.main()


def test_discovery_fails_if_latest_release_has_no_safe_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    source_only = {"filename": "kolega_code-2.0.tar.gz", "packagetype": "sdist", "yanked": False}
    _pypi(monkeypatch, {"1.0": [_wheel("1.0")], "2.0": [source_only]})
    with pytest.raises(ValueError, match="universal wheel"):
        check.discover_release()


def test_discovery_rejects_untrusted_wheel_host(monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = _wheel("1.0")
    wheel["url"] = "https://example.com/kolega_code-1.0-py3-none-any.whl"
    _pypi(monkeypatch, {"1.0": [wheel]})
    with pytest.raises(ValueError, match="download host"):
        check.discover_release()


def test_wheel_hash_mismatch_is_fatal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(check, "urlopen", lambda *args, **kwargs: io.BytesIO(b"not the wheel"))
    with pytest.raises(ValueError, match="SHA-256"):
        check.download_wheel(_wheel("1.0"), tmp_path)


def test_previous_process_environment_is_isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-secret")
    monkeypatch.setenv("UV_EXCLUDE_NEWER", "1 week")
    monkeypatch.setenv("PYTHONPATH", "/fake/checkout")
    monkeypatch.setenv("PIP_INDEX_URL", "https://example.com")
    env = check.clean_environment(tmp_path)
    assert "OPENAI_API_KEY" not in env
    assert "UV_EXCLUDE_NEWER" not in env
    assert "PYTHONPATH" not in env
    assert "PIP_INDEX_URL" not in env
    assert env["HOME"] == str(tmp_path)


def test_checkout_and_mismatched_distribution_imports_are_rejected(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    prefix = checkout / ".venv"
    installed = prefix / "site-packages/kolega_code/__init__.py"
    worker.verify_origin(installed, installed, prefix, checkout)
    source = checkout / "kolega_code/__init__.py"
    with pytest.raises(RuntimeError, match="not distribution module"):
        worker.verify_origin(source, installed, prefix, checkout)
    with pytest.raises(RuntimeError, match="outside isolated"):
        worker.verify_origin(source, source, prefix, checkout)
    with pytest.raises(RuntimeError, match="checkout"):
        worker.verify_origin(source, source, checkout, checkout)


def test_previous_reader_must_replay_original_history_in_order() -> None:
    expected = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    worker.verify_original_history(expected, expected)
    for actual in ([], list(reversed(expected))):
        with pytest.raises(RuntimeError, match="did not replay"):
            worker.verify_original_history(actual, expected)


def test_previous_worker_must_read_original_session_metadata() -> None:
    expected = {
        "schema_version": 2,
        "session_id": "compat-session",
        "name": "compat fixture",
        "config": {"compat": True},
    }
    worker.verify_original_metadata({**expected, "future_field": "ignored"}, expected)
    for actual in ({}, {**expected, "name": "lost"}):
        with pytest.raises(RuntimeError, match="original session metadata"):
            worker.verify_original_metadata(actual, expected)


@pytest.mark.parametrize("damage", ["missing", "truncated", "unknown_fields", "worker_tag", "schema"])
def test_previous_worker_rejects_lost_or_changed_canonical_task_payloads(damage: str) -> None:
    payload = fixture_module._TASK_LIST_INITIAL
    event = AgentEvent(event_type="task_list_updated", sender="compat-fixture", content=payload)
    worker.verify_task_list_payloads([event], [payload])
    if damage == "missing":
        actual = []
    else:
        actual = [event.model_copy(deep=True)]
        if damage == "truncated":
            actual[0].content["markdown"] = payload["markdown"][:-1]
        elif damage == "unknown_fields":
            del actual[0].content["compat_payload"]
        elif damage == "worker_tag":
            actual[0].sub_agent_info = {"agent_id": "worker"}
        else:
            actual[0].schema_version += 1
    with pytest.raises(RuntimeError, match="task-list"):
        worker.verify_task_list_payloads(actual, [payload])


@pytest.mark.parametrize("damage", ["missing", "decimal", "unknown_fields", "worker_tag", "schema"])
def test_previous_worker_rejects_lost_or_changed_canonical_usage_payloads(damage: str) -> None:
    payload = fixture_module._usage_payload(populated=True)
    event = AgentEvent(event_type="usage_updated", sender="compat-fixture", content=payload)
    worker.verify_usage_payloads([event], [payload])
    actual = [] if damage == "missing" else [event.model_copy(deep=True)]
    if actual:
        if damage == "decimal":
            actual[0].content["usage"]["costs"]["currencies"]["USD"]["total"] = "0"
        elif damage == "unknown_fields":
            del actual[0].content["compat_payload"]
        elif damage == "worker_tag":
            actual[0].sub_agent_info = {"agent_id": "worker"}
        else:
            actual[0].schema_version += 1
    with pytest.raises(RuntimeError, match="usage"):
        worker.verify_usage_payloads(actual, [payload])


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "truncated",
        "counter",
        "reason",
        "verdict",
        "unknown_fields",
        "nested_payload",
        "worker_tag",
        "empty_worker_tag",
        "schema",
    ],
)
def test_previous_worker_rejects_lost_or_changed_canonical_goal_payloads(damage: str) -> None:
    expected = [fixture_module._GOAL_INITIAL, fixture_module._GOAL]
    events = [
        AgentEvent(event_type="goal_updated", sender="compat-fixture", content=copy.deepcopy(payload))
        for payload in expected
    ]
    worker.verify_goal_payloads(events, expected)
    if damage == "missing":
        events.pop()
    else:
        target = events[-1]
        if damage == "truncated":
            target.content["goal"]["condition"] = target.content["goal"]["condition"][:500]
        elif damage == "counter":
            target.content["goal"]["tokens_spent"] += 1
        elif damage == "reason":
            target.content["reason"] = "resumed"
        elif damage == "verdict":
            target.content["evaluation_met"] = True
        elif damage == "unknown_fields":
            del target.content["compat_payload"]
        elif damage == "nested_payload":
            target.content["compat_payload"]["preserve"][-1]["text"] = "lost"
        elif damage in {"worker_tag", "empty_worker_tag"}:
            target.sub_agent_info = {"agent_id": "worker"} if damage == "worker_tag" else {}
        else:
            target.schema_version += 1
    with pytest.raises(RuntimeError, match="goal"):
        worker.verify_goal_payloads(events, expected)


def _simulate_writer(root: Path, manifest: dict[str, Any]) -> None:
    settings_store = SettingsStore(root)
    settings = settings_store.load()
    settings.active_theme = "compat-previous-theme"
    settings.agent_models["building"]["model"] = "compat-previous-model"
    settings_store.save(settings)
    sessions = SessionStore(root)
    recorder = sessions.recorder(manifest["session_id"], recover=False)
    recorder.start_turn(Message(role="user", content="previous-release user"))
    recorder.record_assistant(Message(role="assistant", content="previous-release assistant"))
    recorder.finish_turn("completed")
    sessions.release_session_locks()


@pytest.fixture
def completed_roundtrip(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest = fixture_module.create_fixture(tmp_path)
    _simulate_writer(tmp_path, manifest)
    return tmp_path, manifest


def test_roundtrip_checks_current_event_coverage(completed_roundtrip: tuple[Path, dict[str, Any]]) -> None:
    root, manifest = completed_roundtrip
    assert fixture_module._producer_inventory() <= set(manifest["event_types"])
    assert {f"ui.{name}" for name in fixture_module._ui_types()} <= set(manifest["event_types"])
    fixture_module.verify_after_previous(root, manifest)


def test_roundtrip_preserves_ordered_queue_without_semantic_delivery(
    completed_roundtrip: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = completed_roundtrip
    store = SessionStore(root)
    events = asyncio.run(FileSessionEventStore(store.journal(manifest["session_id"])).read(manifest["session_id"]))
    queue = [event for event in events if event.event_type == "queue_updated"]
    worker.verify_queue_payloads(events, manifest["queue_payloads"])
    assert [event.content["revision"] for event in queue] == [0, 1]
    assert queue[0].content["items"] == []
    state = replay(events)
    assert state.queue is not None
    assert [item["queue_id"] for item in state.queue["items"]] == ["compat-local", "compat-peer"]
    assert [item["text"] for item in state.queue["items"]] == ["Literal pending café 🐍 <preview>", "Peer preview"]
    assert [item["image_count"] for item in state.queue["items"]] == [2, 0]
    assert not any("Literal pending" in item.text or item.text == "Peer preview" for item in state.conversation)
    assert not any(
        "Literal pending" in str(message) or "Peer preview" in str(message)
        for message in store.load(manifest["session_id"]).history
    )
    fixture_module.verify_after_previous(root, manifest)


@pytest.mark.parametrize("damage", ["payload", "worker", "projection"])
def test_lost_or_misattributed_queue_fails_roundtrip(
    completed_roundtrip: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    if damage == "projection":
        original_replay = fixture_module.replay

        def broken_replay(events: list[AgentEvent]) -> PresentationState:
            state = original_replay(events)
            state.queue = None
            return state

        monkeypatch.setattr(fixture_module, "replay", broken_replay)
    else:
        original_read = FileSessionEventStore.read

        async def broken_read(self: FileSessionEventStore, session_id: str, **kwargs: Any) -> list[AgentEvent]:
            events = await original_read(self, session_id, **kwargs)
            target = next(event for event in events if event.event_type == "queue_updated")
            if damage == "payload":
                target.content["revision"] = 99
            else:
                target.sub_agent_info = {"agent_id": "child"}
            return events

        monkeypatch.setattr(FileSessionEventStore, "read", broken_read)
    with pytest.raises(AssertionError, match="[Qq]ueue"):
        fixture_module.verify_after_previous(*completed_roundtrip)


def test_roundtrip_preserves_full_goal_observations_without_conversation_or_recovery_authority(
    completed_roundtrip: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = completed_roundtrip
    store = SessionStore(root)
    session_id = manifest["session_id"]
    events = asyncio.run(FileSessionEventStore(store.journal(session_id)).read(session_id))
    goals = [event for event in events if event.event_type == KnownEventType.GOAL_UPDATED]
    worker.verify_goal_payloads(events, manifest["goal_payloads"])
    assert AGENT_EVENT_SCHEMA_VERSION == 2
    assert [event.content["revision"] for event in goals] == [0, 1]
    assert all(event.sub_agent_info is None and event.schema_version == 2 for event in goals)
    first = events.index(goals[0])
    assert replay(events[:first]).goal is None
    initial = replay(events[: first + 1])
    assert initial.goal is not None
    assert initial.goal == {
        key: value for key, value in manifest["goal_payloads"][0].items() if key != "compat_payload"
    }
    assert initial.goal["goal"] is None and initial.goal["evaluation_met"] is None
    state = replay(events)
    assert state.goal is not None
    assert state.goal == manifest["goal"]
    goal = state.goal["goal"]
    assert set(goal) == set(GoalState(condition="", started_at="").to_dict())
    assert len(goal["condition"]) > 500 and goal["condition"].endswith("GOAL COMPAT END")
    assert "<literal> **unchanged**\n  " in goal["condition"]
    assert len(goal["last_reason"]) > 500 and goal["last_reason"].endswith("REASON COMPAT END")
    assert (goal["turns_evaluated"], goal["tokens_spent"], goal["max_turns"]) == (3, 321, 17)
    assert goal["started_at"] == "2026-11-12T10:00:00+00:00"
    assert goal["last_evaluated_at"] == "2026-11-12T10:01:00+00:00"
    assert goal["run_to_completion"] is True and goal["paused"] is True and goal["met"] is False
    assert goal["status_note"] == "Stopped by user — café 🐍 <literal>"
    assert state.goal["evaluation_met"] is False and state.goal["reason"] == "paused"
    assert "compat_payload" not in state.goal
    without_goals = replay([event for event in events if event.event_type != KnownEventType.GOAL_UPDATED])
    assert without_goals.goal is None
    assert state.conversation == without_goals.conversation
    assert state.turns == without_goals.turns
    assert {key: activity.steps for key, activity in state.sub_agents.items()} == {
        key: activity.steps for key, activity in without_goals.sub_agents.items()
    }
    loaded = store.load(session_id)
    assert loaded.goal == {}
    assert not any(goal["condition"] in str(message) for message in loaded.history)
    original = (root / manifest["snapshot"]).read_bytes()
    actual = store.events_path_for(session_id).read_bytes()
    assert len(actual) > len(original) and actual.startswith(original)
    raw_goals = [
        json.loads(line)["payload"]["event"]["content"]
        for line in actual.splitlines()
        if json.loads(line)["type"] == "ui.goal_updated"
    ]
    assert raw_goals == manifest["goal_payloads"]
    assert raw_goals[-1]["compat_payload"]["preserve"] == [
        1,
        False,
        None,
        {"text": "future goal café 🐍 <literal>"},
    ]
    fixture_module.verify_after_previous(root, manifest)


@pytest.mark.parametrize("damage", ["missing", "truncated", "nested_payload", "worker_tag", "schema"])
def test_lost_or_changed_goal_payload_fails_roundtrip(
    completed_roundtrip: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    original_read = FileSessionEventStore.read

    async def damage_goal(self: FileSessionEventStore, session_id: str, **kwargs: Any) -> list[AgentEvent]:
        events = await original_read(self, session_id, **kwargs)
        target = [event for event in events if event.event_type == KnownEventType.GOAL_UPDATED][-1]
        if damage == "missing":
            events.remove(target)
        elif damage == "truncated":
            target.content["goal"]["condition"] = target.content["goal"]["condition"][:500]
        elif damage == "nested_payload":
            target.content["compat_payload"]["preserve"][-1]["text"] = "lost"
        elif damage == "worker_tag":
            target.sub_agent_info = {"agent_id": "compat-parent"}
        else:
            target.schema_version += 1
        return events

    monkeypatch.setattr(FileSessionEventStore, "read", damage_goal)
    with pytest.raises(AssertionError, match="[Gg]oal"):
        fixture_module.verify_after_previous(*completed_roundtrip)


@pytest.mark.parametrize("damage", ["missing", "truncated", "extension", "unknown", "prose", "worker_steps"])
def test_lost_or_inflated_goal_projection_fails_roundtrip(
    completed_roundtrip: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    original_replay = fixture_module.replay

    def damage_goal(events: list[AgentEvent]) -> PresentationState:
        state = original_replay(events)
        if damage == "missing":
            state.goal = None
        elif damage == "truncated":
            assert state.goal is not None
            state.goal["goal"]["condition"] = state.goal["goal"]["condition"][:500]
        elif damage == "extension":
            assert state.goal is not None
            state.goal["compat_payload"] = {"should": "be ignored"}
        elif damage == "unknown":
            state.unknown_event_types.add(KnownEventType.GOAL_UPDATED)
        else:
            event = next(event for event in events if event.event_type == KnownEventType.GOAL_UPDATED)
            row = ConversationItem(kind="system", text="Goal observation", seq=event.seq)
            if damage == "prose":
                state.conversation.append(row)
            else:
                state.sub_agents["compat-parent"].steps.append(row)
        return state

    monkeypatch.setattr(fixture_module, "replay", damage_goal)
    with pytest.raises(AssertionError, match="[Gg]oal"):
        fixture_module.verify_after_previous(*completed_roundtrip)


def test_roundtrip_retains_real_mixed_and_image_only_user_history(
    completed_roundtrip: tuple[Path, dict[str, Any]],
) -> None:
    from PIL import Image

    root, manifest = completed_roundtrip
    store = SessionStore(root)
    users = [Message.from_dict(item) for item in store.load(manifest["session_id"]).history if item["role"] == "user"]
    assert users[0].get_text_content() == "current-release user"
    assert users[1].get_text_content() == ""
    for message in users[:2]:
        (image,) = [block for block in message.content if isinstance(block, ImageBlock)]
        assert image.image_type == "base64" and image.media_type == "image/png"
    path = store.journal(manifest["session_id"]).artifacts_dir / manifest["image_sha256"]
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["image_sha256"]
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (1, 1)
        assert image.format == "PNG"
    fixture_module.verify_after_previous(root, manifest)


def test_roundtrip_retains_nested_background_linkage_and_terminal_output(
    completed_roundtrip: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = completed_roundtrip
    store = SessionStore(root)
    events = asyncio.run(FileSessionEventStore(store.journal(manifest["session_id"])).read(manifest["session_id"]))
    state = replay(events)
    assert state.terminal == "$ printf root\nroot output\n"
    parent = state.sub_agents["compat-parent"]
    nested = state.sub_agents["compat-nested"]
    assert parent.parent_agent_id == "root"
    assert parent.parent_tool_call_id == "compat-dispatch-parent"
    assert parent.depth == 1 and parent.background is True and parent.total_tokens == 37
    assert parent.terminal == "$ printf parent\nparent output\n"
    assert nested.parent_agent_id == "compat-parent"
    assert nested.parent_tool_call_id == "compat-dispatch-nested"
    assert nested.depth == 2 and nested.background is False and nested.total_tokens == 0
    assert nested.effective_routing == {
        "provider": "openai",
        "model": "compat-nested-model",
        "thinking_effort": None,
    }
    assert nested.terminal == "$ printf nested\nnested output\n"
    fixture_module.verify_after_previous(root, manifest)


def test_roundtrip_retains_full_latest_task_list_without_conversation_prose(
    completed_roundtrip: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = completed_roundtrip
    store = SessionStore(root)
    events = asyncio.run(FileSessionEventStore(store.journal(manifest["session_id"])).read(manifest["session_id"]))
    tasks = [event for event in events if event.event_type == KnownEventType.TASK_LIST_UPDATED]
    loaded = store.load(manifest["session_id"])
    assert loaded.task_list_markdown == manifest["task_list"]["markdown"]
    worker.verify_original_metadata(loaded.to_metadata_dict(), manifest["original_metadata"])
    assert AGENT_EVENT_SCHEMA_VERSION == 2
    assert all(event.schema_version == 2 and event.sub_agent_info is None for event in tasks)
    assert [event.content for event in tasks] == manifest["task_list_payloads"]
    initial = replay(events[: events.index(tasks[0]) + 1])
    assert initial.task_list == {
        "markdown": "- [ ] Original task list",
        "entries": [{"text": "Original task list", "status": "pending"}],
    }
    legacy = next(event for event in events if event.content.get("message_type") == "task_list_update")
    legacy_state = replay(events[: events.index(legacy) + 1])
    assert legacy_state.task_list == {
        "markdown": "- [x] Legacy task list",
        "entries": [{"text": "Legacy task list", "status": "completed"}],
    }
    state = replay(events)
    assert state.task_list == manifest["task_list"]
    assert len(manifest["task_list"]["entries"][1]["text"]) > 500
    task_sequences = {event.seq for event in [*tasks, legacy]}
    assert all(item.seq not in task_sequences for item in state.conversation)
    worker.verify_task_list_payloads(events, manifest["task_list_payloads"])
    fixture_module.verify_after_previous(root, manifest)


def test_roundtrip_retains_usage_precision_zero_provenance_and_separate_credits(
    completed_roundtrip: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = completed_roundtrip
    store = SessionStore(root)
    events = asyncio.run(FileSessionEventStore(store.journal(manifest["session_id"])).read(manifest["session_id"]))
    usage_events = [event for event in events if event.event_type == KnownEventType.USAGE_UPDATED]
    assert len(usage_events) == 2
    assert [event.content for event in usage_events] == manifest["usage_payloads"]
    state = replay(events)
    assert state.usage is not None
    assert state.usage["requests"] == state.usage["known_requests"]["total_tokens"] == 2
    assert state.usage["total_tokens"] == 0
    assert state.usage["costs"]["currencies"]["USD"]["total"] == "0.000000000000000000123456789"
    assert state.usage["costs"]["currencies"]["USD"]["estimated_requests"] == 0
    assert state.usage["costs"]["non_monetary"]["api_credits"]["total"] == "0.002"
    assert state.usage["costs"]["non_monetary"]["api_credits"]["reported_requests"] == 0
    assert all(item.seq not in {event.seq for event in usage_events} for item in state.conversation)
    worker.verify_usage_payloads(events, manifest["usage_payloads"])
    fixture_module.verify_after_previous(root, manifest)


def test_roundtrip_retains_background_delivery_evidence_and_full_root_reports(
    completed_roundtrip: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = completed_roundtrip
    store = SessionStore(root)
    journal = store.journal(manifest["session_id"])
    events = asyncio.run(FileSessionEventStore(journal).read(manifest["session_id"]))
    deliveries = [event for event in events if event.event_type == KnownEventType.BACKGROUND_JOB_DELIVERED]
    assert [event.content for event in deliveries] == manifest["background_delivery_payloads"]
    assert all(event.sub_agent_info is None and event.schema_version == 2 for event in deliveries)
    assert {event.content["status"] for event in deliveries} == {"completed", "failed", "cancelled", "interrupted"}
    assert {event.content["disposition"] for event in deliveries} == {"context", "archived"}
    assert any(event.content["metadata"] == {} for event in deliveries)
    semantic = journal.read_events(repair_tail=False)
    for delivery in deliveries:
        payload = delivery.content
        admitted = next(
            event
            for event in semantic
            if event.event_type == "background.job.submitted"
            and event.payload.get("completion_id") == payload["completion_id"]
        )
        terminal = next(
            event
            for event in semantic
            if event.event_type == "background.job.terminal"
            and event.payload.get("completion_id") == payload["completion_id"]
        )
        committed = next(
            event
            for event in semantic
            if event.event_type == "context.message" and event.payload.get("completion_id") == payload["completion_id"]
        )
        assert admitted.seq < terminal.seq < committed.seq
        assert delivery.seq is not None and committed.seq < delivery.seq
        assert admitted.epoch_id == terminal.epoch_id == committed.epoch_id == payload["epoch_id"]
        assert admitted.payload["metadata"] == committed.payload["metadata"] == payload["metadata"]
        assert committed.payload["job_id"] == payload["job_id"]
        assert committed.payload["status"] == payload["status"]
        assert committed.actor == "system" and committed.depth == 0
        assert committed.payload["archived"] == (payload["disposition"] == "archived")
        message = journal.hydrate_message(committed.payload["message"])
        assert Message.from_dict(message).get_text_content() == payload["text"]
        assert len(payload["text"]) > 1000 and "café 🐍 <literal>" in payload["text"]
        assert payload["compat_payload"]["preserve"][:2] == [False, None]
    assert deliveries[0].content["metadata"]["turn_id"] == next(
        event.turn_id for event in semantic if event.event_type == "turn.started"
    )
    fixture_module.verify_after_previous(root, manifest)


def test_background_delivery_replay_deduplicates_and_seeks_without_inventing_legacy_reports(
    completed_roundtrip: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = completed_roundtrip
    store = SessionStore(root)
    events = asyncio.run(FileSessionEventStore(store.journal(manifest["session_id"])).read(manifest["session_id"]))
    first = next(
        index for index, event in enumerate(events) if event.event_type == KnownEventType.BACKGROUND_JOB_DELIVERED
    )
    before = replay(events[:first])
    assert not any(item.kind == "background_delivery" for item in before.conversation)
    initial = replay(events[: first + 1])
    assert [item.text for item in initial.conversation if item.kind == "background_delivery"] == [
        manifest["background_deliveries"][0]["text"]
    ]
    state = replay(events)
    assert [item.text for item in state.conversation if item.kind == "background_delivery"] == [
        payload["text"] for payload in manifest["background_deliveries"]
    ]
    legacy = replay([event for event in events if event.event_type != KnownEventType.BACKGROUND_JOB_DELIVERED])
    assert not any(item.kind == "background_delivery" for item in legacy.conversation)
    assert fixture_module._UNKNOWN_UI in legacy.unknown_event_types
    assert KnownEventType.BACKGROUND_JOB_DELIVERED not in state.unknown_event_types
    assert set(state.sub_agents) == set(legacy.sub_agents) == {"compat-parent", "compat-nested"}


@pytest.mark.parametrize("damage", ["missing", "truncated", "unknown_fields", "worker_tag", "schema"])
def test_lost_or_changed_background_delivery_payload_fails_roundtrip(
    completed_roundtrip: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    original_read = FileSessionEventStore.read

    async def damage_delivery(self: FileSessionEventStore, session_id: str, **kwargs: Any) -> list[AgentEvent]:
        events = await original_read(self, session_id, **kwargs)
        target = next(event for event in events if event.event_type == KnownEventType.BACKGROUND_JOB_DELIVERED)
        if damage == "missing":
            events.remove(target)
        elif damage == "truncated":
            target.content["text"] = target.content["text"][:500]
        elif damage == "unknown_fields":
            del target.content["compat_payload"]
        elif damage == "worker_tag":
            target.sub_agent_info = {"agent_id": "compat-parent"}
        else:
            target.schema_version += 1
        return events

    monkeypatch.setattr(FileSessionEventStore, "read", damage_delivery)
    with pytest.raises(AssertionError, match="[Bb]ackground-delivery"):
        fixture_module.verify_after_previous(*completed_roundtrip)


@pytest.mark.parametrize(
    "damage",
    ["missing", "truncated", "duplicate", "worker", "worker_steps", "metadata", "disposition", "invented_worker"],
)
def test_lost_or_misattributed_background_delivery_projection_fails_roundtrip(
    completed_roundtrip: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    original_replay = fixture_module.replay

    def damage_delivery(events: list[AgentEvent]) -> PresentationState:
        state = original_replay(events)
        target = next(item for item in state.conversation if item.kind == "background_delivery")
        if damage == "missing":
            state.conversation.remove(target)
        elif damage == "truncated":
            target.text = target.text[:500]
        elif damage == "duplicate":
            state.conversation.append(target)
        elif damage == "worker":
            target.sub_agent = "compat-parent"
        elif damage == "worker_steps":
            state.sub_agents["compat-parent"].steps.append(target)
        elif damage in {"metadata", "disposition"}:
            assert target.background_delivery is not None
            target.background_delivery[damage] = {} if damage == "metadata" else "archived"
        else:
            state.sub_agents["invented"] = state.sub_agents["compat-nested"]
        return state

    monkeypatch.setattr(fixture_module, "replay", damage_delivery)
    with pytest.raises(AssertionError, match="[Bb]ackground-delivery"):
        fixture_module.verify_after_previous(*completed_roundtrip)


def test_lost_usage_projection_fails_roundtrip(
    completed_roundtrip: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    original_replay = fixture_module.replay

    def drop_usage(events: list[AgentEvent]) -> PresentationState:
        state = original_replay(events)
        state.usage = None
        return state

    monkeypatch.setattr(fixture_module, "replay", drop_usage)
    with pytest.raises(AssertionError, match="usage projection"):
        fixture_module.verify_after_previous(*completed_roundtrip)


@pytest.mark.parametrize("damage", ["missing", "truncated", "prose"])
def test_lost_or_duplicated_task_list_fails_roundtrip(
    completed_roundtrip: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    original_replay = fixture_module.replay

    def damage_tasks(events: list[AgentEvent]) -> PresentationState:
        state = original_replay(events)
        if damage == "missing":
            state.task_list = None
        elif damage == "truncated":
            assert state.task_list is not None
            state.task_list["entries"][1]["text"] = state.task_list["entries"][1]["text"][:500]
        else:
            legacy = next(event for event in events if event.content.get("message_type") == "task_list_update")
            state.conversation.append(ConversationItem(kind="system", text=legacy.content["text"], seq=legacy.seq))
        return state

    monkeypatch.setattr(fixture_module, "replay", damage_tasks)
    with pytest.raises(AssertionError, match="task-list|Task-list"):
        fixture_module.verify_after_previous(*completed_roundtrip)


def test_lost_subagent_linkage_fails_roundtrip(
    completed_roundtrip: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    original_replay = fixture_module.replay

    def drop_parent(events: list[AgentEvent]) -> PresentationState:
        state = original_replay(events)
        state.sub_agents["compat-nested"].parent_agent_id = None
        return state

    monkeypatch.setattr(fixture_module, "replay", drop_parent)
    with pytest.raises(AssertionError, match="sub-agent linkage"):
        fixture_module.verify_after_previous(*completed_roundtrip)


def test_missing_user_image_artifact_fails_roundtrip(
    completed_roundtrip: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = completed_roundtrip
    artifact = SessionStore(root).journal(manifest["session_id"]).artifacts_dir / manifest["image_sha256"]
    artifact.unlink()
    with pytest.raises(SessionStoreError, match="Missing session artifact"):
        fixture_module.verify_after_previous(root, manifest)


def test_corrupt_user_image_artifact_fails_roundtrip(
    completed_roundtrip: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = completed_roundtrip
    artifact = SessionStore(root).journal(manifest["session_id"]).artifacts_dir / manifest["image_sha256"]
    artifact.write_bytes(b"not the original image")
    with pytest.raises(SessionStoreError, match="integrity"):
        fixture_module.verify_after_previous(root, manifest)


def test_rewritten_journal_bytes_fail(completed_roundtrip: tuple[Path, dict[str, Any]]) -> None:
    root, manifest = completed_roundtrip
    path = SessionStore(root).events_path_for(manifest["session_id"])
    data = path.read_bytes()
    path.write_bytes(b" " + data)  # Still JSON-decodable, but no longer the original bytes.
    with pytest.raises(AssertionError, match="rewritten or truncated"):
        fixture_module.verify_after_previous(root, manifest)


def test_lost_unknown_settings_fail(completed_roundtrip: tuple[Path, dict[str, Any]]) -> None:
    root, manifest = completed_roundtrip
    path = root / "settings.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    del saved["agent_models"]["building"]["compat_future_nested"]
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(AssertionError, match="Settings changed"):
        fixture_module.verify_after_previous(root, manifest)


def test_missing_previous_turn_completion_fails(completed_roundtrip: tuple[Path, dict[str, Any]]) -> None:
    root, manifest = completed_roundtrip
    path = SessionStore(root).events_path_for(manifest["session_id"])
    lines = path.read_bytes().splitlines(keepends=True)
    assert json.loads(lines[-1])["type"] == "turn.completed"
    path.write_bytes(b"".join(lines[:-1]))
    with pytest.raises(AssertionError, match="ordered user, assistant and completion"):
        fixture_module.verify_after_previous(root, manifest)


def test_new_unreviewed_semantic_type_fails_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fixture_module, "_SEMANTIC_TYPES", fixture_module._SEMANTIC_TYPES | {"new.event"})
    with pytest.raises(AssertionError, match="semantic event inventory"):
        fixture_module._producer_inventory()


def test_old_process_failure_is_not_silenced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KOLEGA_SCRATCHPAD", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["check", "--previous-python", str(tmp_path / "python")])
    monkeypatch.setattr(check, "installed_version", lambda *args: "1.0")
    monkeypatch.setattr(check, "create_fixture", lambda root: {"session_id": "fixture"})

    def fail(*args: Any, **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, "previous-release worker")

    monkeypatch.setattr(check.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        check.main()
