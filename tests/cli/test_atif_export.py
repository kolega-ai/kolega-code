"""ATIF v1.7 export: projection, v1 fallbacks, assets, security, atomicity."""

import base64
import json

import pytest

from kolega_code.cli.atif_export import (
    AtifExportError,
    AtifImagesNeedOutputError,
    export_atif_to_path,
    export_atif_to_text,
)
from kolega_code.cli.session_event_protocol import InMemorySessionJournal
from kolega_code.cli.session_journal import (
    TOOL_RESULT_PREVIEW_CHARS,
    SessionJournal,
    SessionRecorder,
    derive_root_agent_id,
)
from kolega_code.llm.ledger import HISTORY_ORIGIN, UsageLedger, llm_call_origin
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
)
from kolega_code.llm.usage import normalize_usage

# A 1x1 red PNG.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)

EXPORT_KW = dict(kolega_version="0.0.0-test", secret_values=(), state_dirs=())


def _usage(inp, out):
    return normalize_usage({"input_tokens": inp, "output_tokens": out}, "anthropic", "claude-opus-5")


def make_journal(tmp_path=None, session_id="atif-s1"):
    if tmp_path is None:
        return InMemorySessionJournal(session_id)
    return SessionJournal(session_id, tmp_path / "session")


def bootstrap(journal, *, project="/Users/evandempsey/git/x"):
    journal.append(
        "session.created",
        actor="system",
        payload={
            "metadata": {
                "mode": "code",
                "title": "t",
                "project_path": project,
                "config": {"long_provider": "anthropic", "long_model": "claude-opus-5", "thinking_effort": "high"},
            },
            "kolega_version": "0.27.1",
        },
        epoch_id="e1",
    )
    journal.append("context.epoch_started", actor="system", payload={"reason": "session_created"}, epoch_id="e1")
    return SessionRecorder(journal, recover=False)


def settled(ledger, message):
    with llm_call_origin(HISTORY_ORIGIN):
        request_id = ledger.begin("anthropic", "claude-opus-5")
        ledger.record_response(request_id, message.usage, message)
    return message


def convert(journal, **overrides):
    kwargs = {**EXPORT_KW, **overrides}
    return json.loads(export_atif_to_text(journal, **kwargs))


RICH_TURN_TOOLS = [
    {"name": "exec_command", "description": "Run a command.", "input_schema": {"type": "object"}},
    {
        "name": "apply_patch",
        "description": "Apply a patch.",
        "input_schema": {"type": "object"},
        "input_kind": "freeform",
    },
]


def drive_rich_turn(recorder, ledger):
    recorder.start_turn(Message(role="user", content=[TextBlock("run the tests")]))
    recorder.record_system_context("You are kolega.")
    recorder.record_session_context(Message(role="user", content=[TextBlock("runtime session context")]))
    recorder.record_tool_definitions(RICH_TURN_TOOLS)
    first = settled(
        ledger,
        Message(
            role="assistant",
            content=[
                ThinkingBlock("let me think", signature="OPAQUE-SIG"),
                TextBlock("Checking."),
                ToolCall(id="p1", name="exec_command", input={"command": "pytest"}, execution_id="tool_exec_1"),
                ToolCall(
                    id="p2",
                    name="apply_patch",
                    input="*** Begin Patch",
                    execution_id="tool_exec_2",
                    input_kind="freeform",
                ),
            ],
            stop_reason="tool_use",
            usage=_usage(100, 20),
        ),
    )
    recorder.record_assistant(first, reasoning_effort="high")
    # Results recorded out of call order: correlation is by id, never order.
    recorder.record_tool_results(
        [
            ToolResult(
                tool_use_id="p2",
                content="patched",
                name="apply_patch",
                is_error=False,
                execution_id="tool_exec_2",
                input_kind="freeform",
            ),
            ToolResult(
                tool_use_id="p1", content="1 passed", name="exec_command", is_error=False, execution_id="tool_exec_1"
            ),
        ]
    )
    second = settled(
        ledger,
        Message(role="assistant", content=[TextBlock("All green.")], stop_reason="end_turn", usage=_usage(150, 10)),
    )
    recorder.record_assistant(second)
    recorder.finish_turn("completed")


def test_rich_turn_projects_and_validates():
    journal = make_journal()
    recorder = bootstrap(journal)
    ledger = UsageLedger()
    drive_rich_turn(recorder, ledger)
    recorder.record_run_terminal("completed", {"status": "completed", "exit_code": 0})

    doc = convert(journal)
    assert doc["schema_version"] == "ATIF-v1.7"
    assert doc["session_id"] == "atif-s1"
    assert doc["trajectory_id"] == derive_root_agent_id("atif-s1")
    assert doc["agent"]["name"] == "kolega-code"
    assert doc["agent"]["model_name"] == "claude-opus-5"
    assert doc["agent"]["tool_definitions"] == RICH_TURN_TOOLS
    assert not any(w["code"] == "tool_definitions_unavailable" for w in doc["extra"]["conversion_warnings"])

    steps = doc["steps"]
    assert [step["step_id"] for step in steps] == list(range(1, len(steps) + 1))
    sources = [step["source"] for step in steps]
    assert sources == ["user", "system", "system", "agent", "agent"]
    assert steps[2]["message"] == "runtime session context"
    assert steps[2]["extra"]["kolega"]["origin"] == "context.session"

    agent_step = steps[3]
    assert agent_step["llm_call_count"] == 1
    assert agent_step["model_name"] == "claude-opus-5"
    assert agent_step["reasoning_effort"] == "high"
    assert agent_step["reasoning_content"] == "let me think"
    calls = agent_step["tool_calls"]
    assert [call["tool_call_id"] for call in calls] == ["tool_exec_1", "tool_exec_2"]
    assert calls[0]["function_name"] == "exec_command"
    assert calls[0]["arguments"] == {"command": "pytest"}
    # Free-form input becomes an object without losing the original text.
    assert calls[1]["arguments"] == {"input": "*** Begin Patch"}
    assert calls[1]["extra"]["kolega"]["input"] == "*** Begin Patch"

    results = agent_step["observation"]["results"]
    assert {result["source_call_id"] for result in results} == {"tool_exec_1", "tool_exec_2"}
    by_call = {result["source_call_id"]: result for result in results}
    assert by_call["tool_exec_1"]["content"] == "1 passed"

    assert agent_step["metrics"] == {"prompt_tokens": 100, "completion_tokens": 20}
    assert doc["final_metrics"]["total_prompt_tokens"] == 250
    assert doc["final_metrics"]["total_completion_tokens"] == 30
    assert doc["final_metrics"]["extra"]["llm_steps"] == 2

    # llm_call_id provenance retained per step; opaque state absent everywhere.
    assert agent_step["extra"]["kolega"]["llm_call_id"]
    assert "OPAQUE-SIG" not in json.dumps(doc)
    assert doc["extra"]["kolega"]["status"]["outcome"] == "completed"
    # Original identity is preserved on every step.
    assert all("seq" in step["extra"]["kolega"] for step in steps)


def test_synthetic_notice_step_has_no_llm_fields():
    journal = make_journal()
    recorder = bootstrap(journal)
    recorder.start_turn(Message(role="user", content=[TextBlock("hi")]))
    recorder.record_synthetic_assistant("Prompt blocked.", notice_code="hook_blocked")
    recorder.finish_turn("completed")

    doc = convert(journal)
    step = next(step for step in doc["steps"] if step["source"] == "agent")
    assert step["llm_call_count"] == 0
    assert "metrics" not in step
    assert "reasoning_content" not in step
    assert "model_name" not in step
    assert step["extra"]["kolega"]["notice_code"] == "hook_blocked"


def test_oversized_tool_result_is_fully_hydrated(tmp_path):
    journal = make_journal(tmp_path)
    recorder = bootstrap(journal)
    ledger = UsageLedger()
    big = "x" * (TOOL_RESULT_PREVIEW_CHARS + 500)
    recorder.start_turn(Message(role="user", content=[TextBlock("go")]))
    first = settled(
        ledger,
        Message(
            role="assistant",
            content=[ToolCall(id="p1", name="bash", input={}, execution_id="tool_exec_big")],
            stop_reason="tool_use",
            usage=_usage(10, 5),
        ),
    )
    recorder.record_assistant(first)
    recorder.record_tool_results(
        [ToolResult(tool_use_id="p1", content=big, name="bash", is_error=False, execution_id="tool_exec_big")]
    )
    recorder.finish_turn("completed")

    doc = convert(journal, state_dirs=(tmp_path,))
    step = next(step for step in doc["steps"] if step.get("tool_calls"))
    content = step["observation"]["results"][0]["content"]
    assert len(content) == len(big)
    assert "[Middle of tool result omitted" not in content
    assert str(tmp_path) not in json.dumps(doc)
    assert step["observation"]["results"][0]["extra"]["kolega"]["hydrated_from_artifact"]["chars"] == len(big)


def test_compaction_and_rewind_are_auditable_system_steps():
    journal = make_journal()
    recorder = bootstrap(journal)
    recorder.start_turn(Message(role="user", content=[TextBlock("first")]))
    recorder.record_assistant(Message(role="assistant", content=[TextBlock("one")], stop_reason="end_turn"))
    recorder.finish_turn("completed")
    recorder.record_compaction({"compacted_through": 2}, info={"trigger": "auto", "input_tokens_before": 90000})
    recorder.record_rewind(recorder.list_rewindable_turns()[0].turn_id)

    doc = convert(journal)
    compaction = next(s for s in doc["steps"] if s.get("extra", {}).get("context_management"))
    assert compaction["source"] == "system"
    assert compaction["extra"]["context_management"] == {"type": "compaction", "boundary": "replace"}
    assert compaction["extra"]["kolega"]["compaction"]["trigger"] == "auto"
    rewind = next(s for s in doc["steps"] if "rewind" in s.get("extra", {}).get("kolega", {}))
    assert rewind["source"] == "system"
    # Pre-boundary steps stay: the trajectory is an audit record.
    assert [s["source"] for s in doc["steps"]].count("user") == 1
    assert any(s["source"] == "agent" for s in doc["steps"])


def test_unmatched_tool_result_is_kept_with_a_warning():
    journal = make_journal()
    recorder = bootstrap(journal)
    recorder.start_turn(Message(role="user", content=[TextBlock("go")]))
    recorder.record_tool_results(
        [
            ToolResult(
                tool_use_id="ghost", content="orphaned", name="bash", is_error=True, execution_id="tool_exec_ghost"
            )
        ]
    )
    recorder.finish_turn("completed")

    doc = convert(journal)
    orphan = next(s for s in doc["steps"] if s.get("observation") and s["source"] == "system")
    result = orphan["observation"]["results"][0]
    assert result["content"] == "orphaned"
    assert "source_call_id" not in result
    assert result["extra"]["kolega"]["uncorrelated_call_id"] == "tool_exec_ghost"
    assert any(w["code"] == "unmatched_tool_result" for w in doc["extra"]["conversion_warnings"])


def test_subagent_trajectories_embed_with_refs():
    journal = make_journal()
    recorder = bootstrap(journal)
    ledger = UsageLedger()
    recorder.start_turn(Message(role="user", content=[TextBlock("dispatch")]))
    dispatch = settled(
        ledger,
        Message(
            role="assistant",
            content=[ToolCall(id="p1", name="dispatch_agent", input={"task": "explore"}, execution_id="tool_exec_d")],
            stop_reason="tool_use",
            usage=_usage(10, 5),
        ),
    )
    recorder.record_assistant(dispatch)

    child = recorder.scoped_child(agent_id="child-1", agent_name="explorer", parent_tool_call_id="tool_exec_d", depth=1)
    child.record_agent_started({"agent_name": "explorer", "task": "explore"}, turn_id=recorder.current_turn_id)
    child.start_turn(Message(role="user", content=[TextBlock("explore")]))
    child.record_assistant(
        settled(
            ledger, Message(role="assistant", content=[TextBlock("found")], stop_reason="end_turn", usage=_usage(5, 2))
        )
    )
    child.finish_turn("completed")
    child.record_agent_terminal("completed", {"summary": "found"})

    recorder.record_tool_results(
        [
            ToolResult(
                tool_use_id="p1", content="found", name="dispatch_agent", is_error=False, execution_id="tool_exec_d"
            )
        ]
    )
    recorder.finish_turn("completed")

    doc = convert(journal)
    subs = doc["subagent_trajectories"]
    assert [t["trajectory_id"] for t in subs] == ["child-1"]
    assert subs[0]["agent"]["extra"]["agent_name"] == "explorer"
    assert subs[0]["agent"]["extra"]["status"] == "completed"
    assert [s["step_id"] for s in subs[0]["steps"]] == [1, 2]
    # Every subagent trajectory starts its own step_id sequence and the parent
    # observation for the dispatch call references it.
    dispatch_step = next(s for s in doc["steps"] if s.get("tool_calls"))
    refs = [r for r in dispatch_step["observation"]["results"] if r.get("subagent_trajectory_ref")]
    assert refs and refs[0]["subagent_trajectory_ref"] == [{"trajectory_id": "child-1"}]
    assert refs[0]["source_call_id"] == "tool_exec_d"
    # Subagent LLM usage is counted in the subagent's own final metrics.
    assert subs[0]["final_metrics"]["total_prompt_tokens"] == 5


def test_v1_legacy_journal_exports_with_derived_identities(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    base = {
        "version": 1,
        "session_id": "legacy1",
        "epoch_id": "e1",
        "turn_id": "t1",
        "artifacts": [],
    }
    lines = [
        {
            **base,
            "id": "ev1",
            "seq": 1,
            "turn_id": None,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "actor": "system",
            "type": "session.created",
            "payload": {"metadata": {"mode": "code", "project_path": "/p", "config": {}}},
        },
        {
            **base,
            "id": "ev2",
            "seq": 2,
            "turn_id": None,
            "timestamp": "2026-01-01T00:00:01+00:00",
            "actor": "system",
            "type": "context.epoch_started",
            "payload": {"reason": "session_created"},
        },
        {
            **base,
            "id": "ev3",
            "seq": 3,
            "timestamp": "2026-01-01T00:00:02+00:00",
            "actor": "user",
            "type": "turn.started",
            "payload": {"message": {"role": "user", "content": [{"type": "text", "text": "hello"}]}},
        },
        {
            **base,
            "id": "ev4",
            "seq": 4,
            "timestamp": "2026-01-01T00:00:03+00:00",
            "actor": "assistant",
            "type": "assistant.message",
            "payload": {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "provider": "anthropic",
                        "model": "m",
                        "reported": True,
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                        "cache_read_input_tokens": None,
                        "cache_write_input_tokens": None,
                        "reasoning_output_tokens": None,
                        "unavailable_reason": None,
                    },
                }
            },
        },
        {
            **base,
            "id": "ev5",
            "seq": 5,
            "timestamp": "2026-01-01T00:00:04+00:00",
            "actor": "assistant",
            "type": "llm.message",
            "payload": {
                "request_id": "req-sub-1",
                "run_id": "r1",
                "provider": "anthropic",
                "model": "m",
                "origin": {
                    "kind": "sub_agent",
                    "agent_name": "investigator",
                    "agent_id": "sub-legacy",
                    "parent_tool_call_id": "tool_exec_v1",
                    "depth": 1,
                },
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "sub answer"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "provider": "anthropic",
                        "model": "m",
                        "reported": True,
                        "input_tokens": 4,
                        "output_tokens": 2,
                        "total_tokens": 6,
                        "cache_read_input_tokens": None,
                        "cache_write_input_tokens": None,
                        "reasoning_output_tokens": None,
                        "unavailable_reason": None,
                    },
                },
            },
        },
        {
            **base,
            "id": "ev6",
            "seq": 6,
            "timestamp": "2026-01-01T00:00:05+00:00",
            "actor": "system",
            "type": "turn.completed",
            "payload": {},
        },
    ]
    (session_dir / "events.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    journal = SessionJournal("legacy1", session_dir)

    doc = convert(journal)
    doc_again = convert(journal)
    # Derived identities are deterministic across conversions.
    assert doc["trajectory_id"] == doc_again["trajectory_id"] == derive_root_agent_id("legacy1")
    root_agent_step = next(s for s in doc["steps"] if s["source"] == "agent")
    again_agent_step = next(s for s in doc_again["steps"] if s["source"] == "agent")
    assert root_agent_step["extra"]["kolega"]["llm_call_id"] == again_agent_step["extra"]["kolega"]["llm_call_id"]
    assert root_agent_step["metrics"] == {"prompt_tokens": 7, "completion_tokens": 3}

    subs = doc["subagent_trajectories"]
    assert [t["trajectory_id"] for t in subs] == ["sub-legacy"]
    assert subs[0]["steps"][0]["extra"]["kolega"]["llm_call_id"] == "req-sub-1"

    codes = {w["code"] for w in doc["extra"]["conversion_warnings"]}
    assert {
        "v1_source_journal",
        "v1_derived_agent_id",
        "v1_derived_llm_call_id",
        "v1_subagent_tools_unrecorded",
    } <= codes
    assert "v1_source_journal" in doc["notes"]
    # v1 sessions never recorded tool schemas: omitted with a warning.
    assert "tool_definitions" not in doc["agent"]
    assert "tool_definitions_unavailable" in codes


def test_secrets_and_state_dir_are_scrubbed(tmp_path):
    journal = make_journal(tmp_path)
    recorder = bootstrap(journal)
    recorder.start_turn(
        Message(role="user", content=[TextBlock(f"key sk-TOPSECRET at {tmp_path}/session/artifacts/x")])
    )
    recorder.finish_turn("completed")

    doc = convert(journal, secret_values=("sk-TOPSECRET",), state_dirs=(tmp_path,))
    rendered = json.dumps(doc)
    assert "sk-TOPSECRET" not in rendered
    assert str(tmp_path) not in rendered
    assert "<state-dir>" in rendered


def test_image_assets_are_copied_hash_verified_and_relative(tmp_path):
    journal = make_journal(tmp_path / "j")
    recorder = bootstrap(journal)
    recorder.start_turn(
        Message(
            role="user",
            content=[TextBlock("look"), ImageBlock("base64", "image/png", base64.b64encode(_PNG).decode())],
        )
    )
    recorder.finish_turn("completed")

    out = tmp_path / "export" / "trajectory.json"
    export_atif_to_path(journal, output=out, **EXPORT_KW)
    doc = json.loads(out.read_text())
    parts = doc["steps"][0]["message"]
    image = next(part for part in parts if part["type"] == "image")
    assert image["source"]["media_type"] == "image/png"
    assert image["source"]["path"].startswith("trajectory.assets/")
    asset = out.parent / image["source"]["path"]
    assert asset.read_bytes() == _PNG
    import hashlib

    assert asset.name.split(".")[0] == hashlib.sha256(_PNG).hexdigest()
    # Provider-opaque artifact purposes are structurally unreachable.
    assert [p.name for p in sorted((out.parent / "trajectory.assets").iterdir())] == [asset.name]


def test_stdout_export_with_images_refuses_before_writing(tmp_path):
    journal = make_journal(tmp_path)
    recorder = bootstrap(journal)
    recorder.start_turn(
        Message(role="user", content=[ImageBlock("base64", "image/png", base64.b64encode(_PNG).decode())])
    )
    recorder.finish_turn("completed")

    with pytest.raises(AtifImagesNeedOutputError):
        export_atif_to_text(journal, **EXPORT_KW)


def test_failed_export_leaves_previous_output_intact(tmp_path, monkeypatch):
    journal = make_journal(tmp_path / "j")
    recorder = bootstrap(journal)
    recorder.start_turn(Message(role="user", content=[TextBlock("one")]))
    recorder.finish_turn("completed")

    out = tmp_path / "trajectory.json"
    export_atif_to_path(journal, output=out, **EXPORT_KW)
    original = out.read_text()

    from kolega_code.cli import atif_export as module

    def boom(document):
        raise AtifExportError("validation forced to fail")

    monkeypatch.setattr(module, "validate_atif_document", boom)
    with pytest.raises(AtifExportError):
        export_atif_to_path(journal, output=out, **EXPORT_KW)
    assert out.read_text() == original
    leftovers = [p.name for p in out.parent.iterdir() if p.name.startswith(".")]
    assert not [name for name in leftovers if "tmp-atif" in name or ".old-" in name]


def test_unknown_event_types_are_recorded_not_fatal():
    journal = make_journal()
    bootstrap(journal)
    journal.append("kolega.future_event", actor="system", payload={"x": 1})

    doc = convert(journal)
    assert doc["extra"]["kolega"]["unknown_events"] == [{"type": "kolega.future_event", "seq": 3}]
    assert any(w["code"] == "unknown_event_type" for w in doc["extra"]["conversion_warnings"])


def test_usage_is_never_fabricated():
    journal = make_journal()
    recorder = bootstrap(journal)
    recorder.start_turn(Message(role="user", content=[TextBlock("go")]))
    recorder.record_assistant(Message(role="assistant", content=[TextBlock("no usage")], stop_reason="end_turn"))
    recorder.finish_turn("completed")

    doc = convert(journal)
    step = next(s for s in doc["steps"] if s["source"] == "agent")
    assert "metrics" not in step
    assert "total_prompt_tokens" not in (doc.get("final_metrics") or {})
    codes = {w["code"] for w in doc["extra"]["conversion_warnings"]}
    assert "usage_missing_for_llm_step" in codes and "usage_totals_partial" in codes


def test_continuation_turn_produces_no_user_step():
    journal = make_journal()
    recorder = bootstrap(journal)
    ledger = UsageLedger()
    recorder.start_continuation_turn()
    recorder.record_system_context("You are kolega.")
    message = settled(
        ledger,
        Message(role="assistant", content=[TextBlock("resumed")], stop_reason="end_turn", usage=_usage(10, 5)),
    )
    recorder.record_assistant(message, reasoning_effort="high")
    recorder.finish_turn("completed")

    doc = convert(journal)

    # No fabricated empty user utterance; the boundary survives in extra.
    assert [step["source"] for step in doc["steps"]].count("user") == 0
    continuations = doc["extra"]["kolega"]["continuation_turns"]
    assert len(continuations) == 1 and continuations[0]["turn_id"]
    agent_steps = [step for step in doc["steps"] if step["source"] == "agent"]
    assert len(agent_steps) == 1 and agent_steps[0]["message"] == "resumed"
