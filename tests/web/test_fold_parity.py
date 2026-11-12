"""The Python and JavaScript folds must agree, event for event.

Two implementations of one projection is a real risk: the TUI and server render
from Python, the player and web client render from JavaScript, and a silent
divergence would mean a shared replay showed something the session did not do.
Rather than trust review, both are folded over the same fixtures here and the
emitted state is compared.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from kolega_code.events import AgentEvent, ArtifactRef, KnownEventType
from kolega_code.session.inbox import PeerMessage, peer_model_text
from kolega_code.session.projection import replay

ASSET_DIR = Path(__file__).resolve().parents[2] / "kolega_code" / "web" / "assets"
TASK_LIST_FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures" / "task_list.json").read_text(encoding="utf-8"))

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is required for fold parity")


def _event(event_type: str, seq: int, *, elapsed_ms: int = 0, uuid: str | None = None, **content) -> AgentEvent:
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


def _sub_agent(event: AgentEvent, **info) -> AgentEvent:
    event.sub_agent_info = dict(info)
    return event


def _fixture_events() -> list[AgentEvent]:
    """Exercise every handler, including the paths most likely to drift."""
    peer_message = PeerMessage(
        message_id="peer-1",
        sender_session_id="session-alpha",
        sender_title="alpha",
        text="Please inspect the failing test.",
    )
    artifact = ArtifactRef(
        sha256="b" * 64,
        bytes=42,
        media_type="text/plain; charset=utf-8",
        purpose="tool_result",
        encoding="utf-8",
        chars=42,
    )
    tool_result = _event(
        KnownEventType.CHAT_MESSAGE,
        11,
        elapsed_ms=1100,
        message_type="tool_result",
        text="3 files changed",
        tool_call_id="call-1",
    )
    tool_result.artifacts = [artifact]
    return [
        _event(KnownEventType.TURN_STARTED, 1, elapsed_ms=0, turn_id="t1", user_text="ship the feature"),
        _event(KnownEventType.THINKING_DELTA, 2, elapsed_ms=90, uuid="th", text="Considering ", complete=False),
        _event(KnownEventType.THINKING_DELTA, 3, elapsed_ms=140, uuid="th", text="options.", complete=True),
        _event(KnownEventType.ASSISTANT_DELTA, 4, elapsed_ms=200, uuid="as", text="Starting ", complete=False),
        _event(KnownEventType.ASSISTANT_DELTA, 5, elapsed_ms=260, uuid="as", text="now.", complete=True),
        _event(
            KnownEventType.CHAT_MESSAGE,
            6,
            elapsed_ms=300,
            message_type="tool_call",
            text="Editing",
            tool_description="edit",
            tool_call_id="call-1",
        ),
        _event(
            KnownEventType.TOOL_STREAMING_UPDATE,
            7,
            elapsed_ms=350,
            tool_call_id="call-1",
            text="patching",
            stream_mode="append",
        ),
        _event(
            KnownEventType.FILE_EDIT_PREVIEW,
            8,
            elapsed_ms=380,
            path="src/app.py",
            diff="@@ -1 +1 @@",
            tool_call_id="call-1",
        ),
        _event(KnownEventType.TERMINAL_COMMAND, 9, elapsed_ms=400, command="pytest -q"),
        _event(KnownEventType.TERMINAL_OUTPUT, 10, elapsed_ms=450, output="12 passed\n"),
        tool_result,
        _event(
            KnownEventType.LLM_CONTEXT_UPDATE,
            12,
            elapsed_ms=500,
            input_tokens=4096,
            max_tokens=200000,
            usage_percentage=2.0,
            alert_level="ok",
            message=None,
            will_compress_at=160000,
        ),
        _event(KnownEventType.COMPACTION_STATUS, 13, elapsed_ms=520, phase="started", message="Compacting"),
        _event(KnownEventType.LLM_STATUS_UPDATE, 14, elapsed_ms=540, status="overloaded", message="Retrying"),
        _event(KnownEventType.LOG_MESSAGE, 15, elapsed_ms=560, level="warning", text="slow tool"),
        _event(KnownEventType.BROWSER_LAUNCHED, 16, elapsed_ms=600, browser_id="b1"),
        _event(KnownEventType.BROWSER_CLOSED, 17, elapsed_ms=620, browser_id="b1"),
        _sub_agent(
            _event(KnownEventType.ASSISTANT_DELTA, 18, elapsed_ms=700, uuid="sub", text="delegated", complete=True),
            dispatch_id="d1",
            agent_name="investigator",
            task="trace it",
        ),
        _sub_agent(
            _event(KnownEventType.COMPACTION_STATUS, 19, elapsed_ms=720, phase="finished"),
            dispatch_id="d1",
        ),
        _event("an_unknown_future_event", 20, elapsed_ms=740, text="ignored"),
        _event(KnownEventType.SYSTEM_MESSAGE, 21, elapsed_ms=760, text="note"),
        # A permission round trip, plus one prompt left outstanding, so both the
        # resolved and pending branches of the prompt fold are compared.
        _event(
            KnownEventType.CONTROL_REQUESTED,
            22,
            elapsed_ms=770,
            request_id="req-1",
            kind="permission",
            payload={"command": "rm -rf build"},
            has_controller=True,
        ),
        _event(
            KnownEventType.CONTROL_RESOLVED,
            23,
            elapsed_ms=780,
            request_id="req-1",
            kind="permission",
            payload={"command": "rm -rf build"},
            response={"allowed": True},
            reason="answered",
        ),
        _event(
            KnownEventType.CONTROL_REQUESTED,
            24,
            elapsed_ms=790,
            request_id="req-2",
            kind="question",
            payload={"question": "which option?"},
            has_controller=True,
        ),
        _event(KnownEventType.STREAM_TRUNCATED, 25, elapsed_ms=800, reason="retention_limit"),
        _event(KnownEventType.TURN_ENDED, 26, elapsed_ms=900, turn_id="t1", status="completed"),
        # A turn whose reasoning and prose interleave. Segments are keyed by uuid,
        # so each stays open across the other's deltas; a fold that keyed on
        # recency instead would agree on the turn above and diverge here.
        _event(KnownEventType.TURN_STARTED, 27, elapsed_ms=1000, turn_id="t2", user_text="again"),
        _event(KnownEventType.THINKING_DELTA, 28, elapsed_ms=1010, uuid="th2", text="Weighing ", complete=False),
        _event(KnownEventType.ASSISTANT_DELTA, 29, elapsed_ms=1020, uuid="as2", text="Doing ", complete=False),
        _event(KnownEventType.THINKING_DELTA, 30, elapsed_ms=1030, uuid="th2", text="the options.", complete=True),
        _event(KnownEventType.ASSISTANT_DELTA, 31, elapsed_ms=1040, uuid="as2", text="it now.", complete=True),
        # A dispatched agent's own turn: it must not reach the turn rail or the
        # activity flag, but its task and reasoning belong to its trajectory.
        _sub_agent(
            _event(KnownEventType.TURN_STARTED, 32, elapsed_ms=1042, turn_id="sub-t", user_text="trace it"),
            dispatch_id="d2",
            agent_name="investigator",
            task="trace it",
        ),
        _sub_agent(
            _event(KnownEventType.THINKING_DELTA, 33, elapsed_ms=1044, uuid="subth", text="Looking.", complete=True),
            dispatch_id="d2",
        ),
        _sub_agent(
            _event(KnownEventType.TURN_ENDED, 34, elapsed_ms=1046, turn_id="sub-t", status="completed"),
            dispatch_id="d2",
        ),
        _sub_agent(
            _event(KnownEventType.CHAT_MESSAGE, 35, elapsed_ms=1048, status="STOPPED", message="done"),
            dispatch_id="d2",
        ),
        _event(KnownEventType.TURN_ENDED, 36, elapsed_ms=1050, turn_id="t2", status="completed"),
        # A prose segment that only ever carries an empty final delta, which the
        # agent emits on every iteration that ended in a tool call.
        _event(KnownEventType.ASSISTANT_DELTA, 37, elapsed_ms=1060, uuid="empty", text="", complete=True),
        # A gigacode run: lifecycle messages whose detail lives in named fields,
        # and two parallel agents that share a name and differ only by agent_id.
        _event(
            KnownEventType.CHAT_MESSAGE,
            38,
            elapsed_ms=1070,
            message_type="workflow_start",
            workflow_run_id="run-1",
            name="triage",
            description="classify in parallel",
            text="",
        ),
        _event(
            KnownEventType.CHAT_MESSAGE,
            39,
            elapsed_ms=1072,
            message_type="workflow_phase",
            workflow_run_id="run-1",
            text="Classify",
        ),
        _sub_agent(
            _event(KnownEventType.ASSISTANT_DELTA, 40, elapsed_ms=1074, uuid="wfa", text="alpha done", complete=True),
            agent_id="wf-alpha",
            agent_name="general-agent",
            label="alpha",
            phase="Classify",
            task="classify alpha",
            workflow_run_id="run-1",
        ),
        _sub_agent(
            _event(KnownEventType.ASSISTANT_DELTA, 41, elapsed_ms=1076, uuid="wfb", text="beta done", complete=True),
            agent_id="wf-beta",
            agent_name="general-agent",
            label="beta",
            phase="Classify",
            task="classify beta",
            workflow_run_id="run-1",
        ),
        _event(
            KnownEventType.CHAT_MESSAGE,
            42,
            elapsed_ms=1078,
            message_type="workflow_end",
            workflow_run_id="run-1",
            status="failed",
            error="budget exhausted",
            text="",
        ),
        _event(
            KnownEventType.PEER_MESSAGE_RECEIVED,
            43,
            elapsed_ms=1080,
            **peer_message.event_content(),
        ),
        _event(
            KnownEventType.PEER_MESSAGE_DELIVERED,
            44,
            elapsed_ms=1082,
            **peer_message.event_content(),
        ),
        _event(
            KnownEventType.PEER_MESSAGE_RECEIVED,
            45,
            elapsed_ms=1084,
            message_id="peer-1",
        ),
        _event(
            KnownEventType.TURN_STARTED,
            46,
            elapsed_ms=1086,
            turn_id="peer-turn",
            user_text=peer_model_text(peer_message),
        ),
        _event(
            KnownEventType.TURN_ENDED,
            47,
            elapsed_ms=1088,
            turn_id="peer-turn",
            status="completed",
        ),
        _sub_agent(
            _event(
                KnownEventType.LLM_STATUS_UPDATE,
                48,
                elapsed_ms=1090,
                status="retrying",
                message="Retrying delegated request",
            ),
            dispatch_id="d2",
        ),
        _sub_agent(
            _event(KnownEventType.LLM_STATUS_UPDATE, 49, elapsed_ms=1092, status="", message=""),
            dispatch_id="d2",
        ),
    ]


def _run_js_fold(events: list[dict], tmp_path: Path) -> dict:
    """Fold events with the shipped player module and return its state as JSON."""
    harness = tmp_path / "harness.mjs"
    harness.write_text(
        "\n".join(
            [
                f"import {{ replay, toDict }} from {json.dumps(str(ASSET_DIR / 'fold.js'))};",
                "import { readFileSync } from 'node:fs';",
                "const events = JSON.parse(readFileSync(process.argv[2], 'utf8'));",
                "process.stdout.write(JSON.stringify(toDict(replay(events))));",
            ]
        ),
        encoding="utf-8",
    )
    payload = tmp_path / "events.json"
    payload.write_text(json.dumps(events), encoding="utf-8")
    result = subprocess.run(
        ["node", str(harness), str(payload)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"node fold failed: {result.stderr}"
    return json.loads(result.stdout)


def test_python_and_javascript_folds_agree(tmp_path: Path) -> None:
    events = _fixture_events()
    expected = replay(events).to_dict()

    actual = _run_js_fold([event.model_dump(mode="json") for event in events], tmp_path)

    assert actual == expected, "the JavaScript fold diverged from the Python fold"
    assert actual["sub_agents"]["d2"]["status"] == "completed"
    assert actual["sub_agents"]["d2"]["notice"] == "Retrying delegated request"


def test_folds_agree_at_every_prefix(tmp_path: Path) -> None:
    """Seeking must agree too, not just the final state."""
    events = _fixture_events()
    serialized = [event.model_dump(mode="json") for event in events]
    for cut in (1, 5, 11, 18, 22, 30, 38, 41, 43, 44, 45, 46, 48, len(events)):
        expected = replay(events[:cut]).to_dict()
        actual = _run_js_fold(serialized[:cut], tmp_path)
        assert actual == expected, f"folds diverged after {cut} events"


def test_subagent_linkage_independent_expected_state_at_every_prefix(tmp_path: Path) -> None:
    fixture = json.loads((Path(__file__).parents[1] / "fixtures" / "subagent_linkage.json").read_text())
    expected = deepcopy(fixture["initial"])
    expected["task_list"] = None
    expected["goal"] = None
    expected_prefixes = [deepcopy(expected)]
    events = []
    for seq, step in enumerate(fixture["steps"], 1):
        event = AgentEvent(session_id="s", sender="agent", seq=seq, elapsed_ms=seq * 10, **step["event"])
        events.append(event.model_dump(mode="json"))
        patch = deepcopy(step["expect"])
        for key, updates in patch.pop("sub_agents", {}).items():
            if key not in expected["sub_agents"]:
                expected["sub_agents"][key] = {**deepcopy(fixture["agent_defaults"]), "key": key, "name": key}
            expected["sub_agents"][key].update(updates)
        expected.update(patch)
        expected.update(last_seq=seq, elapsed_ms=seq * 10)
        expected_prefixes.append(deepcopy(expected))
    harness = tmp_path / "linkage-prefixes.mjs"
    payload = tmp_path / "linkage-events.json"
    payload.write_text(json.dumps(events), encoding="utf-8")
    harness.write_text(
        f"""
import {{ emptyState, fold, replay, toDict }} from {json.dumps(str(ASSET_DIR / "fold.js"))};
import {{ readFileSync }} from "node:fs";
const events = JSON.parse(readFileSync(process.argv[2], "utf8"));
const state = emptyState();
const incremental = [toDict(state)];
const seek = [toDict(replay([]))];
for (let index = 0; index < events.length; index++) {{
  fold(state, events[index]);
  incremental.push(toDict(state));
  seek.push(toDict(replay(events.slice(0, index + 1))));
}}
process.stdout.write(JSON.stringify({{ incremental, seek }}));
""",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(harness), str(payload)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    for cut, expected in enumerate(expected_prefixes):
        assert actual["incremental"][cut] == expected, f"JavaScript incremental prefix {cut}"
        assert actual["seek"][cut] == expected, f"JavaScript seek prefix {cut}"
        assert replay([AgentEvent.model_validate(event) for event in events[:cut]]).to_dict() == expected, (
            f"Python prefix {cut}"
        )


def test_task_list_independent_expected_state_at_every_prefix(tmp_path: Path) -> None:
    expected = deepcopy(TASK_LIST_FIXTURE["initial"])
    expected["goal"] = None
    expected_prefixes = [deepcopy(expected)]
    events = []
    for step in TASK_LIST_FIXTURE["steps"]:
        event = AgentEvent(session_id="s", sender="agent", **step["event"])
        events.append(event)
        expected.update(deepcopy(step["expect"]))
        expected_prefixes.append(deepcopy(expected))
    harness = tmp_path / "task-list-prefixes.mjs"
    payload = tmp_path / "task-list-events.json"
    payload.write_text(json.dumps([event.model_dump(mode="json") for event in events]), encoding="utf-8")
    harness.write_text(
        f"""
import {{ emptyState, fold, replay, toDict }} from {json.dumps(str(ASSET_DIR / "fold.js"))};
import {{ readFileSync }} from "node:fs";
const events = JSON.parse(readFileSync(process.argv[2], "utf8"));
const state = emptyState();
const incremental = [toDict(state)];
const seek = [toDict(replay([]))];
for (let index = 0; index < events.length; index++) {{
  fold(state, events[index]);
  incremental.push(toDict(state));
  seek.push(toDict(replay(events.slice(0, index + 1))));
}}
process.stdout.write(JSON.stringify({{ incremental, seek }}));
""",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(harness), str(payload)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    for cut, expected in enumerate(expected_prefixes):
        assert actual["incremental"][cut] == expected, f"JavaScript incremental prefix {cut}"
        assert actual["seek"][cut] == expected, f"JavaScript seek prefix {cut}"
        assert replay(events[:cut]).to_dict() == expected, f"Python seek prefix {cut}"


def test_task_list_parser_and_full_text_in_javascript(tmp_path: Path) -> None:
    cases = deepcopy(TASK_LIST_FIXTURE["parser_cases"])
    text = ('完成 🐍 <tag> & "quoted" é ' * 100).strip()
    for prefix, status in [
        ("- [x]", "completed"),
        ("- [X]", "completed"),
        ("- [ ]", "pending"),
        ("-", "pending"),
        ("", "pending"),
    ]:
        cases.append({"markdown": f" \t{prefix} {text}\n ", "entries": [{"text": text, "status": status}]})
    payload = tmp_path / "task-list-parser.json"
    payload.write_text(json.dumps(cases), encoding="utf-8")
    harness = tmp_path / "task-list-parser.mjs"
    harness.write_text(
        f"""
import assert from "node:assert/strict";
import {{ emptyState, fold, toDict, taskListEntries }} from {json.dumps(str(ASSET_DIR / "fold.js"))};
import {{ readFileSync }} from "node:fs";
for (const testCase of JSON.parse(readFileSync(process.argv[2], "utf8"))) {{
  assert.deepEqual(taskListEntries(testCase.markdown), testCase.entries, testCase.name);
  const expected = {{ markdown: testCase.markdown, entries: testCase.entries }};
  for (const legacy of [false, true]) {{
    const state = emptyState();
    fold(state, {{
      event_type: legacy ? "chat_message" : "task_list_updated",
      content: legacy ? {{ message_type: "task_list_update", text: testCase.markdown }} :
        {{ markdown: testCase.markdown }},
    }});
    assert.deepEqual(state.taskList, expected, testCase.name);
    assert.deepEqual(toDict(state).task_list, expected, testCase.name);
    assert.deepEqual(state.conversation, []);
    assert.deepEqual([...state.unknownEventTypes], []);
  }}
}}
""",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(harness), str(payload)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_task_update_seq_rejection_in_javascript(tmp_path: Path) -> None:
    harness = tmp_path / "task-list-seq.mjs"
    harness.write_text(
        f"""
import assert from "node:assert/strict";
import {{ emptyState, fold, toDict }} from {json.dumps(str(ASSET_DIR / "fold.js"))};
for (const seq of [1, 3]) {{
  for (const worker of [false, true]) {{
    for (const [eventType, content] of [
      ["task_list_updated", {{ markdown: "- replacement" }}],
      ["task_list_updated", {{ markdown: null }}],
      ["chat_message", {{ message_type: "task_list_update", text: "" }}],
      ["chat_message", {{ message_type: "task_list_update", text: false }}],
    ]) {{
      const state = emptyState();
      fold(state, {{ event_type: "task_list_updated", content: {{ markdown: "- known" }}, seq: 3 }});
      const expected = toDict(state);
      assert.throws(() => fold(state, {{
        event_type: eventType, content, seq, elapsed_ms: 100,
        sub_agent_info: worker ? {{ agent_id: "ignored-worker" }} : null,
      }}), /ascending order/);
      assert.deepEqual(toDict(state), expected);
    }}
  }}
}}
""",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(harness)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_worker_metadata_validation_and_every_handler_in_javascript(tmp_path: Path) -> None:
    harness = tmp_path / "worker-metadata.mjs"
    harness.write_text(
        f"""
import assert from "node:assert/strict";
import {{ emptyState, fold, toDict }} from {json.dumps(str(ASSET_DIR / "fold.js"))};
const types = [
  "assistant_delta", "thinking_delta", "chat_message", "tool_streaming_update",
  "terminal_command", "terminal_output", "terminal_launched", "terminal_closed",
  "log_message", "file_edit_preview", "llm_context_update", "compaction_status",
  "llm_status_update", "status_update", "turn_started", "turn_ended",
  "browser_launched", "browser_closed", "stream_truncated", "control_requested",
  "control_resolved", "system_message", "memory_suggestions", "credit_alert",
  "llm_error", "llm_request", "future_worker",
];
for (const eventType of types) {{
  const state = emptyState();
  fold(state, {{ event_type: "llm_request", seq: 1, content: {{}}, sub_agent_info: {{ dispatch_id: "d" }} }});
  fold(state, {{ event_type: eventType, seq: 2, content: {{}}, sub_agent_info: {{
    dispatch_id: "d", agent_id: "actual", agent_name: "late name", task: "late task",
    label: "late label", phase: "late phase", workflow_run_id: "workflow",
    parent_agent_id: "root", parent_tool_call_id: "call", depth: 3, background: true,
    requested_routing: {{ model: "requested", provider: "fake" }},
    effective_routing: {{ model: "effective" }}, total_tokens: 5,
  }} }});
  const activity = state.subAgents.get("d");
  assert.equal(activity.agentId, "actual", eventType);
  assert.equal(activity.name, "late name", eventType);
  assert.equal(activity.task, "late task", eventType);
  assert.equal(activity.label, "late label", eventType);
  assert.equal(activity.phase, "late phase", eventType);
  assert.equal(activity.workflowRunId, "workflow", eventType);
  assert.equal(activity.parentAgentId, "root", eventType);
  assert.equal(activity.parentToolCallId, "call", eventType);
  assert.equal(activity.depth, 3, eventType);
  assert.equal(activity.background, true, eventType);
  assert.equal(activity.totalTokens, 5, eventType);
  fold(state, {{ event_type: eventType, seq: 3, content: {{ total_tokens: 0 }}, sub_agent_info: {{
    dispatch_id: "d", background: false, requested_routing: {{ model: "revised" }}, effective_routing: {{}},
  }} }});
  const serialized = toDict(state).sub_agents.d;
  assert.equal(activity.agentId, "actual", eventType);
  assert.equal(serialized.agent_id, "actual", eventType);
  assert.equal(serialized.parent_agent_id, "root", eventType);
  assert.equal(serialized.parent_tool_call_id, "call", eventType);
  assert.equal(serialized.name, "late name", eventType);
  assert.equal(serialized.task, "late task", eventType);
  assert.equal(serialized.label, "late label", eventType);
  assert.equal(serialized.phase, "late phase", eventType);
  assert.equal(serialized.workflow_run_id, "workflow", eventType);
  assert.equal(serialized.depth, 3, eventType);
  assert.equal(serialized.background, false, eventType);
  assert.equal(serialized.total_tokens, 0, eventType);
  assert.deepEqual(serialized.requested_routing, {{ model: "revised", provider: "fake" }}, eventType);
  assert.deepEqual(serialized.effective_routing, {{ model: "effective" }}, eventType);
}}
const cases = [
  ["agent_id", [null, true, false, 2, [], {{}}, ""]],
  ["parent_agent_id", [null, true, false, 2, [], {{}}, ""]],
  ["parent_tool_call_id", [null, true, false, 2, [], {{}}, ""]],
  ["agent_name", [null, true, false, 2, [], {{}}, ""]],
  ["task", [null, true, false, 2, [], {{}}, ""]],
  ["label", [null, true, false, 2, [], {{}}, ""]],
  ["phase", [null, true, false, 2, [], {{}}, ""]],
  ["workflow_run_id", [null, true, false, 2, [], {{}}, ""]],
  ["depth", [null, true, false, 0, -1, 1.5, "2", [], {{}}, NaN, Infinity]],
  ["background", [null, 0, 1, "false", [], {{}}]],
  ["total_tokens", [null, true, false, -1, 1.5, "2", [], {{}}, NaN, Infinity]],
  ["requested_routing", [null, true, false, 0, "route", []]],
  ["effective_routing", [null, true, false, 0, "route", []]],
];
for (const [field, values] of cases) {{
  for (const value of values) {{
    const state = emptyState();
    fold(state, {{ event_type: "llm_request", seq: 1, content: {{}}, sub_agent_info: {{
      dispatch_id: "d", agent_id: "actual", agent_name: "name", task: "task", label: "label",
      phase: "phase", workflow_run_id: "workflow", parent_agent_id: "parent", parent_tool_call_id: "call",
      depth: 2, background: true, total_tokens: 4,
      requested_routing: {{ model: "requested" }}, effective_routing: {{ model: "effective" }},
    }} }});
    const expected = toDict(state).sub_agents.d;
    const content = field === "total_tokens" ? {{ total_tokens: value }} : {{}};
    fold(state, {{ event_type: "llm_request", seq: 2, content,
      sub_agent_info: {{ dispatch_id: "d", [field]: value }} }});
    assert.deepEqual(toDict(state).sub_agents.d, expected, field);
  }}
}}
for (const info of [null, [], "bad", 1, false, {{}}]) {{
  const state = emptyState();
  fold(state, {{ event_type: "terminal_output", seq: 1, content: {{ output: "root" }}, sub_agent_info: info }});
  assert.equal(state.terminal, "root");
  assert.equal(state.subAgents.size, 0);
}}
for (const [info, key] of [
  [{{ dispatch_id: {{}}, agent_id: "actual" }}, "actual"],
  [{{ dispatch_id: true, agent_id: [], agent_name: "legacy" }}, "legacy"],
  [{{ dispatch_id: 5, agent_id: false, agent_name: {{}} }}, "sub-agent"],
]) {{
  const state = emptyState();
  fold(state, {{ event_type: "llm_request", seq: 1, content: {{}}, sub_agent_info: info }});
  assert.deepEqual([...state.subAgents.keys()], [key]);
}}
""",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(harness)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_terminal_caps_and_linear_append_in_javascript(tmp_path: Path) -> None:
    harness = tmp_path / "worker-terminals.mjs"
    harness.write_text(
        f"""
import assert from "node:assert/strict";
import {{
  emptyState, fold, toDict, TERMINAL_BUFFER_CHARS as cap,
  TERMINAL_MAX_PARTS as maxParts, TERMINAL_MAX_DROPPED_PREFIX as maxPrefix,
}} from {json.dumps(str(ASSET_DIR / "fold.js"))};
let seq = 0;
const state = emptyState();
function event(type, content, key) {{
  return {{ event_type: type, seq: ++seq, content, sub_agent_info: key ? {{ agent_id: key }} : null }};
}}
fold(state, event("turn_started", {{ turn_id: "root", user_text: "coordinate" }}));
fold(state, event("terminal_output", {{ output: "root" }}));
fold(state, event("terminal_output", {{ output: "parallel" }}, "other"));
let expected = "";
for (const chunk of ["a".repeat(cap + maxPrefix + 1), "b".repeat(7), "c".repeat(cap / 2)]) {{
  fold(state, event("terminal_output", {{ output: chunk }}, "child"));
  expected = (expected + chunk).slice(-cap);
  const child = state.subAgents.get("child");
  assert.equal(child.terminal, expected);
  assert.equal(child.terminalTruncated, true);
  assert.equal(child._terminalChars, cap);
  assert.ok(child._terminalOffset <= maxPrefix);
  assert.equal(state.terminal, "root");
  assert.equal(state.terminalTruncated, false);
  assert.equal(state.activity, "generating");
  assert.equal(state.subAgents.get("other").terminal, "parallel");
  assert.equal(state.subAgents.get("other").terminalTruncated, false);
  assert.deepEqual(child.steps, []);
  assert.equal(toDict(state).sub_agents.child.terminal, expected);
  assert.equal(toDict(state).sub_agents.child.terminal_truncated, true);
}}
const child = state.subAgents.get("child");
child.terminal = "replacement";
fold(state, event("terminal_output", {{ display_output: " tail", output: "hidden" }}, "child"));
assert.equal(child.terminal, "replacement tail");
fold(state, event("terminal_command", {{ command: "child-check" }}, "child"));
assert.equal(child.terminal, "replacement tail$ child-check\\n");
assert.equal(state.activity, "generating");
for (const key of [null, "performance-child"]) {{
  const state = emptyState();
  fold(state, event("terminal_output", {{ output: "a".repeat(cap) }}, key));
  const target = key ? state.subAgents.get(key) : state;
  const guarded = new Proxy(target, {{
    get(object, property) {{
      assert.notEqual(property, "terminal", "fold must not materialize terminal text");
      return Reflect.get(object, property);
    }},
  }});
  if (key) state.subAgents.set(key, guarded);
  let joins = 0;
  let joinedChars = 0;
  const originalJoin = Array.prototype.join;
  const count = maxParts * 12;
  try {{
    Array.prototype.join = function(separator) {{
      joins++;
      joinedChars += this.reduce((sum, part) => sum + (part ? part.length : 0), 0);
      return originalJoin.call(this, separator);
    }};
    for (let index = 0; index < count; index++) {{
      fold(key ? state : guarded, event("terminal_output", {{ output: "b" }}, key));
    }}
  }} finally {{
    Array.prototype.join = originalJoin;
  }}
  assert.ok(joins <= 12, `unexpected joins: ${{joins}}`);
  assert.ok(joinedChars <= (cap + maxPrefix) * 12);
  assert.ok(target._terminalParts.length <= maxParts);
  assert.equal(target._terminalChars, cap);
  assert.equal(target._terminalText, null);
  assert.equal(target.terminal, "a".repeat(cap - count) + "b".repeat(count));
  assert.equal(target.terminalTruncated, true);
  if (key) {{
    assert.equal(state.terminal, "");
    assert.equal(state.terminalTruncated, false);
    assert.deepEqual(target.steps, []);
  }}
}}
""",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(harness)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_user_images_have_independent_expected_states_at_every_prefix(tmp_path: Path) -> None:
    image = ArtifactRef(sha256="a" * 64, bytes=12, media_type="image/png", purpose="image", encoding="base64")
    second = image.model_copy(update={"sha256": "b" * 64})
    missing = image.model_copy(update={"sha256": "c" * 64})
    opaque = image.model_copy(update={"sha256": "d" * 64, "purpose": "provider_signature"})
    text_ref = image.model_copy(update={"media_type": "text/plain"})
    cases = [
        ("compare", [image, second], 2, False, True, 2),
        ("", [image, second], 2, False, True, 2),
        ("", [missing], 3, False, True, 3),
        ("", [], 2, False, True, 2),
        ("legacy text", [], None, False, True, 0),
        ("", [], None, False, False, 0),
        ("", [image], None, False, True, 0),
        ("", [opaque, text_ref], None, False, False, 0),
        ("delegated", [second], 2, True, True, 2),
        ("", [image], 1, True, True, 1),
        ("", [], None, True, False, 0),
        ("continued", [], None, False, True, 0),
        ("integral JSON number", [], 2.0, False, True, 2),
    ]
    events = []
    expected_main = []
    expected_sub = []
    assert replay([]).to_dict()["conversation"] == _run_js_fold([], tmp_path)["conversation"] == []
    for seq, (text, refs, count, delegated, visible, expected_count) in enumerate(cases, 1):
        event = _event(KnownEventType.TURN_STARTED, seq, turn_id=f"t{seq}", user_text=text)
        event.artifacts = refs
        if count is not None:
            event.content["image_count"] = count
        if delegated:
            _sub_agent(event, dispatch_id="images", agent_name="investigator")
        events.append(event)
        if visible:
            row = {"kind": "user", "text": text, "complete": True, "seq": seq, "elapsed_ms": 0}
            if refs:
                row["artifacts"] = [ref.model_dump(exclude_none=True) for ref in refs]
            if expected_count:
                row["image_count"] = expected_count
            if delegated:
                row["sub_agent"] = "images"
            (expected_sub if delegated else expected_main).append(row)
        python = replay(events).to_dict()
        javascript = _run_js_fold([item.model_dump(mode="json") for item in events], tmp_path)
        for state in (python, javascript):
            assert state["conversation"] == expected_main, f"main rows after {seq}"
            assert state["sub_agents"].get("images", {}).get("steps", []) == expected_sub, f"delegate rows after {seq}"
            assert [turn["turn_id"] for turn in state["turns"]] == [
                f"t{index}" for index in range(1, seq + 1) if not cases[index - 1][3]
            ]
        assert javascript == python


@pytest.mark.parametrize("count", [True, False, "2", -1, 0, 1.5, None, float("nan"), float("inf"), float("-inf")])
def test_malformed_image_counts_do_not_create_or_annotate_rows(count: object, tmp_path: Path) -> None:
    events = [
        _event(KnownEventType.TURN_STARTED, 1, turn_id="empty", image_count=count),
        _event(KnownEventType.TURN_STARTED, 2, turn_id="text", user_text="text", image_count=count),
    ]
    expected = [{"kind": "user", "text": "text", "complete": True, "seq": 2, "elapsed_ms": 0}]
    for cut in range(3):
        assert replay(events[:cut]).to_dict()["conversation"] == (expected if cut == 2 else [])
    harness = tmp_path / "malformed-count.mjs"
    harness.write_text(
        f"""
import assert from "node:assert/strict";
import {{ replay, toDict }} from {json.dumps(str(ASSET_DIR / "fold.js"))};
const events = {json.dumps([event.model_dump(mode="json") for event in events])};
for (let cut = 0; cut <= events.length; cut++) {{
  assert.deepEqual(toDict(replay(events.slice(0, cut))).conversation, cut === 2 ? {json.dumps(expected)} : []);
}}
""",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(harness)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_javascript_fold_rejects_out_of_order_events(tmp_path: Path) -> None:
    events = [
        _event(KnownEventType.ASSISTANT_DELTA, 5, uuid="a", text="x", complete=True).model_dump(mode="json"),
        _event(KnownEventType.ASSISTANT_DELTA, 4, uuid="b", text="y", complete=True).model_dump(mode="json"),
    ]
    harness = tmp_path / "harness.mjs"
    harness.write_text(
        "\n".join(
            [
                f"import {{ replay }} from {json.dumps(str(ASSET_DIR / 'fold.js'))};",
                "import { readFileSync } from 'node:fs';",
                "const events = JSON.parse(readFileSync(process.argv[2], 'utf8'));",
                "try { replay(events); process.stdout.write('no-error'); }",
                "catch (error) { process.stdout.write('raised'); }",
            ]
        ),
        encoding="utf-8",
    )
    payload = tmp_path / "events.json"
    payload.write_text(json.dumps(events), encoding="utf-8")
    result = subprocess.run(["node", str(harness), str(payload)], capture_output=True, text=True, timeout=60)
    assert result.stdout == "raised", "the JS fold must refuse a shuffled log like the Python one"


def test_tool_subject_parity_at_every_prefix(tmp_path: Path) -> None:
    events = [
        _event(KnownEventType.CHAT_MESSAGE, 1, message_type="tool_call", tool_call_id="a", tool_subject="a.py"),
        _event(KnownEventType.CHAT_MESSAGE, 2, message_type="tool_call", tool_call_id="b"),
        _sub_agent(
            _event(KnownEventType.CHAT_MESSAGE, 3, message_type="tool_call", tool_call_id="a", tool_subject="sub.py"),
            dispatch_id="d1",
        ),
        _event(KnownEventType.TOOL_STREAMING_UPDATE, 4, tool_call_id="a", text="one", stream_mode="append"),
        _event(
            KnownEventType.TOOL_STREAMING_UPDATE,
            5,
            tool_call_id="b",
            text="two",
            stream_mode="replace",
            tool_subject="b.py",
        ),
        _event(KnownEventType.CHAT_MESSAGE, 6, message_type="tool_call", tool_call_id="a", tool_subject="updated.py"),
        _sub_agent(
            _event(KnownEventType.TOOL_STREAMING_UPDATE, 7, tool_call_id="a", text="sub", tool_subject=""),
            dispatch_id="d1",
        ),
        _event(KnownEventType.TOOL_STREAMING_UPDATE, 8, tool_call_id="a", tool_subject=None, is_complete=True),
        _event(KnownEventType.CHAT_MESSAGE, 9, message_type="tool_result", tool_call_id="a"),
        _event(
            KnownEventType.CHAT_MESSAGE,
            10,
            message_type="tool_error",
            tool_call_id="b",
            tool_subject='<img src=x onerror="alert(1)">',
        ),
        _sub_agent(
            _event(
                KnownEventType.CHAT_MESSAGE, 11, message_type="tool_result", tool_call_id="a", tool_subject="done.py"
            ),
            dispatch_id="d1",
        ),
        _event(KnownEventType.CHAT_MESSAGE, 12, message_type="tool_result", tool_subject="standalone.py"),
        _event(KnownEventType.CHAT_MESSAGE, 13, message_type="tool_error", tool_subject="failed.py"),
        _event(KnownEventType.TOOL_STREAMING_UPDATE, 14, tool_call_id="stream", tool_subject="stream.py"),
        _event(KnownEventType.TOOL_STREAMING_UPDATE, 15, tool_call_id="stream", tool_subject={"path": "invalid"}),
        _event(KnownEventType.CHAT_MESSAGE, 16, message_type="tool_error", tool_call_id="stream"),
        _event(KnownEventType.CHAT_MESSAGE, 17, message_type="tool_call", tool_call_id="legacy"),
        _event(KnownEventType.CHAT_MESSAGE, 18, message_type="tool_result", tool_call_id="legacy"),
    ]
    for subject in (None, "", 42, False, [], {"path": "invalid"}):
        events.append(
            _event(KnownEventType.CHAT_MESSAGE, len(events) + 1, message_type="tool_call", tool_subject=subject)
        )
    serialized = [event.model_dump(mode="json") for event in events]
    for cut in range(1, len(events) + 1):
        assert _run_js_fold(serialized[:cut], tmp_path) == replay(events[:cut]).to_dict(), (
            f"tool subject folds diverged after {cut} events"
        )


def test_player_tool_subject_is_literal_and_metadata_changes_repaint(tmp_path: Path) -> None:
    """Exercise shipped renderer functions in Node; no browser or DOM package required."""
    harness = tmp_path / "player-subject.mjs"
    harness.write_text(
        f"""
import assert from "node:assert/strict";
import {{ readFileSync }} from "node:fs";
import {{ runInNewContext }} from "node:vm";
import {{ emptyState, fold }} from {json.dumps(str(ASSET_DIR / "fold.js"))};

// A minimal DOM surface for tool rows. Any HTML write is a test failure.
class Element {{
  constructor(tag) {{
    this.tagName = tag;
    this.children = [];
    this.dataset = {{}};
    this.attributes = {{}};
    this.textContent = "";
  }}
  set innerHTML(value) {{ throw new Error("HTML writes are forbidden"); }}
  insertAdjacentHTML() {{ throw new Error("HTML writes are forbidden"); }}
  append(...nodes) {{ this.children.push(...nodes); }}
  prepend(...nodes) {{ this.children.unshift(...nodes); }}
  setAttribute(name, value) {{ this.attributes[name] = value; }}
}}
const document = {{
  createElement: (tag) => new Element(tag),
  createDocumentFragment: () => new Element("#fragment"),
}};
const original = readFileSync({json.dumps(str(ASSET_DIR / "player.js"))}, "utf8");
assert.match(original, /^import .* from "\\.\\/fold\\.js";$/m);
assert.match(original, /\\nmain\\(\\);\\s*$/);
// Load the actual functions, but do not bootstrap network/UI event listeners.
const source = original.replace(/^import .* from "\\.\\/fold\\.js";$/m, "")
  .replace(/\\nmain\\(\\);\\s*$/, "");
runInNewContext(source + `
  const subjects = [
    '<img src=x onerror="globalThis.compromised=true">',
    '<a href="javascript:alert(1)">click</a>',
    'https://example.test/path',
  ];
  for (const field of ["toolSubject", "tool_subject"]) {{
    for (const subject of subjects) {{
      for (const agent of [null, {{ key: "d1", name: "delegate" }}]) {{
        const item = {{ kind: "tool", toolName: "read", status: "running", complete: true, text: "" }};
        const row = {{ item, agent }};
        const before = entrySnapshot(row);
        assert.equal(entryChanged(before, row), false);
        item[field] = subject;
        assert.equal(entryChanged(before, row), true, "metadata alone must repaint");
        const rendered = renderEntry(row);
        const head = rendered.children[1].children[0].children[0];
        assert.equal(head.children[0].textContent, "read");
        assert.equal(head.children[1].tagName, "span");
        assert.equal(head.children[1].className, "kc-tool-subject");
        assert.equal(head.children[1].textContent, subject);
        assert.equal(head.children[1].children.length, 0);
        assert.equal(head.children[2].textContent, "running");
        assert.equal(head.children[3].className, "kc-spinner");
        assert.equal(entryChanged(entrySnapshot(row), row), false);
        const snapshot = entrySnapshot(row);
        item[field] = "updated.py";
        assert.equal(entryChanged(snapshot, row), true);
        assert.equal(renderToolBody(item).children[0].children[1].textContent, "updated.py");
      }}
    }}
  }}
  for (const status of ["running", "done", "failed"]) {{
    const item = {{ kind: "tool", tool_name: "read", tool_subject: "app.py", status, text: "output" }};
    const head = renderToolBody(item).children[0];
    assert.equal(head.children[0].tagName, "button");
    assert.equal(head.children[0].attributes["aria-expanded"], "false");
    assert.equal(head.children[2].dataset.status, status);
    assert.equal(head.children.length, status === "running" ? 4 : 3);
  }}
  const legacyHead = renderToolBody({{ toolName: "read", status: "done" }}).children[0];
  assert.equal(legacyHead.children.length, 2);
  assert.equal(legacyHead.children[1].className, "kc-tool-status");
  for (const field of ["imageCount", "image_count"]) {{
    const item = {{ kind: "user", text: "", complete: true }};
    const row = {{ item, agent: null }};
    const before = entrySnapshot(row);
    item[field] = 2;
    assert.equal(entryChanged(before, row), true, "image count alone must repaint");
    const rendered = renderEntry(row);
    assert.equal(rendered.children[1].children[0].textContent, "2 images unavailable");
    assert.equal(entryChanged(entrySnapshot(row), row), false);
  }}
  manifest = {{ artifacts_inline: true }};
  const digest = "a".repeat(64);
  const ref = {{ sha256: digest, media_type: "image/png", purpose: "image" }};
  const item = {{
    kind: "user", text: "", complete: true,
    artifacts: [
      {{ ...ref, purpose: "provider_signature" }},
      {{ ...ref, purpose: "tool_result" }},
      {{ ...ref, media_type: "text/plain" }},
      {{ ...ref, sha256: "https://forbidden.invalid/image.png" }},
    ],
  }};
  const body = renderEntry({{ item }}).children[1];
  assert.equal(body.children.length, 1, "only purpose=image with image media is displayable");
  assert.equal(body.children[0].textContent, "Image unavailable");
  assert.equal(artifactSource({{ ...ref, path: "https://forbidden.invalid/image.png" }}), null);
  globalThis.__KC_REPLAY__ = {{ artifacts: {{ [digest]: "https://forbidden.invalid/image.png" }} }};
  assert.equal(artifactSource(ref), null, "inline map must not turn arbitrary URLs into requests");
  globalThis.__KC_REPLAY__.artifacts[digest] = "data:image/png;base64,aGVsbG8=";
  assert.equal(artifactSource(ref), "data:image/png;base64,aGVsbG8=");
  assert.equal(globalThis.compromised, undefined);
`, {{ document, emptyState, fold, assert }});
""",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(harness)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"player subject regression: {result.stderr}"
