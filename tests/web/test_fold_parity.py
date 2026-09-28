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
from pathlib import Path

import pytest

from kolega_code.events import AgentEvent, ArtifactRef, KnownEventType
from kolega_code.session.projection import replay

ASSET_DIR = Path(__file__).resolve().parents[2] / "kolega_code" / "web" / "assets"

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


def test_folds_agree_at_every_prefix(tmp_path: Path) -> None:
    """Seeking must agree too, not just the final state."""
    events = _fixture_events()
    serialized = [event.model_dump(mode="json") for event in events]
    for cut in (1, 5, 11, 18, 22, 30, 38, 41, len(events)):
        expected = replay(events[:cut]).to_dict()
        actual = _run_js_fold(serialized[:cut], tmp_path)
        assert actual == expected, f"folds diverged after {cut} events"


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
  assert.equal(globalThis.compromised, undefined);
`, {{ document, emptyState, fold, assert }});
""",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(harness)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"player subject regression: {result.stderr}"
