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
    for cut in (1, 5, 11, 18, 22, len(events)):
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
