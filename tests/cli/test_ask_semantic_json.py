"""``ask --json`` streams the public semantic event protocol, one envelope per line."""

import json

from kolega_code.cli.session_store import SessionStore
from kolega_code.llm.ledger import HISTORY_ORIGIN, llm_call_origin
from kolega_code.llm.models import Message, TextBlock, ThinkingBlock, ToolCall, ToolResult
from kolega_code.llm.usage import normalize_usage

from ._app_test_utils import FakeCoderAgent

ENVELOPE_KEYS = {
    "schema",
    "version",
    "id",
    "session_id",
    "seq",
    "epoch_id",
    "turn_id",
    "timestamp",
    "actor",
    "agent_id",
    "agent_name",
    "parent_agent_id",
    "parent_tool_call_id",
    "depth",
    "type",
    "payload",
    "artifacts",
}


def _usage(inp=100, out=20):
    return normalize_usage({"input_tokens": inp, "output_tokens": out}, "anthropic", "claude-opus-5")


class SemanticAskFakeAgent(FakeCoderAgent):
    """Simulates a real turn — tool round-trip then a final answer — driving the
    real ``SessionRecorder`` and ``UsageLedger`` exactly as BaseAgent does, so
    the journal tee produces the full semantic stream."""

    agent_name = "coder"
    instances: list["SemanticAskFakeAgent"] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        SemanticAskFakeAgent.instances.append(self)

    async def fire_hook(self, event, payload):
        class Result:
            additional_context = None
            blocked = False
            end_turn = False

        return Result()

    def _settled_message(self, message):
        ledger = self.kwargs.get("usage_ledger")
        if ledger is not None:
            with llm_call_origin(HISTORY_ORIGIN):
                request_id = ledger.begin("anthropic", "claude-opus-5")
                ledger.record_response(request_id, message.usage, message)
        return message

    async def process_message_stream(self, message, attachments=None):
        self.messages.append(message)
        recorder = self.session_recorder
        assert recorder is not None  # every ask run records now
        user_message = Message(role="user", content=[TextBlock(text=message)])

        first = self._settled_message(
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="let me look", signature="opaque-sig"),
                    TextBlock(text="Checking the tests."),
                    ToolCall(id="t1", name="exec_command", input={"command": "pytest -q"}, execution_id="tool_exec_1"),
                ],
                stop_reason="tool_use",
                usage=_usage(100, 20),
            )
        )
        second = self._settled_message(
            Message(
                role="assistant",
                content=[TextBlock(text="All green.")],
                stop_reason="end_turn",
                usage=_usage(150, 10),
            )
        )

        recorder.start_turn(user_message)
        self.history.append(user_message)
        yield {"type": "response", "content": "Checking the tests.", "complete": False, "uuid": "m1"}
        recorder.record_assistant(first, reasoning_effort="high")
        self.history.append(first)
        yield {"type": "response", "content": "", "complete": True, "uuid": "m1"}

        results = recorder.record_tool_results(
            [
                ToolResult(
                    tool_use_id="t1",
                    content="1 passed",
                    name="exec_command",
                    is_error=False,
                    execution_id="tool_exec_1",
                )
            ]
        )
        self.history.append(Message(role="user", content=list(results)))

        recorder.record_assistant(second)
        self.history.append(second)
        recorder.finish_turn("completed")
        yield {"type": "response", "content": "All green.", "complete": True, "uuid": "m2"}


class SyntheticNoticeFakeAgent(SemanticAskFakeAgent):
    """The hook-block shape: a terminal informational chunk backed by a
    synthetic assistant notice, never an LLM message."""

    async def process_message_stream(self, message, attachments=None):
        self.messages.append(message)
        recorder = self.session_recorder
        assert recorder is not None
        recorder.record_synthetic_assistant("Prompt blocked by policy.", notice_code="hook_blocked")
        yield {"type": "response", "content": "Prompt blocked by policy.", "complete": True, "uuid": "x"}


def _setup(tmp_path, monkeypatch, agent_cls):
    from kolega_code.cli import main as main_module

    SemanticAskFakeAgent.instances = []
    monkeypatch.setattr(main_module, "CoderAgent", agent_cls)
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-opus-5")
    return main_module, project


def _lines(capsys):
    # Strict: every stdout line must be valid JSON.
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_json_streams_complete_semantic_envelopes(tmp_path, capsys, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch, SemanticAskFakeAgent)

    exit_code = main_module.main(["ask", "run the tests", "--project", str(project), "--json"])
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines, "expected semantic events on stdout"
    for line in lines:
        assert set(line) == ENVELOPE_KEYS
        assert line["schema"] == "kolega.session.event"
        assert line["version"] == 2
        assert "kind" not in line
    # Gapless sequence, in order.
    assert [line["seq"] for line in lines] == list(range(1, len(lines) + 1))

    types = [line["type"] for line in lines]
    assert types[0] == "session.created"
    assert types[1] == "context.epoch_started"
    assert types[-1] == "run.completed"
    assert types.count("turn.started") == 1
    assert types.count("assistant.message") == 2
    assert types.count("tool.results") == 1
    assert types.count("turn.completed") == 1

    first, second = [line["payload"] for line in lines if line["type"] == "assistant.message"]
    assert first["origin_type"] == "llm"
    assert first["llm_call_count"] == 1
    assert first["llm_call_id"]
    assert first["provider"] == "anthropic"
    assert first["model"] == "claude-opus-5"
    assert first["reasoning_effort"] == "high"
    message = first["message"]
    assert [block["type"] for block in message["content"]] == ["thinking", "text", "tool_call"]
    assert "signature" not in message["content"][0]
    assert "opaque-sig" not in json.dumps(lines)
    tool_call = message["content"][2]
    assert tool_call["tool_call_id"] == "tool_exec_1"
    assert tool_call["provider_call_id"] == "t1"
    assert tool_call["arguments"] == {"command": "pytest -q"}
    assert message["usage"]["total_tokens"] == 120
    assert "usage_metadata" not in message
    assert second["message"]["usage"]["total_tokens"] == 160
    assert second["llm_call_id"] and second["llm_call_id"] != first["llm_call_id"]

    results = [line for line in lines if line["type"] == "tool.results"]
    result_block = results[0]["payload"]["message"]["content"][0]
    assert result_block["tool_call_id"] == "tool_exec_1"

    run_completed = lines[-1]["payload"]
    assert run_completed["status"] == "completed"
    assert run_completed["exit_code"] == 0
    assert run_completed["totals"]["total_tokens"] == 280
    assert run_completed["kolega_version"]


def test_presentation_events_never_reach_json_stdout(tmp_path, capsys, monkeypatch, isolated_cli_env):
    from kolega_code.events import AgentEvent

    class DeltaEmittingFakeAgent(SemanticAskFakeAgent):
        async def process_message_stream(self, message, attachments=None):
            manager = self.kwargs["connection_manager"]
            await manager.broadcast_event(
                AgentEvent(event_type="assistant_delta", sender="coder", content={"text": "dup", "complete": False}),
                "ws",
                "thread",
            )
            await manager.broadcast_event(
                AgentEvent(event_type="chat_message", sender="coder", content={"message_type": "tool_call"}),
                "ws",
                "thread",
            )
            async for item in SemanticAskFakeAgent.process_message_stream(self, message, attachments):
                yield item

    main_module, project = _setup(tmp_path, monkeypatch, DeltaEmittingFakeAgent)
    exit_code = main_module.main(["ask", "go", "--project", str(project), "--json"])
    assert exit_code == 0

    lines = _lines(capsys)
    assert all(set(line) == ENVELOPE_KEYS for line in lines)
    assert not [line for line in lines if line["type"].startswith("ui.")]


def test_synthetic_notice_is_a_marked_assistant_event(tmp_path, capsys, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch, SyntheticNoticeFakeAgent)
    exit_code = main_module.main(["ask", "anything", "--project", str(project), "--json"])
    assert exit_code == 0

    lines = _lines(capsys)
    notices = [line["payload"] for line in lines if line["type"] == "assistant.message"]
    assert len(notices) == 1
    assert notices[0]["origin_type"] == "synthetic"
    assert notices[0]["llm_call_id"] is None
    assert notices[0]["llm_call_count"] == 0
    assert notices[0]["notice_code"] == "hook_blocked"
    assert notices[0]["message"]["content"] == [{"type": "text", "text": "Prompt blocked by policy."}]


def test_saved_run_live_stream_matches_events_export(tmp_path, capsys, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch, SemanticAskFakeAgent)

    exit_code = main_module.main(["ask", "run the tests", "--project", str(project), "--json", "--save"])
    assert exit_code == 0
    live = _lines(capsys)

    session_id = live[0]["session_id"]
    store = SessionStore()
    exported = [json.loads(line) for line in store.export_events(session_id).splitlines()]
    # Acceptance criterion 1: the live records ARE the exported records —
    # same ids, seqs, timestamps, and content.
    assert live == exported


def test_unsaved_run_creates_no_session_state(tmp_path, capsys, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch, SemanticAskFakeAgent)

    exit_code = main_module.main(["ask", "run the tests", "--project", str(project), "--json"])
    assert exit_code == 0
    lines = _lines(capsys)
    assert [line["type"] for line in lines][-1] == "run.completed"

    store = SessionStore()
    assert store.list() == []
    assert not (store.root / "sessions").exists() or not any((store.root / "sessions").iterdir())


def test_plain_mode_output_is_unchanged(tmp_path, capsys, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch, SemanticAskFakeAgent)
    exit_code = main_module.main(["ask", "run the tests", "--project", str(project)])
    assert exit_code == 0
    captured = capsys.readouterr()
    # Chunk-driven streaming to stdout, exactly as before: deltas concatenate
    # (no newline until a completing chunk carries content), empty flushes
    # print nothing.
    assert captured.out == "Checking the tests.All green.\n"


def test_loop_fresh_streams_every_iteration(tmp_path, capsys, monkeypatch, isolated_cli_env):
    from datetime import datetime, timedelta

    from kolega_code.cli import loop as loop_module
    from kolega_code.cli import main as main_module

    _setup(tmp_path, monkeypatch, SemanticAskFakeAgent)
    project = tmp_path / "project"

    clock = {"now": datetime(2026, 7, 27, 10, 0, 0)}
    monkeypatch.setattr(loop_module, "now_local", lambda: clock["now"])

    async def fake_sleep(seconds):
        clock["now"] += timedelta(seconds=seconds)

    monkeypatch.setattr(main_module.asyncio, "sleep", fake_sleep)

    exit_code = main_module.main(
        [
            "ask",
            "check",
            "--project",
            str(project),
            "--json",
            "--loop",
            "5m",
            "--loop-max-iterations",
            "2",
            "--loop-fresh",
        ]
    )
    assert exit_code == 0

    lines = _lines(capsys)
    types = [line["type"] for line in lines]
    # Two iterations x two assistant messages each, despite clear_history between.
    assert types.count("assistant.message") == 4
    assert types.count("loop.iteration_started") == 2
    assert types.count("loop.completed") == 1
    assert types[-1] == "run.completed"
    assert [line["seq"] for line in lines] == list(range(1, len(lines) + 1))
