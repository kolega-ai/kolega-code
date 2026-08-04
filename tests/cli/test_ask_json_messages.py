"""ask --json emits complete structured messages, never streaming chunks."""

import json

from kolega_code.llm.models import Message, TextBlock, ThinkingBlock, ToolCall
from kolega_code.llm.usage import normalize_usage

from ._app_test_utils import FakeCoderAgent


def _usage(inp=100, out=20):
    return normalize_usage({"input_tokens": inp, "output_tokens": out}, "anthropic", "claude-opus-5")


class MessageAskFakeAgent(FakeCoderAgent):
    """Simulates a real turn: a tool round-trip then a final answer, appending
    complete Messages to history before yielding each message's final chunk —
    the same ordering BaseAgent guarantees."""

    agent_name = "coder"
    instances: list["MessageAskFakeAgent"] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        MessageAskFakeAgent.instances.append(self)

    async def process_message_stream(self, message, attachments=None):
        self.messages.append(message)
        self.history.append(Message(role="user", content=[TextBlock(text=message)]))

        # Message 1: thinking + text + a tool call (ends in tool_use).
        first = Message(
            role="assistant",
            content=[
                ThinkingBlock(thinking="let me look", signature="opaque-sig"),
                TextBlock(text="Checking the tests."),
                ToolCall(id="t1", name="exec_command", input={"command": "pytest -q"}),
            ],
            stop_reason="tool_use",
            usage=_usage(100, 20),
        )
        yield {"type": "response", "content": "Checking the tests.", "complete": False, "uuid": "m1"}
        self.history.append(first)
        # The empty flush BaseAgent emits after a tool round.
        yield {"type": "response", "content": "", "complete": True, "uuid": "m1"}

        # Tool result (user role — must not be emitted).
        self.history.append(Message(role="user", content=[TextBlock(text="[tool result]")]))

        # Message 2: the final answer.
        second = Message(
            role="assistant",
            content=[TextBlock(text="All green.")],
            stop_reason="end_turn",
            usage=_usage(150, 10),
        )
        self.history.append(second)
        yield {"type": "response", "content": "All green.", "complete": True, "uuid": "m2"}

    async def fire_hook(self, event, payload):
        class Result:
            additional_context = None
            blocked = False
            end_turn = False

        return Result()


class NoHistoryFakeAgent(MessageAskFakeAgent):
    """Yields a terminal informational chunk without touching history —
    the hook-block / agent-command / notice shape."""

    async def process_message_stream(self, message, attachments=None):
        self.messages.append(message)
        yield {"type": "response", "content": "Prompt blocked by policy.", "complete": True, "uuid": "x"}


def _setup(tmp_path, monkeypatch, agent_cls):
    from kolega_code.cli import main as main_module

    MessageAskFakeAgent.instances = []
    monkeypatch.setattr(main_module, "CoderAgent", agent_cls)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-opus-5")
    return main_module, project


def _lines(capsys):
    # Strict: every stdout line must be valid JSON.
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_json_emits_one_line_per_completed_message(tmp_path, capsys, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch, MessageAskFakeAgent)

    exit_code = main_module.main(["ask", "run the tests", "--project", str(project), "--json"])
    assert exit_code == 0

    lines = _lines(capsys)
    kinds = [line["kind"] for line in lines]
    assert "chunk" not in kinds

    messages = [line["data"] for line in lines if line["kind"] == "message"]
    assert len(messages) == 2
    assert all(m["role"] == "assistant" for m in messages)

    first, second = messages
    assert first["stop_reason"] == "tool_use"
    block_types = [block["type"] for block in first["content"]]
    assert block_types == ["thinking", "text", "tool_call"]
    tool_call = first["content"][2]
    assert tool_call["name"] == "exec_command"
    assert tool_call["input"] == {"command": "pytest -q"}
    assert first["usage"]["total_tokens"] == 120
    # Opaque provider state stripped; raw provider metadata not exposed.
    assert "signature" not in first["content"][0]
    assert "usage_metadata" not in first

    assert second["stop_reason"] == "end_turn"
    assert second["content"] == [{"type": "text", "text": "All green."}]
    assert second["usage"]["total_tokens"] == 160

    summary = lines[-1]
    assert summary["kind"] == "summary"
    assert summary["messages"] == 2
    assert "chunks" not in summary


def test_json_suppresses_streaming_delta_events(tmp_path, capsys, monkeypatch, isolated_cli_env):
    from kolega_code.events import AgentEvent

    class DeltaEmittingFakeAgent(MessageAskFakeAgent):
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
            async for item in MessageAskFakeAgent.process_message_stream(self, message, attachments):
                yield item

    main_module, project = _setup(tmp_path, monkeypatch, DeltaEmittingFakeAgent)
    exit_code = main_module.main(["ask", "go", "--project", str(project), "--json"])
    assert exit_code == 0

    lines = _lines(capsys)
    event_types = [line["data"]["event_type"] for line in lines if line["kind"] == "event"]
    assert "assistant_delta" not in event_types
    assert "chat_message" in event_types


def test_terminal_chunk_without_history_becomes_synthetic_message(tmp_path, capsys, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch, NoHistoryFakeAgent)
    exit_code = main_module.main(["ask", "anything", "--project", str(project), "--json"])
    assert exit_code == 0

    lines = _lines(capsys)
    messages = [line["data"] for line in lines if line["kind"] == "message"]
    assert messages == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Prompt blocked by policy."}],
            "stop_reason": "end_turn",
        }
    ]


def test_plain_mode_output_is_unchanged(tmp_path, capsys, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch, MessageAskFakeAgent)
    exit_code = main_module.main(["ask", "run the tests", "--project", str(project)])
    assert exit_code == 0
    captured = capsys.readouterr()
    # Chunk-driven streaming to stdout, exactly as before: deltas concatenate
    # (no newline until a completing chunk carries content), empty flushes
    # print nothing.
    assert captured.out == "Checking the tests.All green.\n"


def test_loop_fresh_emitter_survives_history_clear(tmp_path, capsys, monkeypatch, isolated_cli_env):
    from datetime import datetime, timedelta

    from kolega_code.cli import loop as loop_module
    from kolega_code.cli import main as main_module

    _setup(tmp_path, monkeypatch, MessageAskFakeAgent)
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
    messages = [line for line in lines if line["kind"] == "message"]
    # Two iterations x two assistant messages each, despite clear_history between.
    assert len(messages) == 4
    assert lines[-1]["messages"] == 4


def test_serialize_keeps_plain_reasoning_and_strips_opaque_fields():
    # DeepSeek flash retains its raw chain-of-thought as plain text on the
    # reasoning block; --json consumers get it (parity with Anthropic thinking),
    # minus the provider-internal fields.
    from kolega_code.cli.ask_output import serialize_ask_message
    from kolega_code.llm.models import ResponsesReasoningBlock

    message = Message(
        role="assistant",
        content=[
            ResponsesReasoningBlock(content=["consider the recurrence"], item_id="rs_1"),
            TextBlock(text="the answer"),
        ],
        stop_reason="end_turn",
    )
    data = serialize_ask_message(message)
    assert data["content"][0] == {"type": "responses_reasoning", "summary": [], "content": ["consider the recurrence"]}
    assert data["content"][1]["type"] == "text"


def test_serialize_keeps_hosted_web_search_calls():
    # Hosted web_search_call blocks carry only readable action metadata
    # (queries/URL), so --json consumers (e.g. the benchmark jsonl census) get
    # the whole block — nothing opaque to strip.
    from kolega_code.cli.ask_output import serialize_ask_message
    from kolega_code.llm.models import WebSearchCallBlock

    message = Message(
        role="assistant",
        content=[
            WebSearchCallBlock(item_id="ws_1", status="completed", action={"type": "search", "queries": ["q"]}),
            TextBlock(text="the answer"),
        ],
        stop_reason="end_turn",
    )
    data = serialize_ask_message(message)
    assert data["content"][0] == {
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "queries": ["q"]},
        "item_id": "ws_1",
    }
    assert data["content"][1]["type"] == "text"


def test_serialize_drops_encrypted_only_reasoning():
    # OpenAI/ChatGPT reasoning is an opaque blob; once stripped nothing readable
    # remains, so the whole block is dropped as before.
    from kolega_code.cli.ask_output import serialize_ask_message
    from kolega_code.llm.models import ResponsesReasoningBlock

    message = Message(
        role="assistant",
        content=[
            ResponsesReasoningBlock(encrypted_content="ENC", summary=["planning"]),
            TextBlock(text="the answer"),
        ],
        stop_reason="end_turn",
    )
    data = serialize_ask_message(message)
    assert [block["type"] for block in data["content"]] == ["text"]
