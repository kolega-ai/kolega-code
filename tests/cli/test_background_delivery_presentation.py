from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from kolega_code.cli.session_store import SessionStore
from kolega_code.events import AgentEvent, AgentEventEmitter
from kolega_code.llm.models import Message
from kolega_code.session.background import BackgroundAgentManager
from tests.cli.test_app_handoff import _build_handoff_test_app
from tests.cli.test_ask_semantic_json import ENVELOPE_KEYS, SemanticAskFakeAgent, _setup


class _BackgroundAskAgent(SemanticAskFakeAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.background_jobs: BackgroundAgentManager | None = None
        self.background_needs_turn = False
        self.emitter = AgentEventEmitter(
            kwargs["connection_manager"], kwargs["workspace_id"], kwargs["thread_id"], "ask-root"
        )

    def get_background_jobs(self) -> BackgroundAgentManager:
        if self.background_jobs is None:
            self.background_jobs = BackgroundAgentManager(self.session_recorder)
        return self.background_jobs

    async def process_message_stream(self, message: str, attachments: Any = None) -> Any:
        async for item in super().process_message_stream(message, attachments):
            yield item

        async def child() -> str:
            return "background report delivered once"

        jobs = self.get_background_jobs()
        await jobs.submit("ask-child", child, agent_type="investigation", task="inspect offline")
        await jobs.wait()

    async def continue_from_history_stream(self) -> Any:
        self.background_needs_turn = False
        recorder = self.session_recorder
        assert recorder is not None
        recorder.start_continuation_turn()
        recorder.record_synthetic_assistant("Child report received.", notice_code="test")
        self.history.append(Message(role="assistant", content="Child report received."))
        recorder.finish_turn("completed")
        yield {"type": "response", "content": "Child report received.", "complete": True}

    async def cleanup(self) -> None:
        if self.background_jobs is not None:
            await self.background_jobs.close()
        await super().cleanup()


def test_real_background_delivery_preserves_ask_semantic_only_gapless_json(
    tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: Any
) -> None:
    observed: list[dict[str, Any]] = []
    emit = AgentEventEmitter.background_job_delivered

    async def observe(emitter: AgentEventEmitter, completion: dict[str, Any]) -> None:
        observed.append(completion.copy())
        await emit(emitter, completion)

    monkeypatch.setattr(AgentEventEmitter, "background_job_delivered", observe)
    main, project = _setup(tmp_path, monkeypatch, _BackgroundAskAgent)
    assert main.main(["ask", "delegate offline", "--project", str(project), "--json", "--save"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert all(set(line) == ENVELOPE_KEYS for line in lines)
    assert [line["seq"] for line in lines] == list(range(1, len(lines) + 1))
    assert not any(line["type"].startswith("ui.") for line in lines)
    assert len(observed) == 1 and observed[0]["job_id"] == "ask-child"
    completions = [line for line in lines if line["type"] == "context.message" and line["payload"].get("completion_id")]
    assert len(completions) == 1
    assert completions[0]["payload"]["completion_id"] == observed[0]["completion_id"]
    assert completions[0]["payload"]["status"] == "completed"
    assert lines[-1]["type"] == "run.completed"
    assert lines[-1]["payload"]["totals"]["total_tokens"] == 280
    store = SessionStore()
    record = store.load(lines[0]["session_id"])
    events = store.journal(record.session_id).read_events()
    assert not any(event.event_type.startswith("ui.") for event in events)
    assert (
        sum("background report delivered once" in Message.from_dict(row).get_text_content() for row in record.history)
        == 1
    )


@pytest.mark.asyncio
async def test_delivery_event_is_recording_only_in_tui_with_frontend_callback_still_the_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: Any
) -> None:
    app = _build_handoff_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        app.show_logs = True
        entries = list(app.conversation_entries)
        write_log = Mock()
        monkeypatch.setattr(app, "_write_log", write_log)
        app._render_event(
            AgentEvent(
                event_type="background_job_delivered",
                sender="root",
                content={"text": "must not render twice"},
            )
        )
        write_log.assert_not_called()
        assert app.conversation_entries == entries
        await app.action_quit()
