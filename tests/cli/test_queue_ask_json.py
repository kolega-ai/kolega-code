from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kolega_code.agent.goal import GoalVerdict
from kolega_code.cli.session_store import SessionStore
from kolega_code.session.inbox import PeerMessage
from tests.cli.test_ask_goal import GoalAskFakeAgent
from tests.cli.test_ask_semantic_json import ENVELOPE_KEYS, SemanticAskFakeAgent, _setup


def test_real_headless_queue_preserves_semantic_only_gapless_ask_json(
    tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: Any
) -> None:
    class Agent(SemanticAskFakeAgent):
        apply_goal = GoalAskFakeAgent.apply_goal

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.prompt_extensions = list(kwargs.get("prompt_extensions", []))

        async def evaluate_goal_condition(self, condition: str) -> GoalVerdict:
            return GoalVerdict(met=True, reason="done")

    main, project = _setup(tmp_path, monkeypatch, Agent)
    start = main.start_headless_peer_inbox
    inboxes = []

    async def enqueue(**kwargs: Any) -> Any:
        inbox = await start(**kwargs)
        assert inbox is not None
        inbox._enqueue(PeerMessage.create(sender_session_id="peer", sender_title="teammate", text="follow-up"))
        inboxes.append(inbox)
        return inbox

    monkeypatch.setattr(main, "start_headless_peer_inbox", enqueue)
    assert main.main(["ask", "--goal", "finish offline", "--project", str(project), "--json", "--save"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert all(set(line) == ENVELOPE_KEYS for line in lines)
    assert [line["seq"] for line in lines] == list(range(1, len(lines) + 1))
    assert all(not line["type"].startswith("ui.") for line in lines)
    assert lines[-1]["type"] == "run.completed"
    assert len(inboxes) == 1 and not inboxes[0].pending_texts()
    agent = SemanticAskFakeAgent.instances[0]
    assert len(agent.messages) == 2 and agent.messages[1].endswith("follow-up")
    assert "information from another agent session" in agent.messages[1]
    events = SessionStore().journal(lines[0]["session_id"]).read_events()
    assert not any(event.event_type.startswith("ui.") for event in events)
    assert lines[-1]["payload"]["totals"]["total_tokens"] == 560
