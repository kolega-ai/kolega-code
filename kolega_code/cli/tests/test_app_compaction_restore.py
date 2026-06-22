"""Tests for compaction summary placement when restoring the conversation transcript.

When the transcript is rebuilt from saved history (session resume, model switch,
agent rebuild), the compaction summary must sit where it appeared in the live
transcript: after the retained tail messages, before any newer turns. The history
length captured at compaction time (``compacted_history_length``) anchors that
position; old sessions without it fall back gracefully.
"""

from pathlib import Path

import pytest

from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.session_store import SessionStore


def _text_message(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _build_history(n: int) -> list[dict]:
    return [_text_message("user" if i % 2 == 0 else "assistant", f"msg-{i}") for i in range(n)]


@pytest.mark.asyncio
async def test_restore_places_summary_after_retained_tail(tmp_path: Path) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    app_module.CoderAgent = FakeCoderAgent  # noqa: SLF001
    try:
        project = tmp_path / "project"
        project.mkdir()
        config = build_test_config(project)
        store = SessionStore(tmp_path / "state")
        session = store.create(project, "code", config_summary(config))

        # 10 messages at compaction time; boundary at 4 (folded prefix).
        # The retained tail was messages 4..9 (6 messages). After compaction,
        # 2 more turns were added, so history is now 12 long.
        history = _build_history(12)
        session.history = history
        session.compaction = {
            "summary": "THE SUMMARY",
            "compacted_through": 4,
            "compacted_history_length": 10,
        }
        store.save(session)

        app = KolegaCodeApp(
            project_path=project,
            config=config,
            mode="code",
            store=store,
            session=session,
        )

        async with app.run_test():
            app._restore_conversation_history(history)

            kinds = [entry.kind for entry in app.conversation_entries]
            # Startup entry is always first.
            assert kinds[0] == "startup"
            # The compaction_summary sits at index 1 + 10 = 11 (startup + 10 history entries).
            summary_index = kinds.index("compaction_summary")
            assert summary_index == 11
            # Everything before it (besides startup) is history[:10].
            before = app.conversation_entries[1:summary_index]
            assert len(before) == 10
            assert all(entry.kind in {"user", "assistant"} for entry in before)
            # The retained tail (msg-4..msg-9) is immediately before the summary.
            assert [entry.content for entry in before[-2:]] == ["msg-8", "msg-9"]
            # Newer turns (msg-10, msg-11) come after the summary.
            after = app.conversation_entries[summary_index + 1 :]
            assert [entry.content for entry in after] == ["msg-10", "msg-11"]
    finally:
        del app_module.CoderAgent


@pytest.mark.asyncio
async def test_restore_old_session_without_history_length_falls_back(tmp_path: Path) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    app_module.CoderAgent = FakeCoderAgent  # noqa: SLF001
    try:
        project = tmp_path / "project"
        project.mkdir()
        config = build_test_config(project)
        store = SessionStore(tmp_path / "state")
        session = store.create(project, "code", config_summary(config))

        # Old session: no compacted_history_length key.
        history = _build_history(12)
        session.history = history
        session.compaction = {"summary": "OLD SUMMARY", "compacted_through": 4}
        store.save(session)

        app = KolegaCodeApp(
            project_path=project,
            config=config,
            mode="code",
            store=store,
            session=session,
        )

        async with app.run_test():
            # Must not raise; summary still renders (at the fallback boundary position).
            app._restore_conversation_history(history)
            kinds = [entry.kind for entry in app.conversation_entries]
            assert "compaction_summary" in kinds
    finally:
        del app_module.CoderAgent


def build_test_config(project: Path):
    return build_agent_config(
        project,
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
        },
    )
