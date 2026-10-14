"""Late renders must not touch a transcript whose children are being pruned."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from textual.widgets import Collapsible
from textual.widgets._collapsible import CollapsibleTitle

from kolega_code.cli.tui.startup import StartupDisclosure, StartupEntryWidget
from kolega_code.cli.tui.state import ConversationEntry
from kolega_code.cli.tui.widgets import ToolEntryWidget

from ._app_test_utils import _build_sub_agent_test_app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["flush-dirty", "flush-append", "render", "sync", "dirty", "all", "rebuild", "expand", "compensate"]
)
async def test_late_render_skips_attached_view_with_pruned_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    checked = False
    errors: list[Exception] = []

    async with app.run_test() as pilot:
        await pilot.pause()
        card = app.query_one(StartupEntryWidget)
        outer = card.query_one(Collapsible)
        view = app._conversation
        window = app._transcript_window
        assert window is not None
        assert card.content_size.width > 0
        mounted = dict(window.widgets)

        async def render_during_unmount(disclosure: StartupDisclosure) -> None:
            nonlocal checked
            if disclosure is not outer:
                return
            checked = True
            try:
                # Real Textual teardown: children exit before their parent, but
                # the conversation is still attached. No sleeps or fake flags.
                assert view.is_attached
                assert view._pruning
                assert not disclosure.query(CollapsibleTitle)

                def unexpected_refresh() -> None:
                    raise AssertionError("a pruning entry must not be refreshed")

                def unexpected_mount(entry: ConversationEntry) -> ToolEntryWidget:
                    raise AssertionError("a pruning view must not mount new entries")

                monkeypatch.setattr(card, "refresh_content", unexpected_refresh)
                monkeypatch.setattr(window, "_make_widget", unexpected_mount)
                dirty = {card.entry.entry_id}
                if operation.startswith("flush"):
                    if operation == "flush-append":
                        app.conversation_entries.append(ConversationEntry(kind="assistant", content="late"))
                    app._dirty_entry_ids.update(dirty)
                    app._render_pending = True
                    app._flush_conversation_render()
                    assert not app._render_pending
                    assert not app._dirty_entry_ids
                elif operation == "render":
                    app._render_conversation()
                elif operation == "sync":
                    window.sync(app.conversation_entries, dirty, follow_bottom=True)
                    assert not dirty
                elif operation == "dirty":
                    window.refresh_dirty(dirty)
                    assert not dirty
                elif operation == "all":
                    window.refresh_all()
                elif operation == "expand":
                    window.mounted_start = 1
                    entries = [ConversationEntry(kind="user", content="older"), *app.conversation_entries]
                    assert not window.expand_up(entries)
                    assert window.mounted_start == 1
                    assert not window._expand_pending
                elif operation == "compensate":
                    callbacks: list[Callable[[], None]] = []
                    scroll_calls: list[dict[str, object]] = []
                    monkeypatch.setattr(view, "call_after_refresh", callbacks.append)
                    monkeypatch.setattr(view, "scroll_to", lambda **kwargs: scroll_calls.append(kwargs))
                    window._expand_pending = True
                    window._schedule_expand_compensation(-1)
                    assert len(callbacks) == 1
                    callbacks[0]()
                    assert not window._expand_pending
                    assert not scroll_calls
                else:
                    window.rebuild(app.conversation_entries)
                assert window.widgets == mounted
            except Exception as error:
                # Let Textual finish shutdown even when testing the broken code.
                errors.append(error)

        monkeypatch.setattr(StartupDisclosure, "on_unmount", render_during_unmount, raising=False)

    assert checked, "the regression must run inside real DOM teardown"
    assert not errors, repr(errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["startup", "tool_result"])
async def test_entry_refresh_tolerates_pruned_disclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        if kind == "tool_result":
            app._add_conversation_entry(
                ConversationEntry(kind=kind, content="done", tool_name="read", tool_subject="file.py")
            )
            app._flush_conversation_render()
        await pilot.pause()
        card = next(widget for widget in app.query(ToolEntryWidget) if widget.entry.kind == kind)
        assert card.content_size.width > 0
        disclosure = card.query_one(Collapsible)
        await disclosure.remove()
        assert card.is_attached
        assert disclosure._pruning
        assert not disclosure.query(CollapsibleTitle)

        # Resize events and direct refreshes can outlive only the child subtree.
        card.refresh_content()
        card._refresh_title()
