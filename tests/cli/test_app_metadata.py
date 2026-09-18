"""The metadata strip follows real app state without synchronous Git lookups."""

import threading
import time
from unittest.mock import Mock

import pytest
from rich.color import Color as RichColor
from rich.console import Console
from rich.style import Style
from rich.text import Text

from kolega_code.cli.theme import Color
from kolega_code.cli.tui.metadata import MetadataStrip
from kolega_code.cli.tui.session_diff import DiffScope
from kolega_code.permissions import PermissionMode

from ._app_test_utils import _build_mention_test_app


def _style_color(text: Text, label: str) -> RichColor | None:
    return text.get_style_at_offset(Console(), text.plain.index(label)).color


@pytest.mark.asyncio
async def test_scope_probe_refreshes_metadata_off_thread_and_rejects_stale_results(tmp_path, monkeypatch) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 40)):
        main_thread = threading.get_ident()
        calls = []
        tracker = Mock()

        def scope(*, checkpoint_id=None):
            calls.append(threading.get_ident())
            return DiffScope(branch="feat/readable", root_path=str(app.active_project_path))

        tracker.scope.side_effect = scope
        app._session_diff_tracker = tracker
        generation = app._session_diff_generation
        await app._scope_probe_worker(tracker, generation)
        strip = app.query_one(MetadataStrip)
        assert strip.branch == "feat/readable"
        assert calls and all(thread != main_thread for thread in calls)
        tracker.scope.return_value = None
        tracker.scope.side_effect = lambda **kwargs: DiffScope(branch="stale")
        await app._scope_probe_worker(tracker, generation - 1)
        assert strip.branch == "feat/readable"
        tracker.scope.reset_mock()
        app._meta_content()
        strip.render()
        tracker.scope.assert_not_called()
        # Restore teardown's tracker contract.
        app._session_diff_tracker = None


@pytest.mark.asyncio
async def test_metadata_keeps_risk_visible_and_updates_in_place(tmp_path, monkeypatch) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(60, 35)) as pilot:
        app._set_sidebar_visible(False)
        app._session_diff_scope = DiffScope(branch="feat/a-long-branch-name")
        app.permission_mode = PermissionMode.AUTO
        app._update_mode_chrome()
        strip = app.query_one(MetadataStrip)
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            await pilot.pause(0.02)
            if strip.content_size.width == 60:
                break
        text = strip.render()
        assert text.cell_len <= 60 and "build" in text.plain and "auto" in text.plain
        assert strip.region.height == 1
        assert "feat/a-long-branch-name" in strip.full_context
        app.interaction_mode = "plan"
        app.permission_mode = PermissionMode.ASK
        app._update_mode_chrome()
        assert app.query_one(MetadataStrip) is strip
        assert "plan" in strip.render().plain and "ask" in strip.render().plain


@pytest.mark.asyncio
async def test_interaction_mode_toggle_updates_metadata_mode_color(tmp_path, monkeypatch) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(80, 35)) as pilot:
        app._set_sidebar_visible(False)
        strip = app.query_one(MetadataStrip)
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            await pilot.pause(0.02)
            if strip.content_size.width == 80 and "build" in strip.render().plain:
                break
        assert strip.render().cell_len <= 80

        build = strip.render()
        assert "build" in build.plain
        build_color = _style_color(build, "build")
        assert build_color == Style.parse(Color.ACCENT).color

        await app.action_toggle_interaction_mode()
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            await pilot.pause(0.02)
            if strip.interaction_mode == "plan" and "plan" in strip.render().plain:
                break
        plan = strip.render()
        assert "plan" in plan.plain
        plan_color = _style_color(plan, "plan")
        assert plan_color == Style.parse(Color.SUCCESS).color
        assert plan_color != build_color

        await app.action_toggle_interaction_mode()
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            await pilot.pause(0.02)
            if strip.interaction_mode == "build" and "build" in strip.render().plain:
                break
        rebuilt = strip.render()
        assert "build" in rebuilt.plain
        assert _style_color(rebuilt, "build") == build_color
        assert app.query_one(MetadataStrip) is strip
