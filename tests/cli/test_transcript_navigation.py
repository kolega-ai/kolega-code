from collections.abc import AsyncIterator, Callable
from pathlib import Path
import re

import pytest
import pytest_asyncio
from rich.console import RenderableType
from textual import events
from textual.geometry import Region
from textual.pilot import Pilot
from textual.screen import Screen

from kolega_code.cli.app import KolegaCodeApp

from ._app_test_utils import _build_sub_agent_test_app, wait_for_layout
from .test_app_rendering_window import _add_entries

pytestmark = pytest.mark.asyncio
PREFIX = 540
TOTAL = 1600


def _app(pilot: Pilot) -> KolegaCodeApp:
    assert isinstance(pilot.app, KolegaCodeApp)
    return pilot.app


def _labels(app: KolegaCodeApp) -> dict[int, int]:
    box = app._conversation.content_region
    strips = app.screen._compositor.render_strips()
    return {
        int(match[1]): y
        for y, strip in enumerate(strips[box.y : box.bottom])
        if (match := re.search(r"\bmessage (\d+)\b", strip.crop(box.x, box.right).text))
    }


async def _wait(pilot: Pilot, condition: Callable[[], bool], description: str) -> None:
    try:
        await wait_for_layout(pilot, condition)
    except AssertionError as exc:
        app = _app(pilot)
        raise AssertionError(f"{description}; thumb: {_thumb(app)}; painted IDs and rows: {_labels(app)}") from exc


async def _wheel(app: KolegaCodeApp, ticks: int) -> None:
    view = app._conversation
    event = events.MouseScrollDown if ticks > 0 else events.MouseScrollUp
    x, y = view.content_region.center
    for _ in range(abs(ticks)):
        # Pilot pauses between inputs. Dispatch the real events without a layout turn.
        await view._on_message(event(view, x - view.region.x, y - view.region.y, 0, 0, 0, False, False, False, x, y))


@pytest_asyncio.fixture
async def browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None) -> AsyncIterator[Pilot]:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 44)) as pilot:
        await wait_for_layout(pilot, app._startup_complete.is_set)
        _add_entries(app, PREFIX)
        app._flush_conversation_render()
        await _wait(pilot, lambda: PREFIX - 1 in _labels(app), "initial tail not visible")
        await pilot.wait_for_scheduled_animations()
        await _wheel(app, -8)
        await _wait(pilot, lambda: bool(ids := _labels(app)) and max(ids) < PREFIX - 1, "reader never left tail")
        _add_entries(app, TOTAL - PREFIX, start=PREFIX)
        app._flush_conversation_render()
        await pilot.wait_for_scheduled_animations()
        await _wait(
            pilot,
            lambda: bool(ids := _labels(app)) and PREFIX - 50 <= min(ids) <= max(ids) < PREFIX,
            "reader did not settle in the seeded browsing page",
        )
        yield pilot


def _thumb(app: KolegaCodeApp) -> tuple[Region, list[int]]:
    track = app._conversation.vertical_scrollbar.region
    rows = [
        y for y in range(track.y, track.bottom) if app.screen.get_style_at(track.x, y).meta.get("@mouse.down") == "grab"
    ]
    return track, rows


def _observe_paints(app: KolegaCodeApp, monkeypatch: pytest.MonkeyPatch, sample: Callable[[], None]) -> None:
    display = app._display

    def record(screen: Screen, renderable: RenderableType | None) -> None:
        if screen is app.screen and renderable is not None and not app._batch_count:
            sample()
        display(screen, renderable)

    monkeypatch.setattr(app, "_display", record)


async def test_scrollbar_thumb_tracks_visible_history(browser: Pilot) -> None:
    app = _app(browser)

    def thumb_matches_reader() -> bool:
        track, thumb = _thumb(app)
        labels = _labels(app)
        if not thumb or not labels:
            return False
        # The startup card precedes message 0; allow one terminal cell of quantization.
        center_id = (min(labels) + max(labels)) / 2
        expected = track.y + (center_id + 1) / (TOTAL + 1) * (track.height - 1)
        return abs((thumb[0] + thumb[-1]) / 2 - expected) <= 1

    await _wait(browser, thumb_matches_reader, "painted thumb does not represent the reader's place in history")


@pytest.mark.parametrize("destination", ["first", "last", "middle"])
async def test_single_thumb_drag_seeks_history(
    browser: Pilot, monkeypatch: pytest.MonkeyPatch, destination: str
) -> None:
    app = _app(browser)
    await wait_for_layout(browser, lambda: bool(_thumb(app)[1]))
    track, thumb = _thumb(app)
    bar = app._conversation.vertical_scrollbar
    target = {"first": 0, "last": track.height - 1, "middle": (track.height - 1) // 2}[destination]
    seen: set[int] = set()
    _observe_paints(app, monkeypatch, lambda: seen.update(_labels(app)))
    assert await browser.mouse_down(bar, offset=(0, thumb[len(thumb) // 2] - track.y))
    assert await browser._post_mouse_events([events.MouseMove], bar, (0, target), button=1)
    await browser.mouse_up(offset=(track.x, track.y + target))

    def arrived() -> bool:
        labels = _labels(app)
        if destination == "first":
            box = app._conversation.content_region
            hit, _ = app.screen.get_widget_at(box.x + 1, box.y)
            first_id = app.conversation_entries[0].entry_id
            return any(
                getattr(getattr(node, "entry", None), "entry_id", None) == first_id for node in hit.ancestors_with_self
            )
        if destination == "last":
            return TOTAL - 1 in labels
        # A cell on this track represents several dozen entries.
        return bool(labels) and abs((min(labels) + max(labels)) / 2 - TOTAL / 2) <= TOTAL / (track.height - 1)

    await _wait(browser, arrived, f"one thumb drag did not reach {destination}")
    distance = abs({"first": 0, "middle": TOTAL // 2, "last": TOTAL - 1}[destination] - PREFIX)
    assert 0 < len(seen) < distance // 2, f"drag painted {len(seen)} distinct messages"


def _ruler(app: KolegaCodeApp) -> tuple[int, Callable[[], int | None]]:
    labels = _labels(app)
    first = min(labels)
    pitch = labels[first + 1] - labels[first]
    assert pitch > 0

    def position() -> int | None:
        offsets = {index * pitch - y for index, y in _labels(app).items()}
        if not offsets:
            return None
        assert len(offsets) == 1, f"fixture rows must be uniformly spaced: {offsets}"
        return offsets.pop()

    return pitch, position


@pytest.mark.parametrize("direction", [-1, 1], ids=["up", "down"])
async def test_fast_wheel_burst_preserves_exact_travel(browser: Pilot, direction: int) -> None:
    app = _app(browser)
    pitch, position = _ruler(app)
    before = position()
    assert before is not None
    step = int(app.scroll_sensitivity_y)
    ticks = direction * (250 * pitch // step + 1)
    await _wheel(app, ticks)
    await _wait(browser, lambda: position() == before + ticks * step, f"expected line {before + ticks * step}")
    await browser.wait_for_scheduled_animations()
    assert position() == before + ticks * step


async def test_wheel_reversal_before_layout_does_not_chase_stale_motion(
    browser: Pilot, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(browser)
    pitch, position = _ruler(app)
    before = position()
    assert before is not None
    step = int(app.scroll_sensitivity_y)
    forward = 220 * pitch // step
    backward = 420 * pitch // step
    frames: list[int] = []

    def sample() -> None:
        if (p := position()) is not None:
            frames.append(p)

    await _wheel(app, forward)
    _observe_paints(app, monkeypatch, sample)
    await _wheel(app, -backward)
    # Old motion may already have been serviced or may have been cancelled.
    # Either way the reverse burst must carry us beyond the starting page.
    low = before - backward * step
    high = before + (forward - backward) * step
    await _wait(
        browser, lambda: (p := position()) is not None and low <= p <= high, f"expected line in [{low}, {high}]"
    )
    await browser.wait_for_scheduled_animations()
    assert frames and low <= frames[-1] <= high
    # After reversal, do not visibly page forward through the superseded burst.
    assert max(frames) <= before + app._conversation.content_region.height, frames


def _painted_handle(app: KolegaCodeApp) -> tuple[int, int]:
    track, grab = _thumb(app)
    assert grab and grab == list(range(grab[0], grab[-1] + 1))
    strips = app.screen._compositor.render_strips()
    colors = []
    for y in range(track.y, track.bottom):
        glyph = strips[y].crop(track.x, track.x + 1).text
        assert glyph in {" ", "█"}, f"partial scrollbar cell {glyph!r} at row {y}"
        style = app.screen.get_style_at(track.x, y)
        ink, paper = style.color or "default foreground", style.bgcolor or "default background"
        if style.reverse:
            ink, paper = paper, ink
        colors.append(ink if glyph == "█" else paper)
    background = colors[0 if grab[0] > track.y else -1]
    assert [track.y + i for i, color in enumerate(colors) if color != background] == grab
    return grab[0] - track.y, len(grab)


@pytest.mark.parametrize("height", [32, 64], ids=["one-row", "multirow"])
async def test_painted_thumb_stays_constant_across_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, height: int
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(100, height)) as pilot:
        await wait_for_layout(pilot, app._startup_complete.is_set)
        total, tall_lines = 360, 192
        tall_ids = range(80, 100)
        app._restore_conversation_history(
            [
                {"role": "user", "content": "\n".join([f"message {i}"] * (tall_lines if i in tall_ids else 1))}
                for i in range(total)
            ]
        )
        jump = app.query_one("#jump_to_bottom")
        await _wait(pilot, lambda: total - 1 in _labels(app) and not jump.region and bool(_thumb(app)[1]), "live tail")
        tail_pos, tail_size = _painted_handle(app)
        assert tail_size == 1 if height == 32 else tail_size > 1
        track_height = _thumb(app)[0].height
        pitch, _ = _ruler(app)
        step = int(app.scroll_sensitivity_y)
        short_ticks = 120 * pitch // step
        long_ticks = ((total - tall_ids.stop) * pitch + tall_lines) // step - short_ticks
        newest = app.conversation_entries[-1].entry_id
        frames: set[tuple[int, ...]] = set()
        _observe_paints(app, monkeypatch, lambda: frames.add(tuple(_thumb(app)[1])))

        async def wheel_to(ticks: int, ids: range, *, live: bool = False) -> int:
            await _wheel(app, ticks)

            def reached() -> bool:
                labels = _labels(app)
                track, grab = _thumb(app)
                return (
                    bool(labels and grab)
                    and max(labels) in ids
                    and (track.height == track_height if live else track.height < track_height)
                    and bool(jump.region) != live
                    and app._conversation._scrollback_anchor is None
                    and not app._conversation.wheel_offset
                )

            await _wait(pilot, reached, f"wheel {ticks} to {ids}")
            top, size = _painted_handle(app)
            assert size == tail_size, (top, size, _labels(app))
            return top

        short_ids = range(230, 250)
        short_up = await wheel_to(-short_ticks, short_ids)
        long_pos = await wheel_to(-long_ticks, tall_ids)
        assert set(_labels(app)).issubset(tall_ids)
        assert newest not in app._entry_widgets
        short_down = await wheel_to(long_ticks, short_ids)
        latest = await wheel_to(short_ticks + 2, range(total - 1, total), live=True)
        assert newest in app._entry_widgets
        assert tail_pos > short_up > long_pos
        assert long_pos < short_down < latest == tail_pos
        assert frames
        for rows in frames:
            assert rows and rows == tuple(range(rows[0], rows[0] + tail_size))


async def test_painted_thumb_remeasures_after_resize_and_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 36)) as pilot:
        await wait_for_layout(pilot, app._startup_complete.is_set)

        def replace(first: int, lines: int) -> None:
            app._restore_conversation_history(
                [{"role": "user", "content": "\n".join([f"message {i}"] * lines)} for i in range(first, first + 40)]
            )

        async def proportional_height(last: int, min_track: int = 0) -> int:
            await _wait(
                pilot,
                lambda: last in _labels(app) and _thumb(app)[0].height > min_track and bool(_thumb(app)[1]),
                f"fully mounted tail {last}",
            )
            entries = app.conversation_entries
            assert len(app._entry_widgets) == len(entries) == 41
            first_box = app._entry_widgets[entries[0].entry_id].region
            last_box = app._entry_widgets[entries[-1].entry_id].region
            track = _thumb(app)[0]
            expected = round(track.height * app._conversation.content_region.height / (last_box.bottom - first_box.y))
            assert 1 < expected < track.height
            assert _painted_handle(app)[1] == expected
            return expected

        replace(0, 1)
        small = await proportional_height(39)
        old_track = _thumb(app)[0].height
        await pilot.resize_terminal(100, 58)
        large = await proportional_height(39, old_track)
        replace(100, 6)
        replaced = await proportional_height(139)
        assert small < large and replaced < large

        view = app._conversation
        view.styles.height = 30
        await _wait(pilot, lambda: _thumb(app)[0].height == 30, "pane resized without a terminal resize")
        pane = await proportional_height(139)
        assert pane < replaced


async def test_thumb_remeasures_for_reflow_and_new_offscreen_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 58)) as pilot:
        await wait_for_layout(pilot, app._startup_complete.is_set)
        _add_entries(app, 48)
        app._flush_conversation_render()
        await _wait(pilot, lambda: 47 in _labels(app) and bool(_thumb(app)[1]), "short messages at tail")
        before = _painted_handle(app)[1]
        entry = app.conversation_entries[-1]
        entry.content = "message 47\n" * 120
        app._invalidate_conversation(entry)
        app._flush_conversation_render()
        await _wait(pilot, lambda: set(_labels(app)) == {47} and 0 < len(_thumb(app)[1]) < before, "tall message")

        await _wheel(app, -100)
        await _wait(pilot, lambda: 0 in _labels(app) and bool(app.query_one("#jump_to_bottom").region), "first page")
        labels = _labels(app)
        mounted = tuple(app._entry_widgets)
        original = _painted_handle(app)[1]
        _add_entries(app, 200, start=48)
        app._flush_conversation_render()
        await _wait(pilot, lambda: 0 < len(_thumb(app)[1]) < original, "history grew beyond mounted page")
        assert tuple(app._entry_widgets) == mounted
        assert _labels(app) == labels
