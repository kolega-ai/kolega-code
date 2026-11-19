import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.scrollbar import ScrollBar
from textual.widgets import Static

from kolega_code.cli.tui.transcript_scrollbar import TranscriptScrollBar

from ._app_test_utils import wait_for_layout


class _View(VerticalScroll):
    @property
    def vertical_scrollbar(self) -> ScrollBar:
        if self._vertical_scrollbar is None:
            bar = TranscriptScrollBar(name="vertical", thickness=self.scrollbar_size_vertical)
            self._vertical_scrollbar = bar
            bar.display = False
            self.app._start_widget(self, bar)
        return self._vertical_scrollbar

    def compose(self) -> ComposeResult:
        yield Static("local content\n" * 100)


class _Transcript(_View):
    metrics: tuple[float, float, float] = (0.375, 1, 0.25)
    thumb_rows: int = 5

    def __init__(self) -> None:
        super().__init__()
        self.seeks: list[float] = []

    def transcript_scroll_metrics(self) -> tuple[float, float, float]:
        return self.metrics

    def transcript_thumb_rows(self, track_rows: int) -> int:
        assert track_rows == self.vertical_scrollbar.size.height
        return self.thumb_rows

    def seek_transcript(self, fraction: float) -> None:
        self.seeks.append(fraction)
        _, total, visible = self.metrics
        self.metrics = (fraction * (total - visible), total, visible)
        self.vertical_scrollbar.refresh()


class _App(App[None]):
    CSS = "VerticalScroll { height: 20; margin-top: 2; }"

    def __init__(self, view: _View) -> None:
        super().__init__()
        self.view = view

    def compose(self) -> ComposeResult:
        yield self.view


def _actions(bar: ScrollBar) -> list[str | None]:
    return [bar.screen.get_style_at(bar.region.x, y).meta.get("@mouse.down") for y in bar.region.line_range]


@pytest.mark.asyncio
@pytest.mark.parametrize("thumb", [1, 4])
async def test_painted_rows_match_hit_area(thumb: int) -> None:
    view = _Transcript()
    view.thumb_rows = thumb
    async with _App(view).run_test() as pilot:
        bar = view.vertical_scrollbar
        for fraction in (0, 0.17, 0.5, 0.85, 1, 0.25):
            view.metrics = (fraction * 8, 10, 2)
            bar.refresh()
            start = round(fraction * (20 - thumb))
            expected = ["scroll_up"] * start + ["grab"] * thumb + ["scroll_down"] * (20 - start - thumb)
            await wait_for_layout(pilot, lambda: _actions(bar) == expected)
            for row, action in enumerate(expected):
                assert bar.render_line(row).text == " " * bar.size.width
                for x in bar.region.column_range:
                    style = bar.screen.get_style_at(x, bar.region.y + row)
                    assert style.meta.get("@mouse.down") == action
                    assert bool(style.reverse) == (action == "grab")


@pytest.mark.asyncio
async def test_track_pages_in_global_units_and_keeps_local_geometry() -> None:
    view = _Transcript()
    async with _App(view).run_test() as pilot:
        bar = view.vertical_scrollbar
        middle = ["scroll_up"] * 8 + ["grab"] * 5 + ["scroll_down"] * 7
        await wait_for_layout(pilot, lambda: _actions(bar) == middle)
        local = bar.position, bar.window_size, bar.window_virtual_size

        assert await pilot.click(bar, offset=(0, 0))
        assert view.seeks == pytest.approx([1 / 6])
        assert await pilot.click(bar, offset=(0, 19))
        assert view.seeks[-1] == pytest.approx(0.5)
        assert (bar.position, bar.window_size, bar.window_virtual_size) == local

        for position, action in [(-10, "scroll_down"), (10, "scroll_up")]:
            view.metrics = (position, 1, 0.25)
            bar.refresh()
            expected = ["grab"] * 5 + [action] * 15 if position < 0 else [action] * 15 + ["grab"] * 5
            await wait_for_layout(pilot, lambda: _actions(bar) == expected)

        view.metrics = (0, 1, 1)
        bar.refresh()
        await wait_for_layout(pilot, lambda: _actions(bar) == [None] * 20)


@pytest.mark.asyncio
@pytest.mark.parametrize(("fraction", "start", "grip"), [(0.5, 8, 0), (0.5, 8, 4), (0.25, 4, 2)])
async def test_drag_from_painted_grip_to_middle_and_endpoints(fraction: float, start: int, grip: int) -> None:
    view = _Transcript()
    view.metrics = (fraction * 8, 10, 2)
    grab_row = start + grip
    async with _App(view).run_test() as pilot:
        bar = view.vertical_scrollbar
        expected_actions = ["scroll_up"] * start + ["grab"] * 5 + ["scroll_down"] * (15 - start)
        await wait_for_layout(pilot, lambda: _actions(bar) == expected_actions)
        assert await pilot.mouse_down(bar, offset=(0, grab_row))
        await wait_for_layout(pilot, lambda: bar.grabbed is not None)
        assert await pilot.hover(bar, offset=(0, grab_row))
        assert view.seeks == []
        for row, expected in [(grab_row + 1, (start + 1) / 15), (0, 0), (19, 1), (grab_row, fraction)]:
            assert await pilot.hover(bar, offset=(0, row))
            assert view.seeks[-1] == expected
        assert await pilot.mouse_up(bar, offset=(0, grab_row))
        assert bar.grabbed is None
        assert await pilot.hover(bar, offset=(0, 0))
        assert view.seeks == [(start + 1) / 15, 0, 1, fraction]
        assert view.scroll_y == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("fraction", "row", "opposite", "endpoint"), [(0.985, 19, 0, 1), (0.015, 0, 19, 0)])
async def test_drag_back_to_thumb_at_track_edge_reaches_exact_endpoint(
    fraction: float, row: int, opposite: int, endpoint: int
) -> None:
    view = _Transcript()
    view.thumb_rows = 1
    view.metrics = (fraction * 8, 10, 2)
    async with _App(view).run_test() as pilot:
        bar = view.vertical_scrollbar
        await wait_for_layout(pilot, lambda: _actions(bar)[row : row + 1] == ["grab"])
        assert await pilot.mouse_down(bar, offset=(0, row))
        assert await pilot.hover(bar, offset=(0, opposite))
        assert view.seeks == [1 - endpoint]
        assert await pilot.hover(bar, offset=(0, row))
        assert view.seeks == [1 - endpoint, endpoint]
        assert await pilot.mouse_up(bar, offset=(0, row))


@pytest.mark.asyncio
async def test_tail_resize_under_stationary_pointer_does_not_seek() -> None:
    view = _Transcript()
    async with _App(view).run_test() as pilot:
        bar = view.vertical_scrollbar
        await wait_for_layout(pilot, lambda: _actions(bar)[8:9] == ["grab"])
        assert await pilot.mouse_down(bar, offset=(0, 8))
        assert await pilot.hover(bar, offset=(0, 19))
        assert view.seeks == [1]
        view.styles.height = 22
        await wait_for_layout(pilot, lambda: _actions(bar) == ["scroll_up"] * 17 + ["grab"] * 5)
        assert await pilot.hover(bar, offset=(0, 19))
        assert await pilot.mouse_up(bar, offset=(0, 19))
        assert view.seeks == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("height", "thumb", "actions"),
    [(1, 5, [None]), (2, 5, ["grab", "scroll_down"]), (3, 0, [None] * 3), (3, -1, [None] * 3)],
)
async def test_tiny_track_and_invalid_thumb(height: int, thumb: int, actions: list[str | None]) -> None:
    view = _Transcript()
    view.styles.height = height
    view.thumb_rows = thumb
    async with _App(view).run_test() as pilot:
        await wait_for_layout(pilot, lambda: _actions(view.vertical_scrollbar) == actions)


@pytest.mark.asyncio
async def test_plain_view_uses_native_drag_and_page() -> None:
    view = _View()
    async with _App(view).run_test() as pilot:
        bar = view.vertical_scrollbar
        await wait_for_layout(pilot, lambda: len(_actions(bar)) == 20 and _actions(bar)[0] == "grab")
        assert await pilot.mouse_down(bar)
        assert await pilot.hover(bar, offset=(0, 4))
        await wait_for_layout(pilot, lambda: view.scroll_y > 0)
        assert await pilot.mouse_up(bar, offset=(0, 4))
        before = view.scroll_y
        assert await pilot.click(bar, offset=(0, 19))
        await wait_for_layout(pilot, lambda: view.scroll_y > before)
