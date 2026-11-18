import pytest
from rich.segment import Segment
from rich.text import Text
from textual.app import App
from textual.geometry import Offset
from textual.selection import Selection
from textual.strip import Strip

from kolega_code.cli.tui.widgets import LogOutputLog, TerminalOutputLog

from ._app_test_utils import wait_for_layout


@pytest.fixture(params=[TerminalOutputLog, LogOutputLog])
def log_type(request):
    return request.param


def test_finite_log_selection_reads_only_the_selected_strip(log_type):
    reads = 0

    class CountingStrip(Strip):
        @property
        def text(self) -> str:
            nonlocal reads
            reads += 1
            return super().text

    log = log_type()
    log.lines = [CountingStrip([Segment(f"{index:04d}{'x' * 76}")]) for index in range(2000)]

    assert log.get_selection(Selection(Offset(0, 1700), Offset(4, 1700))) == ("1700", "\n")
    assert reads == 1


@pytest.mark.parametrize(
    ("rows", "selection"),
    [
        pytest.param([], Selection(None, None), id="no-rows"),
        pytest.param(["  ", "\t "], Selection(Offset(0, 0), Offset(1, 1)), id="only-blank-buffer"),
        pytest.param(["head ", "  ", "tail "], Selection(Offset(0, 1), Offset(3, 1)), id="only-blank-selection"),
        pytest.param(["head ", "  ", "tail "], Selection(Offset(1, 0), Offset(2, 2)), id="multiline-with-blank"),
        pytest.param(["head ", "  ", "tail "], Selection(Offset(1, 0), Offset(3, 1)), id="end-on-interior-blank"),
        pytest.param(["head ", "  "], Selection(Offset(1, 0), Offset(2, 1)), id="end-on-final-blank"),
        pytest.param(["first", "  padded  ", "last"], Selection(Offset(1, 1), Offset(10, 1)), id="padded"),
        pytest.param(["first", "middle ", "last"], Selection(None, Offset(2, 1)), id="from-start"),
        pytest.param(["first", "middle ", "last"], Selection(Offset(1, 1), None), id="to-end"),
        pytest.param(["first", "middle ", "last"], Selection(None, None), id="copy-all"),
        pytest.param(["first", "middle ", "last"], Selection(Offset(1, 1), Offset(2, 9)), id="end-beyond-buffer"),
    ],
)
def test_log_copy_matches_full_buffer_extraction(log_type, rows, selection):
    log = log_type()
    log.lines = [Strip([Segment(text)]) for text in rows]
    joined = "\n".join(text.rstrip() for text in rows)
    expected = (selection.extract(joined), "\n") if joined.strip() else None

    assert log.get_selection(selection) == expected


@pytest.mark.asyncio
async def test_log_copy_uses_wrapped_row_offsets_after_eviction(log_type):
    log = log_type(wrap=True, max_lines=3)

    class LogApp(App):
        def compose(self):
            yield log

    async with LogApp().run_test() as pilot:
        await wait_for_layout(pilot, lambda: log._size_known)
        log.write("old", width=5)
        log.write(Text("abcdefghijklmn", overflow="fold"), width=5)

        assert log._start_line == 1
        assert [strip.text.rstrip() for strip in log.lines] == ["abcde", "fghij", "klmn"]
        assert log.get_selection(Selection(Offset(2, 0), Offset(2, 2))) == ("cde\nfghij\nkl", "\n")
