"""Display-only terminal control sanitization for the Textual UI.

The UI is a renderer, not a terminal emulator. PTY output and other dynamic text
may contain cursor movement, mode switches, hyperlinks, resets, and other control
bytes that must never be forwarded as widget text. The parser in this module is
shared by the readable terminal transcript and a final Textual line filter; raw
model, tool, and persisted values remain untouched.
"""

from __future__ import annotations

from enum import Enum, auto

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style
from textual.color import Color
from textual.filter import LineFilter


class _State(Enum):
    NORMAL = auto()
    ESC = auto()
    ESC_INTERMEDIATE = auto()
    CSI = auto()
    OSC = auto()
    STRING = auto()
    OSC_ESC = auto()
    STRING_ESC = auto()


class _TerminalControlParser:
    """Incrementally remove terminal controls under a configurable display policy."""

    _C1_CSI = 0x9B
    _C1_OSC = 0x9D
    _C1_ST = 0x9C
    _C1_STRING_INTRODUCERS = frozenset({0x90, 0x98, 0x9E, 0x9F})
    _CANCEL_CONTROLS = frozenset({0x18, 0x1A})

    def __init__(self, *, preserve_width: bool, terminal_transcript: bool) -> None:
        self._preserve_width = preserve_width
        self._terminal_transcript = terminal_transcript
        self._state = _State.NORMAL
        self._pending_cr = False

    def reset(self) -> None:
        """Drop any pending partial escape/control sequence."""
        self._state = _State.NORMAL
        self._pending_cr = False

    def flush(self) -> str:
        """Finish the stream, discarding incomplete escape/control sequences."""
        self.reset()
        return ""

    def feed(self, text: str) -> str:
        """Return display-safe text while retaining partial parser state."""
        if not text:
            return ""

        out: list[str] = []
        suppressed: list[str] = []

        def flush_suppressed() -> None:
            if self._preserve_width and suppressed:
                if width := cell_len("".join(suppressed)):
                    out.append(" " * width)
            suppressed.clear()

        def suppress(ch: str) -> None:
            if self._preserve_width:
                suppressed.append(ch)

        def emit(ch: str) -> None:
            flush_suppressed()
            out.append(ch)

        index = 0
        length = len(text)
        while index < length:
            ch = text[index]
            code = ord(ch)
            state = self._state

            if state is _State.NORMAL and self._terminal_transcript and self._pending_cr:
                self._pending_cr = False
                if ch == "\n":
                    index += 1
                    continue

            if state is _State.NORMAL:
                if ch == "\x1b":
                    suppress(ch)
                    self._state = _State.ESC
                elif code == self._C1_CSI:
                    suppress(ch)
                    self._state = _State.CSI
                elif code == self._C1_OSC:
                    suppress(ch)
                    self._state = _State.OSC
                elif code in self._C1_STRING_INTRODUCERS:
                    suppress(ch)
                    self._state = _State.STRING
                elif ch == "\r" and self._terminal_transcript:
                    # CRLF is a newline; standalone CR progress redraws become
                    # stable transcript line breaks.
                    emit("\n")
                    self._pending_cr = True
                elif (ch == "\b" or ch == "\x7f") and self._terminal_transcript:
                    flush_suppressed()
                    self._erase_previous_output_char(out)
                elif ch == "\n" or ch == "\t":
                    emit(ch)
                elif self._is_c0_or_c1_control(code):
                    suppress(ch)
                else:
                    emit(ch)

            elif state is _State.ESC:
                suppress(ch)
                if ch == "\x1b":
                    self._state = _State.ESC
                elif ch == "[":
                    self._state = _State.CSI
                elif ch == "]":
                    self._state = _State.OSC
                elif ch in {"P", "^", "_", "X"}:
                    self._state = _State.STRING
                elif code == self._C1_CSI:
                    self._state = _State.CSI
                elif code == self._C1_OSC:
                    self._state = _State.OSC
                elif code in self._C1_STRING_INTRODUCERS:
                    self._state = _State.STRING
                elif code in self._CANCEL_CONTROLS:
                    self._state = _State.NORMAL
                elif 0x20 <= code <= 0x2F:
                    self._state = _State.ESC_INTERMEDIATE
                elif 0x30 <= code <= 0x7E:
                    # Complete two-byte sequences include ESC 7/8 and ESC c.
                    self._state = _State.NORMAL
                elif code > 0x9F:
                    # Invalid escape bytes are suppressed, then parsing resumes.
                    self._state = _State.NORMAL

            elif state is _State.ESC_INTERMEDIATE:
                suppress(ch)
                if ch == "\x1b":
                    self._state = _State.ESC
                elif code in self._CANCEL_CONTROLS:
                    self._state = _State.NORMAL
                elif 0x30 <= code <= 0x7E:
                    self._state = _State.NORMAL
                elif not (0x20 <= code <= 0x2F) and code > 0x1F:
                    self._state = _State.NORMAL

            elif state is _State.CSI:
                suppress(ch)
                if ch == "\x1b":
                    self._state = _State.ESC
                elif code == self._C1_CSI:
                    self._state = _State.CSI
                elif code == self._C1_OSC:
                    self._state = _State.OSC
                elif code in self._C1_STRING_INTRODUCERS:
                    self._state = _State.STRING
                elif code in self._CANCEL_CONTROLS:
                    self._state = _State.NORMAL
                elif 0x40 <= code <= 0x7E:
                    # CSI final byte range per ECMA-48.
                    self._state = _State.NORMAL

            elif state is _State.OSC:
                suppress(ch)
                if ch == "\x07" or code == self._C1_ST:
                    self._state = _State.NORMAL
                elif code in self._CANCEL_CONTROLS:
                    self._state = _State.NORMAL
                elif ch == "\x1b":
                    self._state = _State.OSC_ESC

            elif state is _State.OSC_ESC:
                suppress(ch)
                if ch == "\\" or ch == "\x07" or code == self._C1_ST:
                    self._state = _State.NORMAL
                elif code in self._CANCEL_CONTROLS:
                    self._state = _State.NORMAL
                elif ch != "\x1b":
                    self._state = _State.OSC

            elif state is _State.STRING:
                suppress(ch)
                if code == self._C1_ST or code in self._CANCEL_CONTROLS:
                    self._state = _State.NORMAL
                elif ch == "\x1b":
                    self._state = _State.STRING_ESC

            elif state is _State.STRING_ESC:
                suppress(ch)
                if ch == "\\" or code == self._C1_ST or code in self._CANCEL_CONTROLS:
                    self._state = _State.NORMAL
                elif ch != "\x1b":
                    self._state = _State.STRING

            index += 1

        flush_suppressed()
        return "".join(out)

    @staticmethod
    def _is_c0_or_c1_control(code: int) -> bool:
        return (0 <= code < 32) or (0x7F <= code <= 0x9F)

    @staticmethod
    def _erase_previous_output_char(out: list[str]) -> None:
        # Do not erase across emitted line boundaries or parser feed calls.
        if out and out[-1] != "\n":
            out.pop()


class TerminalDisplayNormalizer:
    """Incrementally sanitize PTY output for the readable terminal transcript.

    Standalone carriage returns become newlines so progress redraws remain
    readable. Backspace/delete remove previous characters only within the current
    emitted chunk; earlier RichLog lines are not mutated.
    """

    def __init__(self) -> None:
        self._parser = _TerminalControlParser(
            preserve_width=False,
            terminal_transcript=True,
        )

    def reset(self) -> None:
        """Drop any pending partial escape/control sequence."""
        self._parser.reset()

    def flush(self) -> str:
        """Finish the stream, discarding incomplete escape/control sequences."""
        return self._parser.flush()

    def feed(self, text: str) -> str:
        """Return a display-safe representation of incremental terminal output."""
        return self._parser.feed(text)


class TerminalControlFilter(LineFilter):
    """Remove untrusted terminal controls from final Textual widget segments."""

    @staticmethod
    def _safe_style(segment: Segment) -> Style | None:
        """Drop unsafe link metadata before Rich turns it into an OSC hyperlink."""
        style = segment.style
        if style is not None:
            link = style.link
            if link is not None and any(_TerminalControlParser._is_c0_or_c1_control(ord(ch)) for ch in link):
                return style.update_link(None)
        return style

    def apply(self, segments: list[Segment], background: Color) -> list[Segment]:
        """Sanitize one independently rendered line while preserving Rich styles."""
        del background
        parser = _TerminalControlParser(
            preserve_width=True,
            terminal_transcript=False,
        )
        return [Segment(parser.feed(segment.text), self._safe_style(segment), segment.control) for segment in segments]
