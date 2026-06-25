"""Display-only terminal stream normalization for the Textual sidebar.

The sidebar terminal is a readable transcript, not a terminal emulator.  Real PTY
output may contain cursor movement, erase commands, hyperlinks, carriage-return
progress redraws, and other control bytes that RichLog/Textual should not receive
raw.  This module strips terminal controls and keeps stable text suitable for a
log-style widget while leaving the model-facing command output untouched.
"""

from __future__ import annotations

from enum import Enum, auto


class _State(Enum):
    NORMAL = auto()
    ESC = auto()
    CSI = auto()
    OSC = auto()
    STRING = auto()
    OSC_ESC = auto()
    STRING_ESC = auto()


class TerminalDisplayNormalizer:
    """Incrementally sanitize terminal output for a log-style display.

    The normalizer is intentionally conservative: it strips terminal control
    sequences instead of trying to emulate screen state. Standalone carriage
    returns become newlines so progress redraws become readable transcript lines
    rather than raw cursor controls. Backspace/delete remove previous characters
    only within the currently emitted chunk; earlier RichLog lines are not
    mutated.
    """

    def __init__(self) -> None:
        self._state = _State.NORMAL

    def reset(self) -> None:
        """Drop any pending partial escape/control sequence."""
        self._state = _State.NORMAL

    def flush(self) -> str:
        """Finish the stream, discarding incomplete escape/control sequences."""
        self.reset()
        return ""

    def feed(self, text: str) -> str:
        """Return a display-safe representation of ``text``.

        Args:
            text: Incremental terminal output text. It may split escape sequences
                across calls.
        """
        if not text:
            return ""

        out: list[str] = []
        index = 0
        length = len(text)
        while index < length:
            ch = text[index]
            code = ord(ch)
            state = self._state

            if state is _State.NORMAL:
                if ch == "\x1b":
                    self._state = _State.ESC
                elif ch == "\r":
                    # CRLF is a newline; standalone CR progress redraws become
                    # stable transcript line breaks.
                    if index + 1 < length and text[index + 1] == "\n":
                        out.append("\n")
                        index += 1
                    else:
                        out.append("\n")
                elif ch == "\b" or ch == "\x7f":
                    self._erase_previous_output_char(out)
                elif ch == "\n" or ch == "\t":
                    out.append(ch)
                elif self._is_c0_or_c1_control(code):
                    # Drop BEL, NUL, form-feed, vertical tab, etc. RichLog is a
                    # display surface, not a control receiver.
                    pass
                else:
                    out.append(ch)

            elif state is _State.ESC:
                if ch == "[":
                    self._state = _State.CSI
                elif ch == "]":
                    self._state = _State.OSC
                elif ch in {"P", "^", "_", "X"}:
                    self._state = _State.STRING
                elif 0x40 <= code <= 0x5F or 0x60 <= code <= 0x7E:
                    # Two-byte escape sequence: ESC c, ESC 7, ESC 8, etc.
                    self._state = _State.NORMAL
                elif self._is_c0_or_c1_control(code):
                    # Ignore controls inside escape dispatch.
                    pass
                else:
                    self._state = _State.NORMAL

            elif state is _State.CSI:
                # CSI final byte range per ECMA-48.
                if 0x40 <= code <= 0x7E:
                    self._state = _State.NORMAL

            elif state is _State.OSC:
                if ch == "\x07":
                    self._state = _State.NORMAL
                elif ch == "\x1b":
                    self._state = _State.OSC_ESC

            elif state is _State.OSC_ESC:
                self._state = _State.NORMAL if ch == "\\" else _State.OSC

            elif state is _State.STRING:
                if ch == "\x1b":
                    self._state = _State.STRING_ESC

            elif state is _State.STRING_ESC:
                self._state = _State.NORMAL if ch == "\\" else _State.STRING

            index += 1

        return "".join(out)

    @staticmethod
    def _is_c0_or_c1_control(code: int) -> bool:
        return (0 <= code < 32) or (0x7F <= code <= 0x9F)

    @staticmethod
    def _erase_previous_output_char(out: list[str]) -> None:
        # Do not erase across emitted line boundaries.
        if out and out[-1] != "\n":
            out.pop()
