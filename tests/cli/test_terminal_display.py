from kolega_code.cli.tui.terminal_display import TerminalDisplayNormalizer


def _normalize(*chunks: str) -> str:
    normalizer = TerminalDisplayNormalizer()
    return "".join(normalizer.feed(chunk) for chunk in chunks) + normalizer.flush()


def test_terminal_display_normalizer_strips_ansi_sgr_and_csi_controls() -> None:
    text = _normalize("plain \x1b[31mred\x1b[0m \x1b[Kdone\x1b[2J")

    assert text == "plain red done"
    assert "\x1b" not in text


def test_terminal_display_normalizer_strips_osc_hyperlinks_split_across_chunks() -> None:
    text = _normalize("before ", "\x1b]8;;https://example.com\x1b\\", "link", "\x1b]8;;\x1b\\ after")

    assert text == "before link after"
    assert "https://example.com" not in text
    assert "\x1b" not in text


def test_terminal_display_normalizer_converts_carriage_return_progress_to_lines() -> None:
    assert _normalize("10%\r20%\rdone\n") == "10%\n20%\ndone\n"
    assert _normalize("line\r\nnext\n") == "line\nnext\n"


def test_terminal_display_normalizer_handles_backspace_rewrites_within_chunk() -> None:
    assert _normalize("abc\b\bd\n") == "ad\n"


def test_terminal_display_normalizer_drops_unknown_control_characters() -> None:
    assert _normalize("a\x00\x07b\x0cc") == "abc"


def test_terminal_display_normalizer_holds_split_escape_sequences() -> None:
    normalizer = TerminalDisplayNormalizer()

    assert normalizer.feed("a\x1b[") == "a"
    assert normalizer.feed("31mred") == "red"
    assert normalizer.flush() == ""
