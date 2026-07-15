import pytest
from rich.cells import cell_len
from rich.console import Console
from rich.segment import Segment
from rich.style import Style
from textual.color import Color
from textual.strip import Strip

from kolega_code.cli.tui.terminal_display import TerminalControlFilter, TerminalDisplayNormalizer


def _normalize(*chunks: str) -> str:
    normalizer = TerminalDisplayNormalizer()
    return "".join(normalizer.feed(chunk) for chunk in chunks) + normalizer.flush()


def _filter_segments(*segments: Segment) -> list[Segment]:
    return TerminalControlFilter().apply(list(segments), Color.parse("#000000"))


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


def test_terminal_display_normalizer_handles_crlf_split_across_chunks() -> None:
    assert _normalize("line\r", "\nnext") == "line\nnext"
    assert _normalize("line\r", "next") == "line\nnext"


def test_terminal_display_normalizer_reset_clears_pending_crlf_state() -> None:
    normalizer = TerminalDisplayNormalizer()

    assert normalizer.feed("line\r") == "line\n"
    normalizer.reset()
    assert normalizer.feed("\nnext") == "\nnext"


def test_terminal_display_normalizer_handles_backspace_rewrites_within_chunk() -> None:
    assert _normalize("abc\b\bd\n") == "ad\n"


def test_terminal_display_normalizer_drops_unknown_control_characters() -> None:
    assert _normalize("a\x00\x07b\x0cc") == "abc"


def test_terminal_display_normalizer_holds_split_escape_sequences() -> None:
    normalizer = TerminalDisplayNormalizer()

    assert normalizer.feed("a\x1b[") == "a"
    assert normalizer.feed("31mred") == "red"
    assert normalizer.flush() == ""


@pytest.mark.parametrize(
    "control",
    [
        "\x1b[?1049l",
        "\x1b[?1000h",
        "\x1b[?1002h",
        "\x1b[?1003h",
        "\x1b[?1004h",
        "\x1b[?1006h",
        "\x1b[?2004h",
        "\x9b?1049l",
        "\x9b?1006h",
        "\x1bc",
        "\x1b7",
        "\x1b8",
        "\x1b(0",
    ],
)
def test_terminal_display_normalizer_strips_modes_resets_and_escape_sequences(control: str) -> None:
    assert _normalize(f"before{control}after") == "beforeafter"


@pytest.mark.parametrize(
    ("control", "payload"),
    [
        ("\x1b]2;osc-bel\x07", "osc-bel"),
        ("\x1b]2;osc-st\x1b\\", "osc-st"),
        ("\x9d2;osc-c1-st\x9c", "osc-c1-st"),
        ("\x1bPdcs-payload\x1b\\", "dcs-payload"),
        ("\x1bXsos-payload\x1b\\", "sos-payload"),
        ("\x1b^pm-payload\x1b\\", "pm-payload"),
        ("\x1b_apc-payload\x1b\\", "apc-payload"),
        ("\x90c1-dcs\x9c", "c1-dcs"),
        ("\x98c1-sos\x9c", "c1-sos"),
        ("\x9ec1-pm\x9c", "c1-pm"),
        ("\x9fc1-apc\x9c", "c1-apc"),
    ],
)
def test_terminal_display_normalizer_strips_string_controls(control: str, payload: str) -> None:
    text = _normalize(f"before{control}after")

    assert text == "beforeafter"
    assert payload not in text


@pytest.mark.parametrize(
    "control",
    [
        "\x1b[31m",
        "\x1b]2;split-osc\x1b\\",
        "\x1bPsplit-dcs\x1b\\",
        "\x9d2;split-c1-osc\x9c",
    ],
)
def test_terminal_display_normalizer_handles_every_split_point(control: str) -> None:
    for split_at in range(len(control) + 1):
        assert _normalize("before", control[:split_at], control[split_at:], "after") == "beforeafter"


@pytest.mark.parametrize(
    "control",
    [
        "\x1b[31\x18",
        "\x1b]2;cancelled\x1a",
        "\x1bPcancelled\x18",
    ],
)
def test_terminal_display_normalizer_resumes_safely_after_cancel_controls(control: str) -> None:
    assert _normalize(f"before{control}visible") == "beforevisible"


def test_terminal_display_normalizer_strips_malformed_and_intermediate_escapes() -> None:
    assert _normalize("before\x1b%Gafter\x1b\x00[31mend") == "beforeafterend"


def test_terminal_display_normalizer_discards_incomplete_sequences_on_flush() -> None:
    for incomplete in ("\x1b", "\x1b[?1049", "\x1b]unfinished", "\x1bPunfinished", "\x9b?1000"):
        assert _normalize(f"before{incomplete}") == "before"


def test_terminal_display_normalizer_preserves_unicode_newlines_and_tabs_only() -> None:
    standalone_controls = "\x00\x01\x07\x0b\x0c\x0e\x18\x1a\x7f\x80\x81\x85\x99\x9a\x9c"

    assert _normalize(f"héllo\t世界\n{standalone_controls}done") == "héllo\t世界\ndone"


def test_terminal_control_filter_handles_sequences_split_across_styled_segments() -> None:
    styles = [
        Style(color="red", meta={"source": "one"}),
        Style(bold=True, meta={"source": "two"}),
        Style(italic=True, meta={"source": "three"}),
        Style(color="blue", meta={"source": "four"}),
    ]
    source = [
        Segment("before\x1b[?10", styles[0]),
        Segment("49lmiddle\x1b]2;OSC-", styles[1]),
        Segment("PAYLOAD\x1b", styles[2]),
        Segment("\\after\x1bPDCS-PAYLOAD\x1b\\", styles[3]),
    ]

    filtered = _filter_segments(*source)
    text = "".join(segment.text for segment in filtered)

    assert "before" in text
    assert "middle" in text
    assert "after" in text
    assert "\x1b" not in text
    assert "1049" not in text
    assert "OSC-PAYLOAD" not in text
    assert "DCS-PAYLOAD" not in text
    assert [segment.style for segment in filtered] == styles
    assert [segment.control for segment in filtered] == [segment.control for segment in source]
    assert cell_len(text) == sum(segment.cell_length for segment in source)


def test_terminal_control_filter_preserves_width_for_wide_unicode_control_payloads() -> None:
    source = [
        Segment("before\x1b]2;\u6f22"),
        Segment("\U0001f642e\u0301\x07after", Style(italic=True)),
    ]
    filtered = _filter_segments(*source)
    text = "".join(segment.text for segment in filtered)

    assert "\u6f22" not in text
    assert "\U0001f642e\u0301" not in text
    assert text.startswith("before")
    assert text.endswith("after")
    assert cell_len(text) == sum(segment.cell_length for segment in source)


def test_terminal_control_filter_drops_backspace_delete_and_standalone_c1_without_rewriting_text() -> None:
    filtered = _filter_segments(Segment("ab\b\x7f\x85\x9ctext"))

    assert "".join(segment.text for segment in filtered) == "abtext"


def test_terminal_control_filter_preserves_rich_generated_styling_controls() -> None:
    filtered = _filter_segments(Segment("safe\x1b[?1049ltext", Style(color="red", bold=True)))
    rendered = Strip(filtered).render(Console(force_terminal=True, color_system="truecolor"))

    assert "\x1b[?1049l" not in rendered
    assert "\x1b[" in rendered
    assert "safe" in rendered
    assert "text" in rendered


@pytest.mark.parametrize(
    ("unsafe_link", "payload"),
    [
        ("https://example.invalid/\x9d2;C1-OSC-LINK\x07", "C1-OSC-LINK"),
        ("https://example.invalid/\x1bPDCS-LINK\x1b\\", "DCS-LINK"),
        ("https://example.invalid/\x1b[?1049h", "1049h"),
    ],
)
def test_terminal_control_filter_drops_unsafe_rich_link_controls(unsafe_link: str, payload: str) -> None:
    style = Style(color="red", bold=True, link=unsafe_link, meta={"source": "untrusted"})
    filtered = _filter_segments(Segment("linked text", style))
    filtered_style = filtered[0].style
    rendered = Strip(filtered).render(Console(force_terminal=True, color_system="truecolor"))

    assert filtered_style is not None
    assert filtered_style.link is None
    assert filtered_style.color == style.color
    assert filtered_style.bold == style.bold
    assert filtered_style.meta == style.meta
    assert payload not in rendered
    assert "linked text" in rendered


def test_terminal_control_filter_preserves_safe_rich_hyperlinks() -> None:
    style = Style(underline=True, link="https://example.com/safe")
    filtered = _filter_segments(Segment("safe link", style))
    rendered = Strip(filtered).render(Console(force_terminal=True, color_system="truecolor"))

    assert filtered[0].style is style
    assert "\x1b]8;" in rendered
    assert "https://example.com/safe" in rendered
    assert "safe link" in rendered


def test_terminal_control_filter_does_not_share_state_between_rendered_lines() -> None:
    control_filter = TerminalControlFilter()
    background = Color.parse("#000000")

    first = control_filter.apply([Segment("safe\x1b[?1049")], background)
    second = control_filter.apply([Segment("l suffix")], background)
    full_repaint = control_filter.apply([Segment("safe\x1b[?1049l suffix")], background)

    first_text = "".join(segment.text for segment in first)
    second_text = "".join(segment.text for segment in second)
    repaint_text = "".join(segment.text for segment in full_repaint)
    assert "\x1b" not in first_text
    assert second_text == "l suffix"
    assert "1049" not in repaint_text
    assert repaint_text.endswith(" suffix")
