import os

from kolega_code.services.terminal_buffer import (
    GLOBAL_MAX_TOOL_OUTPUT_TOKENS,
    LINE_CAP_BYTES,
    LINE_TRUNCATION_MARKER,
    MAX_POLL_MS,
    MAX_YIELD_MS,
    MIN_POLL_MS,
    MIN_YIELD_MS,
    HeadTailBuffer,
    TerminalOutputAccumulator,
    TerminalSpillStore,
    cap_tokens,
    clamp_output_tokens,
    clamp_yield,
)


def test_headtail_no_truncation():
    buffer = HeadTailBuffer(head_bytes=10, tail_bytes=10)
    buffer.append(b"hello")
    assert buffer.text() == "hello"
    assert buffer.omitted_bytes == 0


def test_headtail_truncates_middle_with_marker():
    buffer = HeadTailBuffer(head_bytes=5, tail_bytes=5)
    buffer.append(b"A" * 5 + b"B" * 100 + b"C" * 5)
    text = buffer.text()
    assert text.startswith("AAAAA")
    assert text.endswith("CCCCC")
    assert "omitted" in text
    assert buffer.omitted_bytes == 100


def test_headtail_append_in_chunks():
    buffer = HeadTailBuffer(head_bytes=4, tail_bytes=4)
    for chunk in (b"12", b"34", b"56", b"78", b"90"):
        buffer.append(chunk)
    text = buffer.text()
    assert text.startswith("1234")
    assert text.endswith("7890")
    assert buffer.total_bytes == 10
    assert buffer.omitted_bytes == 2


def test_headtail_reset():
    buffer = HeadTailBuffer()
    buffer.append(b"x")
    buffer.reset()
    assert buffer.text() == ""
    assert len(buffer) == 0


def test_cap_tokens_under_budget():
    out = cap_tokens("short text", 1000)
    assert out.truncated is False
    assert out.text == "short text"


def test_cap_tokens_over_budget_truncates_middle():
    text = "word " * 5000
    out = cap_tokens(text, 50)
    assert out.truncated is True
    assert out.original_token_count > 50
    assert "truncated to fit" in out.text
    assert len(out.text) < len(text)


def test_output_token_request_is_globally_clamped():
    assert clamp_output_tokens(1_000_000) == GLOBAL_MAX_TOOL_OUTPUT_TOKENS
    out = cap_tokens("x" * 100_000, 1_000_000)
    assert len(out.text) <= GLOBAL_MAX_TOOL_OUTPUT_TOKENS * 4


def test_accumulator_spills_complete_stream_before_preview_truncation(tmp_path):
    store = TerminalSpillStore(tmp_path / "terminal-output")
    accumulator = TerminalOutputAccumulator(store)
    head = "BEGIN\n" + ("head-" + "h" * 90 + "\n") * 250
    middle = "RECOVERABLE-MIDDLE\n" + ("middle-" + "m" * 88 + "\n") * 300
    tail = ("tail-" + "t" * 90 + "\n") * 350 + "END\n"
    complete = head + middle + tail

    for start in range(0, len(complete), 997):
        accumulator.append_text(complete[start : start + 997])
    accumulator.finalize()
    result = accumulator.read_delta(1_000_000)

    assert result.spill_path is not None
    spill_path = tmp_path / "terminal-output" / os.path.basename(result.spill_path)
    assert spill_path.read_text(encoding="utf-8") == complete
    assert result.spill_bytes == len(complete.encode("utf-8"))
    assert "BEGIN" in result.text
    assert "END" in result.text
    assert "RECOVERABLE-MIDDLE" not in result.text
    assert result.preview_omitted_bytes > 0
    assert len(result.text) <= GLOBAL_MAX_TOOL_OUTPUT_TOKENS * 4
    assert spill_path.stat().st_mode & 0o777 == 0o400


def test_spill_ids_continue_after_existing_files(tmp_path):
    root = tmp_path / "terminal-output"
    root.mkdir()
    (root / "000007.exec_command.log").write_text("old", encoding="utf-8")
    store = TerminalSpillStore(root)

    path = store.allocate_path("exec_command")

    assert path.name == "000008.exec_command.log"
    assert (root / "000007.exec_command.log").read_text(encoding="utf-8") == "old"


def test_streaming_line_cap_cannot_be_bypassed_across_reads():
    accumulator = TerminalOutputAccumulator(None)
    accumulator.append_text("a" * 800)
    first = accumulator.read_delta(10_000)
    accumulator.append_text("b" * 800)
    second = accumulator.read_delta(10_000)
    accumulator.append_text("c" * 100)
    third = accumulator.read_delta(10_000)
    accumulator.append_text("\nnext\n")
    fourth = accumulator.read_delta(10_000)

    combined_line = first.text + second.text + third.text
    assert len(combined_line.encode("utf-8")) <= LINE_CAP_BYTES
    assert combined_line.count(LINE_TRUNCATION_MARKER) == 1
    assert second.line_truncated_count == 1
    assert second.line_truncated_bytes > 0
    assert third.line_truncated_count == 1
    assert third.line_truncated_bytes == 100
    assert fourth.text == "\nnext\n"


def test_split_utf8_is_normalized_before_spill_and_preview(tmp_path):
    accumulator = TerminalOutputAccumulator(TerminalSpillStore(tmp_path / "terminal-output"))
    euro = "€".encode("utf-8")
    accumulator.append_bytes(euro[:1])
    accumulator.append_bytes(euro[1:])
    accumulator.finalize()

    result = accumulator.read_delta(100)

    assert result.text == "€"
    assert "�" not in result.text
    assert result.original_token_count == 1


def test_clamp_yield_write():
    assert clamp_yield(10, poll=False) == MIN_YIELD_MS
    assert clamp_yield(999999, poll=False) == MAX_YIELD_MS
    assert clamp_yield(None, poll=False) == 10000


def test_clamp_yield_poll():
    assert clamp_yield(10, poll=True) == MIN_POLL_MS
    assert clamp_yield(999999, poll=True) == MAX_POLL_MS
