from kolega_code.services.terminal_buffer import (
    MAX_POLL_MS,
    MAX_YIELD_MS,
    MIN_POLL_MS,
    MIN_YIELD_MS,
    HeadTailBuffer,
    cap_tokens,
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


def test_clamp_yield_write():
    assert clamp_yield(10, poll=False) == MIN_YIELD_MS
    assert clamp_yield(999999, poll=False) == MAX_YIELD_MS
    assert clamp_yield(None, poll=False) == 10000


def test_clamp_yield_poll():
    assert clamp_yield(10, poll=True) == MIN_POLL_MS
    assert clamp_yield(999999, poll=True) == MAX_POLL_MS
