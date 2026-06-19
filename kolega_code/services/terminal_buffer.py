"""Bounded output buffering and token-capping for terminal sessions.

Replaces the previous fast-LLM "compression" of terminal output with a
deterministic head-tail buffer (codex-style): keep the beginning and end of
the output, drop the middle with a marker, and cap the returned text to a
token budget. No model calls, fast and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

# Byte caps for the rolling per-read buffer. We keep the first HEAD_BYTES and
# the last TAIL_BYTES of whatever a command emits between reads; anything in
# between is dropped with a marker. This bounds memory regardless of how much a
# command prints.
HEAD_BYTES = 512 * 1024
TAIL_BYTES = 512 * 1024

# Yield-time clamps (milliseconds). A write/exec waits up to MAX_YIELD_MS for
# output or exit; an empty poll may wait much longer.
MIN_YIELD_MS = 250
MAX_YIELD_MS = 30_000
MIN_POLL_MS = 5_000
MAX_POLL_MS = 300_000
DEFAULT_YIELD_MS = 10_000


def clamp_yield(value, *, poll: bool) -> int:
    """Clamp a requested yield window to the allowed range for its kind."""
    try:
        millis = int(value)
    except (TypeError, ValueError):
        millis = DEFAULT_YIELD_MS
    if millis <= 0:
        millis = DEFAULT_YIELD_MS
    low = MIN_POLL_MS if poll else MIN_YIELD_MS
    high = MAX_POLL_MS if poll else MAX_YIELD_MS
    return max(low, min(high, millis))


def _omitted_marker(num_bytes: int) -> str:
    return f"\n[... omitted {num_bytes:,} bytes ...]\n"


class HeadTailBuffer:
    """Accumulates bytes, retaining only the head and tail past a size cap."""

    def __init__(self, head_bytes: int = HEAD_BYTES, tail_bytes: int = TAIL_BYTES):
        self._head_cap = head_bytes
        self._tail_cap = tail_bytes
        self._head = bytearray()
        self._tail = bytearray()
        self.total_bytes = 0

    def append(self, data: bytes) -> None:
        if not data:
            return
        self.total_bytes += len(data)
        # Fill the head first (it never changes once full).
        if len(self._head) < self._head_cap:
            take = self._head_cap - len(self._head)
            self._head += data[:take]
            data = data[take:]
        if data:
            self._tail += data
            excess = len(self._tail) - self._tail_cap
            if excess > 0:
                del self._tail[:excess]

    @property
    def omitted_bytes(self) -> int:
        return max(0, self.total_bytes - len(self._head) - len(self._tail))

    def text(self) -> str:
        """Render the retained bytes, with a marker for any dropped middle.

        Decoded with errors="replace" so a multibyte character split at the
        head/tail boundary degrades gracefully instead of raising.
        """
        omitted = self.omitted_bytes
        if omitted == 0:
            return (bytes(self._head) + bytes(self._tail)).decode("utf-8", errors="replace")
        head = bytes(self._head).decode("utf-8", errors="replace")
        tail = bytes(self._tail).decode("utf-8", errors="replace")
        return head + _omitted_marker(omitted) + tail

    def reset(self) -> None:
        self._head.clear()
        self._tail.clear()
        self.total_bytes = 0

    def __len__(self) -> int:
        return self.total_bytes


# --- token capping ---------------------------------------------------------

_ENCODING_NAME = "o200k_base"
_encoder = None
_encoder_failed = False


def _get_encoder():
    """Lazily load the tiktoken encoder, falling back to None if unavailable."""
    global _encoder, _encoder_failed
    if _encoder is not None or _encoder_failed:
        return _encoder
    try:
        import tiktoken

        _encoder = tiktoken.get_encoding(_ENCODING_NAME)
    except Exception:
        # tiktoken may need to download the encoding on first use; if that
        # fails (e.g. offline), fall back to a character heuristic.
        _encoder_failed = True
        _encoder = None
    return _encoder


@dataclass
class CappedOutput:
    text: str
    truncated: bool
    original_token_count: int


def _truncation_marker(max_tokens: int) -> str:
    return f"\n[... output truncated to fit {max_tokens} tokens ...]\n"


def cap_tokens(text: str, max_tokens: int) -> CappedOutput:
    """Cap ``text`` to ``max_tokens``, dropping the middle if needed.

    Returns the (possibly truncated) text, whether truncation happened, and the
    original token count so callers can tell the model there is more output.
    """
    if max_tokens <= 0:
        max_tokens = 1

    encoder = _get_encoder()
    if encoder is None:
        # Heuristic fallback: ~4 characters per token.
        approx = max(1, (len(text) + 3) // 4)
        if approx <= max_tokens:
            return CappedOutput(text, False, approx)
        budget_chars = max_tokens * 4
        head = budget_chars // 2
        tail = budget_chars - head
        capped = text[:head] + _truncation_marker(max_tokens) + (text[-tail:] if tail else "")
        return CappedOutput(capped, True, approx)

    tokens = encoder.encode(text)
    original = len(tokens)
    if original <= max_tokens:
        return CappedOutput(text, False, original)
    head = max_tokens // 2
    tail = max_tokens - head
    capped = (
        encoder.decode(tokens[:head])
        + _truncation_marker(max_tokens)
        + (encoder.decode(tokens[-tail:]) if tail else "")
    )
    return CappedOutput(capped, True, original)
