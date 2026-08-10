"""Bounded, recoverable model-facing output for terminal sessions."""

from __future__ import annotations

import codecs
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

from kolega_code.local_state import ensure_private_dir, ensure_private_file

# A model may request less, but never more, from exec_command/write_stdin.
GLOBAL_MAX_TOOL_OUTPUT_TOKENS = 10_000

# Complete normalized terminal streams spill once they cross this threshold.
SPILL_THRESHOLD_BYTES = 50 * 1024

# Model-visible terminal deltas retain an approximately 20 KiB / 30 KiB
# head-tail preview before the final token cap.
PREVIEW_HEAD_BYTES = 20 * 1024
PREVIEW_TAIL_BYTES = 30 * 1024

# A single logical terminal line, including its explicit marker, may contribute
# at most this many UTF-8 bytes to model-visible output.
LINE_CAP_BYTES = 1024
LINE_TRUNCATION_MARKER = "… [line truncated]"

# Backwards-compatible names for callers/tests that construct HeadTailBuffer
# without explicit caps. Terminal sessions now use the smaller preview limits.
HEAD_BYTES = PREVIEW_HEAD_BYTES
TAIL_BYTES = PREVIEW_TAIL_BYTES

# Yield-time clamps (milliseconds). A write/exec waits up to MAX_YIELD_MS for
# output or exit; an empty poll may wait much longer.
MIN_YIELD_MS = 250
MAX_YIELD_MS = 30_000
MIN_POLL_MS = 5_000
MAX_POLL_MS = 300_000
DEFAULT_YIELD_MS = 10_000

_SPILL_NAME_RE = re.compile(r"^(?P<id>[0-9]+)\.[^.]+\.log$")


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


def clamp_output_tokens(value: object) -> int:
    """Return the effective terminal output budget for a model-facing call."""
    try:
        tokens = int(cast(Any, value))
    except (TypeError, ValueError):
        tokens = GLOBAL_MAX_TOOL_OUTPUT_TOKENS
    return max(1, min(tokens, GLOBAL_MAX_TOOL_OUTPUT_TOKENS))


def _omitted_marker(num_bytes: int, num_lines: int) -> str:
    noun = "line" if num_lines == 1 else "lines"
    return f"\n[... omitted {num_bytes:,} bytes across {num_lines:,} {noun} ...]\n"


class HeadTailBuffer:
    """Accumulate bytes while retaining a bounded head and rolling tail."""

    def __init__(self, head_bytes: int = HEAD_BYTES, tail_bytes: int = TAIL_BYTES):
        self._head_cap = head_bytes
        self._tail_cap = tail_bytes
        self._head = bytearray()
        self._tail = bytearray()
        self._omitted_line_breaks = 0
        self._omitted_ends_newline = False
        self.total_bytes = 0

    def append(self, data: bytes) -> None:
        if not data:
            return
        self.total_bytes += len(data)
        if len(self._head) < self._head_cap:
            take = self._head_cap - len(self._head)
            self._head += data[:take]
            data = data[take:]
        if data:
            self._tail += data
            excess = len(self._tail) - self._tail_cap
            if excess > 0:
                removed = bytes(self._tail[:excess])
                self._omitted_line_breaks += removed.count(b"\n")
                self._omitted_ends_newline = removed.endswith(b"\n")
                del self._tail[:excess]

    @property
    def omitted_bytes(self) -> int:
        return self.view().omitted_bytes

    @property
    def omitted_lines(self) -> int:
        return self.view().omitted_lines

    def view(self) -> "HeadTailView":
        """Return valid UTF-8 retained regions and exact omitted accounting."""
        raw_omitted = max(0, self.total_bytes - len(self._head) - len(self._tail))
        if raw_omitted == 0:
            text = (bytes(self._head) + bytes(self._tail)).decode("utf-8")
            return HeadTailView(head=text, tail="")

        head = bytes(self._head).decode("utf-8", errors="ignore")
        tail = bytes(self._tail).decode("utf-8", errors="ignore")
        retained_bytes = len(head.encode("utf-8")) + len(tail.encode("utf-8"))
        omitted_bytes = max(0, self.total_bytes - retained_bytes)
        skipped_tail_bytes = len(self._tail) - len(tail.encode("utf-8"))
        omitted_ends_newline = self._omitted_ends_newline if skipped_tail_bytes == 0 else False
        omitted_lines = self._omitted_line_breaks + (0 if omitted_ends_newline else 1)
        return HeadTailView(
            head=head,
            tail=tail,
            omitted_bytes=omitted_bytes,
            omitted_newlines=self._omitted_line_breaks,
            omitted_ends_newline=omitted_ends_newline,
            omitted_lines=max(1, omitted_lines),
        )

    def text(self) -> str:
        """Render retained valid UTF-8 with a middle-elision marker."""
        view = self.view()
        if view.omitted_bytes == 0:
            return view.head
        return view.head + _omitted_marker(view.omitted_bytes, view.omitted_lines) + view.tail

    def reset(self) -> None:
        self._head.clear()
        self._tail.clear()
        self._omitted_line_breaks = 0
        self._omitted_ends_newline = False
        self.total_bytes = 0

    def __len__(self) -> int:
        return self.total_bytes


@dataclass(frozen=True)
class HeadTailView:
    head: str
    tail: str
    omitted_bytes: int = 0
    omitted_newlines: int = 0
    omitted_ends_newline: bool = False
    omitted_lines: int = 0


@dataclass
class CappedOutput:
    text: str
    truncated: bool
    original_token_count: int
    spill_path: Optional[str] = None
    spill_bytes: int = 0
    line_truncated_count: int = 0
    line_truncated_bytes: int = 0
    preview_omitted_bytes: int = 0
    preview_omitted_lines: int = 0


def _truncation_marker(max_tokens: int) -> str:
    return f"\n[... output truncated to fit {max_tokens} tokens ...]\n"


def _estimated_tokens(char_count: int) -> int:
    return 0 if char_count <= 0 else (char_count + 3) // 4


def cap_chars(
    text: str,
    budget_chars: int,
    *,
    marker: str,
    original_token_count: Optional[int] = None,
) -> CappedOutput:
    """Cap text to an exact character budget with a deterministic head/tail."""
    budget = max(0, int(budget_chars))
    original = original_token_count if original_token_count is not None else _estimated_tokens(len(text))
    if len(text) <= budget:
        return CappedOutput(text, False, original)
    if budget == 0:
        return CappedOutput("", True, original)
    if len(marker) >= budget:
        return CappedOutput(marker[:budget], True, original)
    available = budget - len(marker)
    head = available // 2
    tail = available - head
    capped = text[:head] + marker + (text[-tail:] if tail else "")
    return CappedOutput(capped, True, original)


def cap_tokens(
    text: str,
    max_tokens: int,
    *,
    protected_suffix: str = "",
    original_token_count: Optional[int] = None,
    hard_limit: bool = True,
) -> CappedOutput:
    """Hard-cap terminal text while preserving a recovery suffix when possible.

    The four-characters-per-token estimate is intentionally deterministic and
    dependency-free. Unlike the previous implementation, the truncation marker
    itself is included in the character budget, so the returned ``text`` never
    exceeds ``effective_max_tokens * 4`` characters.
    """
    effective_max_tokens = clamp_output_tokens(max_tokens) if hard_limit else max(1, int(max_tokens))
    original = original_token_count if original_token_count is not None else _estimated_tokens(len(text))
    full = text + protected_suffix
    budget_chars = effective_max_tokens * 4
    if len(full) <= budget_chars:
        return CappedOutput(full, original > effective_max_tokens, original)

    marker = _truncation_marker(effective_max_tokens)
    suffix = protected_suffix if len(protected_suffix) <= budget_chars else ""
    available = max(0, budget_chars - len(marker) - len(suffix))
    if available == 0:
        # ``spill_path`` is also a structured result field, so an exceptionally
        # small caller-requested budget may omit the prose footer without losing
        # the recovery path.
        capped = suffix[-budget_chars:] if suffix else ""
    else:
        content = cap_chars(
            text,
            available + len(marker),
            marker=marker,
            original_token_count=original,
        ).text
        capped = content + suffix
    return CappedOutput(capped[:budget_chars], True, original)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while persisting terminal output")
        view = view[written:]


class TerminalSpillStore:
    """Allocate collision-safe terminal spill files under one session directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()
        self._next_id: Optional[int] = None

    def allocate_path(self, tool_name: str = "exec_command") -> Path:
        safe_tool = re.sub(r"[^a-z0-9_-]+", "-", tool_name.lower()).strip("-") or "terminal"
        with self._lock:
            ensure_private_dir(self.root)
            if self._next_id is None:
                existing_ids = [
                    int(match.group("id"))
                    for path in self.root.iterdir()
                    if path.is_file() and (match := _SPILL_NAME_RE.match(path.name))
                ]
                self._next_id = max(existing_ids, default=0) + 1
            while True:
                spill_id = self._next_id
                self._next_id += 1
                path = self.root / f"{spill_id:06d}.{safe_tool}.log"
                try:
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    continue
                os.close(fd)
                ensure_private_file(path)
                return path


class _SpillWriter:
    """Buffer up to the threshold, then persist every normalized byte."""

    def __init__(
        self,
        store: Optional[TerminalSpillStore],
        *,
        tool_name: str,
        backing_path: Optional[Path] = None,
    ) -> None:
        self.store = store
        self.tool_name = tool_name
        self.backing_path = backing_path
        self.total_bytes = 0
        self._path = backing_path
        self._fd: Optional[int] = None
        self._pending = bytearray()
        self._active = False
        self._finalized = False

    @property
    def path(self) -> Optional[Path]:
        return self._path if self._active else None

    def append(self, data: bytes) -> None:
        if not data:
            return
        if self._finalized:
            raise RuntimeError("Cannot append to finalized terminal output")
        self.total_bytes += len(data)

        # Detached sessions write directly to ``backing_path`` so their child
        # can continue appending after the Kolega process exits. We only decide
        # whether to expose that path here.
        if self.backing_path is not None:
            if self.total_bytes > SPILL_THRESHOLD_BYTES:
                self._active = True
            return

        if self._fd is not None:
            _write_all(self._fd, data)
            return

        if self.store is None:
            # Non-model internal managers may intentionally omit spill storage.
            # Keep only the threshold prefix so memory remains bounded.
            remaining = max(0, SPILL_THRESHOLD_BYTES + 1 - len(self._pending))
            self._pending.extend(data[:remaining])
            return

        if self.total_bytes <= SPILL_THRESHOLD_BYTES:
            self._pending.extend(data)
            return

        self._path = self.store.allocate_path(self.tool_name)
        self._fd = os.open(self._path, os.O_WRONLY | os.O_APPEND)
        _write_all(self._fd, bytes(self._pending))
        self._pending.clear()
        _write_all(self._fd, data)
        self._active = True

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._fd is not None:
            os.fsync(self._fd)
            os.close(self._fd)
            self._fd = None
        if self._active and self._path is not None:
            try:
                os.chmod(self._path, 0o400)
            except OSError:
                pass
        if self.backing_path is not None and not self._active:
            try:
                self.backing_path.unlink()
            except OSError:
                pass
            self._path = None
        self._pending.clear()


class _StreamingLineCap:
    """Cap logical lines across arbitrary streaming chunk/read boundaries."""

    def __init__(self, cap_bytes: int = LINE_CAP_BYTES):
        marker_bytes = len(LINE_TRUNCATION_MARKER.encode("utf-8"))
        self.cap_bytes = max(marker_bytes, cap_bytes)
        self._payload_bytes = max(
            0,
            self.cap_bytes - marker_bytes,
        )
        self._visible_bytes = 0
        self._line_truncated = False
        self._delta_line_counted = False
        self._delta_lines = 0
        self._delta_bytes = 0

    def append(self, text: str) -> bytes:
        rendered: list[str] = []
        for char in text:
            if char == "\n":
                rendered.append(char)
                self._visible_bytes = 0
                self._line_truncated = False
                self._delta_line_counted = False
                continue

            encoded_size = len(char.encode("utf-8"))
            if not self._line_truncated and self._visible_bytes + encoded_size <= self._payload_bytes:
                rendered.append(char)
                self._visible_bytes += encoded_size
                continue

            if not self._line_truncated:
                rendered.append(LINE_TRUNCATION_MARKER)
                self._line_truncated = True
            if not self._delta_line_counted:
                self._delta_lines += 1
                self._delta_line_counted = True
            self._delta_bytes += encoded_size
        return "".join(rendered).encode("utf-8")

    def take_delta_stats(self) -> tuple[int, int]:
        result = (self._delta_lines, self._delta_bytes)
        self._delta_lines = 0
        self._delta_bytes = 0
        # A continued logical line should count once again if a later read also
        # drops bytes, even though its inline marker remains one-per-line.
        self._delta_line_counted = False
        return result


def _allocate_preview_chars(head_length: int, tail_length: int, available: int) -> tuple[int, int]:
    """Allocate a bounded preview with the configured 20/30 head-tail ratio."""
    available = max(0, available)
    head = min(head_length, available * PREVIEW_HEAD_BYTES // (PREVIEW_HEAD_BYTES + PREVIEW_TAIL_BYTES))
    tail = min(tail_length, available - head)
    remaining = available - head - tail
    if remaining:
        extra_head = min(remaining, head_length - head)
        head += extra_head
        remaining -= extra_head
    if remaining:
        tail += min(remaining, tail_length - tail)
    return head, tail


def _render_preview(view: HeadTailView, budget_chars: int) -> tuple[str, int, int]:
    """Render a bounded head/tail preview whose marker accounts for all omissions."""
    budget = max(0, budget_chars)
    if view.omitted_bytes == 0:
        full = view.head
        if len(full) <= budget:
            return full, 0, 0
        split = len(full) * PREVIEW_HEAD_BYTES // (PREVIEW_HEAD_BYTES + PREVIEW_TAIL_BYTES)
        source_head = full[:split]
        source_tail = full[split:]
    else:
        source_head = view.head
        source_tail = view.tail

    marker = _omitted_marker(max(1, view.omitted_bytes), max(1, view.omitted_lines))
    head_keep = tail_keep = -1
    omitted_bytes = view.omitted_bytes
    omitted_lines = view.omitted_lines

    for _ in range(6):
        available = max(0, budget - len(marker))
        next_head_keep, next_tail_keep = _allocate_preview_chars(
            len(source_head),
            len(source_tail),
            available,
        )
        removed_head = source_head[next_head_keep:]
        removed_tail = source_tail[: len(source_tail) - next_tail_keep] if next_tail_keep else source_tail
        omitted_bytes = view.omitted_bytes + len(removed_head.encode("utf-8")) + len(removed_tail.encode("utf-8"))
        omitted_newlines = view.omitted_newlines + removed_head.count("\n") + removed_tail.count("\n")
        if removed_tail:
            omitted_ends_newline = removed_tail.endswith("\n")
        elif view.omitted_bytes:
            omitted_ends_newline = view.omitted_ends_newline
        else:
            omitted_ends_newline = removed_head.endswith("\n")
        omitted_lines = max(1, omitted_newlines + (0 if omitted_ends_newline else 1))
        next_marker = _omitted_marker(omitted_bytes, omitted_lines)
        if next_head_keep == head_keep and next_tail_keep == tail_keep and next_marker == marker:
            break
        head_keep, tail_keep, marker = next_head_keep, next_tail_keep, next_marker

    if len(marker) > budget:
        capped = cap_chars(
            source_head + source_tail,
            budget,
            marker=_truncation_marker(max(1, budget // 4)),
        )
        return capped.text, omitted_bytes, omitted_lines

    head_text = source_head[: max(0, head_keep)]
    tail_text = source_tail[-tail_keep:] if tail_keep > 0 else ""
    return head_text + marker + tail_text, omitted_bytes, omitted_lines


class TerminalOutputAccumulator:
    """Persist a complete stream and expose bounded model-facing deltas."""

    def __init__(
        self,
        spill_store: Optional[TerminalSpillStore],
        *,
        tool_name: str = "exec_command",
        backing_path: Optional[Path] = None,
        line_cap_bytes: int = LINE_CAP_BYTES,
        retain_full_delta: bool = False,
    ) -> None:
        self._spill = _SpillWriter(spill_store, tool_name=tool_name, backing_path=backing_path)
        self._preview = HeadTailBuffer(PREVIEW_HEAD_BYTES, PREVIEW_TAIL_BYTES)
        self._line_cap = _StreamingLineCap(line_cap_bytes)
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._full_delta = bytearray() if retain_full_delta else None
        self._delta_chars = 0
        self._finalized = False

    @property
    def spill_path(self) -> Optional[str]:
        path = self._spill.path
        return str(path) if path is not None else None

    @property
    def total_bytes(self) -> int:
        return self._spill.total_bytes

    def append_bytes(self, data: bytes) -> None:
        if not data:
            return
        text = self._decoder.decode(data, final=False)
        self._append_normalized(text)

    def append_text(self, text: str) -> None:
        if text:
            self._append_normalized(text)

    def _append_normalized(self, text: str) -> None:
        if not text:
            return
        normalized = text.encode("utf-8")
        self._spill.append(normalized)
        if self._full_delta is not None:
            self._full_delta.extend(normalized)
        self._preview.append(self._line_cap.append(text))
        self._delta_chars += len(text)

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._append_normalized(self._decoder.decode(b"", final=True))
        self._spill.finalize()

    def read_delta(self, max_output_tokens: int, *, hard_limit: bool = True) -> CappedOutput:
        if not hard_limit and self._full_delta is not None:
            text = bytes(self._full_delta).decode("utf-8")
            self._line_cap.take_delta_stats()
            result = CappedOutput(
                text=text,
                truncated=False,
                original_token_count=_estimated_tokens(len(text)),
                spill_path=self.spill_path,
                spill_bytes=self.total_bytes if self.spill_path else 0,
            )
            self._preview.reset()
            self._full_delta.clear()
            self._delta_chars = 0
            return result

        effective_tokens = clamp_output_tokens(max_output_tokens) if hard_limit else max(1, int(max_output_tokens))
        text, omitted_bytes, omitted_lines = _render_preview(
            self._preview.view(),
            effective_tokens * 4,
        )
        line_count, line_bytes = self._line_cap.take_delta_stats()
        spill_path = self.spill_path
        original_tokens = _estimated_tokens(self._delta_chars)
        result = CappedOutput(
            text=text,
            truncated=omitted_bytes > 0 or line_bytes > 0,
            original_token_count=original_tokens,
            spill_path=spill_path,
            spill_bytes=self.total_bytes if spill_path else 0,
            line_truncated_count=line_count,
            line_truncated_bytes=line_bytes,
            preview_omitted_bytes=omitted_bytes,
            preview_omitted_lines=omitted_lines,
        )
        self._preview.reset()
        if self._full_delta is not None:
            self._full_delta.clear()
        self._delta_chars = 0
        return result
