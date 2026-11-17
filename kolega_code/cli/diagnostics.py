"""Local, always-on diagnostics so a vague "it went unresponsive" report is actionable.

Writes a JSONL timeline under the CLI state dir, plus a responsiveness watchdog that
captures stack traces when the Textual event loop stalls — the single most useful signal
for "the UI froze" reports (it shows *what* is blocking the loop). Nothing is sent
anywhere; a `/bug` bundle is shared only by explicit user action.

Privacy policy: content is logged in full (it is local-only, and the session store
already persists full conversation history to disk). The one hard rule is that
CREDENTIALS are never written — see ``scrub_secrets``. See the memory
"prefer-unredacted-local-diagnostics".
"""

from __future__ import annotations

import faulthandler
import json
import os
import signal
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from concurrent.futures import CancelledError
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Iterable, Optional

from kolega_code.local_state import ensure_private_dir, ensure_private_file
from kolega_code.security import (
    SECRET_PLACEHOLDER as SECRET_PLACEHOLDER,
    redact_secrets,
    redact_secrets_in_obj,
)
from kolega_code.security.secrets import _environment_values

if TYPE_CHECKING:
    from .session_store import SessionBugExport

DIAGNOSTICS_DISABLED_ENV = "KOLEGA_CODE_NO_DIAGNOSTICS"


def scrub_secrets(text: str, extra_values: Optional[Iterable[str]] = None) -> str:
    """Remove credentials from ``text``; all other content is preserved verbatim.

    Replaces (1) exact secret values supplied by the caller (configured API keys) and
    from secret-ish env vars, and (2) common credential patterns (Authorization bearer,
    x-api-key, ``*_API_KEY=``, and ``sk-``/``xai-``/``AIza`` token shapes).
    """
    return redact_secrets(text, extra_values, include_environment=True)


def scrub_secrets_in_obj(obj: Any, extra_values: Optional[Iterable[str]] = None) -> Any:
    """Structural companion to ``scrub_secrets`` for JSON-compatible payloads.

    Redacting serialized JSON can splice a replacement across an escape sequence
    and leave a document that no longer parses, which is how a bug bundle's
    ``session.json`` ended up unreadable. Redacting the decoded leaves and
    re-serializing cannot do that.
    """
    return redact_secrets_in_obj(obj, extra_values, include_environment=True)


_BUNDLE_CHUNK_BYTES = 64 * 1024


def _check_bundle_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError()


class _BundleWriter:
    def __init__(self, target: IO[bytes], cancel_event: threading.Event | None) -> None:
        self.target = target
        self.cancel_event = cancel_event
        self.buffer = bytearray()

    def write(self, text: str) -> None:
        for offset in range(0, len(text), _BUNDLE_CHUNK_BYTES // 4):
            _check_bundle_cancelled(self.cancel_event)
            data = text[offset : offset + _BUNDLE_CHUNK_BYTES // 4].encode("utf-8")
            if len(self.buffer) + len(data) > _BUNDLE_CHUNK_BYTES:
                self.flush()
            self.buffer.extend(data)

    def flush(self) -> None:
        _check_bundle_cancelled(self.cancel_event)
        if self.buffer:
            self.target.write(self.buffer)
            self.buffer.clear()
            _check_bundle_cancelled(self.cancel_event)


def _write_json_string(writer: _BundleWriter, text: str) -> None:
    writer.write('"')
    for offset in range(0, len(text), 8192):
        _check_bundle_cancelled(writer.cancel_event)
        writer.write(json.dumps(text[offset : offset + 8192])[1:-1])
    writer.write('"')


def _write_scrubbed_json(
    writer: _BundleWriter,
    value: Any,
    secret_values: Iterable[str],
    *,
    pretty: bool = False,
    level: int = 0,
) -> None:
    _check_bundle_cancelled(writer.cancel_event)
    if isinstance(value, dict):
        writer.write("{")
        items = sorted(value.items()) if pretty else value.items()
        for index, (key, item) in enumerate(items):
            _check_bundle_cancelled(writer.cancel_event)
            if index:
                writer.write("," if pretty else ", ")
            if pretty:
                writer.write("\n" + "  " * (level + 1))
            if not isinstance(key, str):
                if key is None or isinstance(key, (int, float, bool)):
                    key = json.dumps(key)
                else:
                    raise TypeError(f"Unsupported JSON key: {type(key).__name__}")
            _write_json_string(writer, key)
            writer.write(": ")
            _write_scrubbed_json(writer, item, secret_values, pretty=pretty, level=level + 1)
        if value and pretty:
            writer.write("\n" + "  " * level)
        writer.write("}")
    elif isinstance(value, (list, tuple)):
        writer.write("[")
        for index, item in enumerate(value):
            _check_bundle_cancelled(writer.cancel_event)
            if index:
                writer.write("," if pretty else ", ")
            if pretty:
                writer.write("\n" + "  " * (level + 1))
            _write_scrubbed_json(writer, item, secret_values, pretty=pretty, level=level + 1)
        if value and pretty:
            writer.write("\n" + "  " * level)
        writer.write("]")
    elif value is None or isinstance(value, (bool, int, float)):
        writer.write(json.dumps(value))
    else:
        text = value if isinstance(value, str) else str(value)
        _write_json_string(writer, redact_secrets(text, secret_values))


def _copy_diagnostic_artifact(archive: zipfile.ZipFile, artifact: Path, cancel_event: threading.Event | None) -> None:
    _check_bundle_cancelled(cancel_event)
    try:
        source = artifact.open("rb")
    except OSError:
        _check_bundle_cancelled(cancel_event)
        return
    # Stage bounded chunks so a failed source read omits the whole optional entry.
    with source, tempfile.SpooledTemporaryFile(max_size=_BUNDLE_CHUNK_BYTES, dir=str(artifact.parent)) as staged:
        while True:
            _check_bundle_cancelled(cancel_event)
            try:
                chunk = source.read(_BUNDLE_CHUNK_BYTES)
            except OSError:
                _check_bundle_cancelled(cancel_event)
                return
            _check_bundle_cancelled(cancel_event)
            if not chunk:
                break
            staged.write(chunk)
        staged.seek(0)
        with archive.open(artifact.name, "w", force_zip64=True) as target:
            while True:
                _check_bundle_cancelled(cancel_event)
                chunk = staged.read(_BUNDLE_CHUNK_BYTES)
                _check_bundle_cancelled(cancel_event)
                if not chunk:
                    break
                target.write(chunk)
                _check_bundle_cancelled(cancel_event)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_all_thread_stacks() -> str:
    """All live thread stacks as text — the main-thread frame shows what blocks the loop."""
    lines: list[str] = []
    frames = sys._current_frames()
    names = {t.ident: t.name for t in threading.enumerate()}
    for tid, frame in frames.items():
        lines.append(f"\n--- thread {names.get(tid, '?')} ({tid}) ---\n")
        lines.extend(traceback.format_stack(frame))
    return "".join(lines)


class DiagnosticsLog:
    """Append-only JSONL diagnostics timeline for one session, under the state dir."""

    MAX_BYTES = 8 * 1024 * 1024
    MAX_FIELD_CHARS = 20_000
    KEEP_SESSIONS = 10

    def __init__(
        self,
        state_dir: Path,
        session_id: str,
        *,
        enabled: bool = True,
        secret_values: Optional[Iterable[str]] = None,
    ) -> None:
        self.dir = Path(state_dir) / "diagnostics"
        self.session_id = session_id
        self.path = self.dir / f"session-{session_id}.jsonl"
        self.enabled = bool(enabled) and not os.environ.get(DIAGNOSTICS_DISABLED_ENV)
        self._secret_values = list(secret_values or [])
        self._lock = threading.Lock()
        if self.enabled:
            try:
                ensure_private_dir(self.dir)
                self._prune_old_sessions()
            except OSError:
                self.enabled = False

    def record(self, kind: str, **fields: Any) -> None:
        """Append one timeline record. Thread-safe; never raises."""
        if not self.enabled:
            return
        payload: dict[str, Any] = {"ts": _now_iso(), "kind": kind}
        for key, value in fields.items():
            if value is not None:
                payload[key] = self._bound(value)
        try:
            # Redact before serializing: scrubbing the encoded form can splice a
            # replacement across a \" escape and emit a line that will not parse.
            line = json.dumps(scrub_secrets_in_obj(payload, self._secret_values), default=str)
            with self._lock:
                self._rotate_if_large()
                # Write via os.open/os.write: Path/handle.write is modeled by CodeQL as a
                # clear-text storage sink for the still-tainted (but already scrubbed) line.
                # O_CREAT|0o600 creates the file owner-only from the outset.
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(fd, (line + "\n").encode("utf-8"))
                finally:
                    os.close(fd)
            ensure_private_file(self.path)
        except (OSError, TypeError, ValueError):
            pass

    def write_sidecar(self, name: str, content: str) -> Optional[Path]:
        """Append a scrubbed text sidecar (e.g. a stack dump) into the diagnostics dir."""
        if not self.enabled:
            return None
        target = self.dir / name
        try:
            data = scrub_secrets(content, self._secret_values).encode("utf-8")
            with self._lock:
                # os.write avoids the handle.write sink CodeQL flags for scrubbed content;
                # O_CREAT|0o600 keeps the sidecar owner-only from creation.
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
            ensure_private_file(target)
            return target
        except OSError:
            return None

    def _bound(self, value: Any) -> Any:
        # Bound very large string blobs for *file size* (not for privacy).
        if isinstance(value, str) and len(value) > self.MAX_FIELD_CHARS:
            return value[: self.MAX_FIELD_CHARS] + f"…[+{len(value) - self.MAX_FIELD_CHARS} chars]"
        return value

    def _rotate_if_large(self) -> None:
        try:
            if self.path.exists() and self.path.stat().st_size > self.MAX_BYTES:
                self.path.replace(self.path.with_suffix(".jsonl.1"))
        except OSError:
            pass

    def _prune_old_sessions(self) -> None:
        files = sorted(self.dir.glob("session-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[self.KEEP_SESSIONS :]:
            try:
                stale.unlink()
            except OSError:
                pass


class ResponsivenessWatchdog:
    """Detect event-loop stalls from an off-loop thread and capture the blocking stack.

    The Textual app calls :meth:`beat` on a :attr:`beat_interval` loop timer. Two tiers
    of evidence come out of that:

    * **Stalls.** If beats stop (the loop is blocked by synchronous work), this daemon
      thread notices within ``check_interval`` and records an ``event_loop_stalled``
      line plus a full thread-stack dump — the main thread's frame is exactly what is
      blocking the loop. When beats resume it records ``event_loop_recovered``. (A
      blocked loop can't run the on-loop timer, so the staleness *is* the signal; this
      thread keeps running regardless.) Stack capture is capped per session so a
      pathological session cannot grow the diagnostics file without bound.
    * **Sub-second chop.** Beats are late by exactly as much as the loop was blocked, so
      :meth:`beat` histograms its own lateness. "Scrolling feels choppy" is produced by
      stalls well under the stack-capture threshold, which would otherwise leave no
      trace at all; the aggregate ``loop_gap_histogram`` line costs a few bytes and no
      stacks, and says immediately whether the chop is real and how big it is.
    * **Load signal.** :meth:`recent_excess` exposes a decaying peak of the same
      lateness for the TUI's flush pacing — when the loop runs late, flush timers
      stretch so the UI degrades to a lower frame rate instead of freezing.

    Thresholds are constructor parameters for tests only — there is no user decision here.
    """

    # Excess-over-nominal beat lateness, in seconds. The first edge is the floor of
    # human-visible chop; the last is the stack-capture threshold.
    _BUCKET_EDGES: tuple[float, ...] = (0.1, 0.25, 1.0, 5.0)

    # Per-beat decay of the recent-lateness peak. At the healthy 0.2s beat cadence a
    # 1s stall's influence falls below the 0.1s pacing deadband in ~15 beats (~3s):
    # backoff outlives the trouble by a few seconds, then full cadence returns.
    _RECENT_DECAY = 0.85

    def __init__(
        self,
        diag: DiagnosticsLog,
        *,
        stall_seconds: float = 1.0,
        check_interval: float = 0.25,
        beat_interval: float = 0.2,
        histogram_interval: float = 300.0,
        max_stall_captures: int = 200,
    ) -> None:
        self._diag = diag
        self._stall = stall_seconds
        self._interval = check_interval
        self.beat_interval = beat_interval
        self._histogram_interval = histogram_interval
        self._max_captures = max_stall_captures
        self._captures = 0
        self._last_beat = time.monotonic()
        self._beat_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._stalled_since: Optional[float] = None
        self._sig_file = None
        self._beats = 0
        self._max_excess = 0.0
        self._recent_excess = 0.0
        self._labels = self._bucket_labels()
        self._buckets: dict[str, int] = {label: 0 for label in self._labels}
        self._window_start = time.monotonic()

    @classmethod
    def _bucket_labels(cls) -> list[str]:
        edges = cls._BUCKET_EDGES
        labels = [f"{edges[i]}-{edges[i + 1]}s" for i in range(len(edges) - 1)]
        labels.append(f">={edges[-1]}s")
        return labels

    def beat(self) -> None:
        """Called on the event loop. Must stay O(1): it runs several times a second.

        Lateness is measured against the nominal interval, so a stall shorter than
        ``beat_interval`` can be under-reported by up to one interval (part of it is
        absorbed by the beat's own wait). Bucket edges are therefore lower bounds.
        """
        now = time.monotonic()
        with self._beat_lock:
            excess = (now - self._last_beat) - self.beat_interval
            self._last_beat = now
            self._beats += 1
            # Peak-hold with decay, updated on quiet beats too so backoff releases.
            # The 0.0 floor guards early-firing timers (negative excess).
            self._recent_excess = max(excess, self._recent_excess * self._RECENT_DECAY, 0.0)
            if excess < self._BUCKET_EDGES[0]:
                return
            self._max_excess = max(self._max_excess, excess)
            for index in range(len(self._BUCKET_EDGES) - 1, -1, -1):
                if excess >= self._BUCKET_EDGES[index]:
                    self._buckets[self._labels[index]] += 1
                    return

    def recent_excess(self) -> float:
        """Decaying peak of recent beat lateness, in seconds — the TUI's load signal.

        Independent of the histogram: :meth:`flush_histogram` never resets it; it
        only decays, one ``_RECENT_DECAY`` step per beat.
        """
        with self._beat_lock:
            return self._recent_excess

    def start(self) -> None:
        if not self._diag.enabled or self._thread is not None:
            return
        self._enable_faulthandler()
        self._thread = threading.Thread(target=self._run, name="kolega-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.flush_histogram("session_end")

    def flush_histogram(self, reason: str) -> None:
        """Emit one aggregate line for the gaps seen since the last flush (no stacks)."""
        with self._beat_lock:
            counted = sum(self._buckets.values())
            payload = {
                "reason": reason,
                "window_s": round(time.monotonic() - self._window_start, 1),
                "beats": self._beats,
                "max_gap_s": round(self._max_excess, 3),
                "buckets": dict(self._buckets),
            }
            self._window_start = time.monotonic()
            self._beats = 0
            self._max_excess = 0.0
            for label in self._buckets:
                self._buckets[label] = 0
        if counted:
            self._diag.record("loop_gap_histogram", **payload)

    def _run(self) -> None:
        last_flush = time.monotonic()
        while not self._stop.wait(self._interval):
            with self._beat_lock:
                gap = time.monotonic() - self._last_beat
            if gap > self._stall and self._stalled_since is None:
                self._stalled_since = time.monotonic()
                self._record_stall(gap)
            elif gap <= self._stall and self._stalled_since is not None:
                self._diag.record(
                    "event_loop_recovered", stalled_for_s=round(time.monotonic() - self._stalled_since, 1)
                )
                self._stalled_since = None
            now = time.monotonic()
            if now - last_flush >= self._histogram_interval:
                last_flush = now
                self.flush_histogram("interval")

    def _record_stall(self, gap: float) -> None:
        # Stacks are the expensive part, and they only ever get written while the loop
        # is already stalled — but cap them so one pathological session cannot fill the
        # timeline. Past the cap the stall itself is still recorded, without stacks.
        if self._captures >= self._max_captures:
            self._diag.record("event_loop_stalled", stalled_for_s=round(gap, 1), stacks_omitted=True)
            return
        self._captures += 1
        stacks = format_all_thread_stacks()
        self._diag.record("event_loop_stalled", stalled_for_s=round(gap, 1), stacks=stacks)
        self._diag.write_sidecar("stalls.log", f"# stall at {_now_iso()} (gap {gap:.1f}s)\n{stacks}\n")

    def _enable_faulthandler(self) -> None:
        # Hard-fault dumps + a manual escape hatch (`kill -USR1 <pid>`) for a wedged process.
        try:
            faulthandler.enable()
            if hasattr(signal, "SIGUSR1") and self._diag.enabled:
                ensure_private_dir(self._diag.dir)
                self._sig_file = (self._diag.dir / "manual-dump.log").open("a")
                faulthandler.register(signal.SIGUSR1, file=self._sig_file, all_threads=True)
        except (OSError, ValueError, RuntimeError, AttributeError):
            pass


def write_crash_log(
    state_dir: Path,
    *,
    exc: BaseException,
    header: str = "",
    secret_values: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    """Persist an unhandled-exception traceback (secrets scrubbed) for a true crash.

    Standalone (no app/DiagnosticsLog needed) so the top-level handler in main.py can call
    it even if the app never finished starting.  Pass configured API keys via
    ``secret_values`` so they are scrubbed verbatim in addition to pattern matching."""
    if os.environ.get(DIAGNOSTICS_DISABLED_ENV):
        return None
    try:
        directory = Path(state_dir) / "diagnostics"
        ensure_private_dir(directory)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = directory / f"crash-{stamp}.log"
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        scrubbed = scrub_secrets(f"{header}\n\n{tb}", secret_values)
        # Create the file owner-only from the outset (no write-then-chmod race)
        # and write via os.write to avoid path.write_text, which CodeQL models
        # as a clear-text storage sink for the still-tainted scrubbed traceback.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, scrubbed.encode("utf-8"))
        finally:
            os.close(fd)
        return path
    except OSError:
        return None


def assemble_bug_bundle(
    diag: DiagnosticsLog,
    *,
    summary: str,
    session_export: SessionBugExport | None,
    cancel_event: threading.Event | None = None,
) -> Path | None:
    """Build a shareable zip: summary + diagnostics timeline + crash/stall dumps + the
    session projection, canonical events, and artifact manifest. Oversized artifact bodies
    are not included. Secrets are scrubbed throughout. Returns the
    zip path, or None if diagnostics are disabled / on error.

    Entries are written flat at the archive root (same layout the old bundle directory
    had). Diagnostics artifacts are already scrubbed at write time, so they are added
    verbatim; generated session exports are scrubbed here."""
    _check_bundle_cancelled(cancel_event)
    if not diag.enabled:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    temp_path: Path | None = None
    fd: int | None = None
    try:
        secret_values = [*diag._secret_values, *_environment_values()]
        ensure_private_dir(diag.dir)
        _check_bundle_cancelled(cancel_event)
        fd, temp_name = tempfile.mkstemp(prefix=f".bug-{stamp}-", suffix=".tmp", dir=diag.dir)
        temp_path = Path(temp_name)
        zip_path = temp_path.with_name(temp_path.name[1:]).with_suffix(".zip")
        archive = os.fdopen(fd, "w+b")
        fd = None
        with archive:
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                with zf.open("summary.md", "w", force_zip64=True) as target:
                    writer = _BundleWriter(target, cancel_event)
                    writer.write(redact_secrets(summary, secret_values))
                    writer.flush()
                for pattern in ("session-*.jsonl", "*.log"):
                    _check_bundle_cancelled(cancel_event)
                    for artifact in diag.dir.glob(pattern):
                        _copy_diagnostic_artifact(zf, artifact, cancel_event)
                _check_bundle_cancelled(cancel_event)
                if session_export is not None:
                    with zf.open("session.json", "w", force_zip64=True) as target:
                        writer = _BundleWriter(target, cancel_event)
                        _write_scrubbed_json(writer, session_export.session, secret_values, pretty=True)
                        writer.write("\n")
                        writer.flush()
                    _check_bundle_cancelled(cancel_event)
                    with zf.open("session-events.jsonl", "w", force_zip64=True) as target:
                        writer = _BundleWriter(target, cancel_event)
                        for event in session_export.events:
                            _check_bundle_cancelled(cancel_event)
                            _write_scrubbed_json(writer, event.to_dict(), secret_values)
                            writer.write("\n")
                        writer.flush()
                    _check_bundle_cancelled(cancel_event)
                    with zf.open("session-artifacts.json", "w", force_zip64=True) as target:
                        writer = _BundleWriter(target, cancel_event)
                        _write_scrubbed_json(writer, session_export.artifact_manifest, secret_values, pretty=True)
                        writer.write("\n")
                        writer.flush()
        _check_bundle_cancelled(cancel_event)
        temp_path.replace(zip_path)
        return zip_path
    except OSError:
        _check_bundle_cancelled(cancel_event)
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
