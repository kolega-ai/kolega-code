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
import threading
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

from kolega_code.local_state import ensure_private_dir, ensure_private_file
from kolega_code.security import SECRET_PLACEHOLDER as SECRET_PLACEHOLDER, redact_secrets

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
            line = scrub_secrets(json.dumps(payload, default=str), self._secret_values)
            with self._lock:
                self._rotate_if_large()
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            ensure_private_file(self.path)
        except (OSError, TypeError, ValueError):
            pass

    def write_sidecar(self, name: str, content: str) -> Optional[Path]:
        """Append a scrubbed text sidecar (e.g. a stack dump) into the diagnostics dir."""
        if not self.enabled:
            return None
        target = self.dir / name
        try:
            with self._lock:
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(scrub_secrets(content, self._secret_values))
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

    The Textual app calls :meth:`beat` on a ~1 s loop timer. If beats stop (the loop is
    blocked by synchronous work), this daemon thread notices within ``check_interval`` and
    records an ``event_loop_stalled`` line plus a full thread-stack dump — the main
    thread's frame is exactly what is blocking the loop. When beats resume it records
    ``event_loop_recovered``. (A blocked loop can't run the on-loop timer, so the
    staleness *is* the signal; this thread keeps running regardless.)
    """

    def __init__(
        self,
        diag: DiagnosticsLog,
        *,
        stall_seconds: float = 5.0,
        check_interval: float = 1.0,
    ) -> None:
        self._diag = diag
        self._stall = stall_seconds
        self._interval = check_interval
        self._last_beat = time.monotonic()
        self._beat_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._stalled_since: Optional[float] = None
        self._sig_file = None

    def beat(self) -> None:
        with self._beat_lock:
            self._last_beat = time.monotonic()

    def start(self) -> None:
        if not self._diag.enabled or self._thread is not None:
            return
        self._enable_faulthandler()
        self._thread = threading.Thread(target=self._run, name="kolega-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            with self._beat_lock:
                gap = time.monotonic() - self._last_beat
            if gap > self._stall and self._stalled_since is None:
                self._stalled_since = time.monotonic()
                stacks = format_all_thread_stacks()
                self._diag.record("event_loop_stalled", stalled_for_s=round(gap, 1), stacks=stacks)
                self._diag.write_sidecar("stalls.log", f"# stall at {_now_iso()} (gap {gap:.1f}s)\n{stacks}\n")
            elif gap <= self._stall and self._stalled_since is not None:
                self._diag.record(
                    "event_loop_recovered", stalled_for_s=round(time.monotonic() - self._stalled_since, 1)
                )
                self._stalled_since = None

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
    session_export: Optional["SessionBugExport"],
) -> Optional[Path]:
    """Build a shareable zip: summary + diagnostics timeline + crash/stall dumps + the
    session projection, canonical events, and artifact manifest. Oversized artifact bodies
    are not included. Secrets are scrubbed throughout. Returns the
    zip path, or None if diagnostics are disabled / on error.

    Entries are written flat at the archive root (same layout the old bundle directory
    had). Diagnostics artifacts are already scrubbed at write time, so they are added
    verbatim; generated session exports are scrubbed here."""
    if not diag.enabled:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    zip_path = diag.dir / f"bug-{stamp}.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("summary.md", scrub_secrets(summary, diag._secret_values))
            # Diagnostics artifacts already scrubbed at write time; add as-is.
            for artifact in list(diag.dir.glob("session-*.jsonl")) + list(diag.dir.glob("*.log")):
                try:
                    zf.write(artifact, artifact.name)
                except OSError:
                    pass
            if session_export is not None:
                zf.writestr("session.json", scrub_secrets(session_export.session_json, diag._secret_values))
                zf.writestr(
                    "session-events.jsonl",
                    scrub_secrets(session_export.events_jsonl, diag._secret_values),
                )
                zf.writestr(
                    "session-artifacts.json",
                    scrub_secrets(session_export.artifact_manifest_json, diag._secret_values),
                )
        ensure_private_file(zip_path)
        return zip_path
    except OSError:
        return None
