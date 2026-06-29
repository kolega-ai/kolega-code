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
import re
import shutil
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from kolega_code.local_state import ensure_private_dir, ensure_private_file

DIAGNOSTICS_DISABLED_ENV = "KOLEGA_CODE_NO_DIAGNOSTICS"
SECRET_PLACEHOLDER = "‹secret›"  # ‹secret›

# Whole-token credential shapes (replaced entirely).
_TOKEN_PATTERNS = (
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bxai-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{10,}"),  # Google API keys
)
# Prefix=value shapes (keep the label, replace the value).
_PREFIX_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)\S+"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)\S+"),
    re.compile(r"(?im)^([A-Za-z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)[A-Za-z0-9_]*\s*[=:]\s*)\S+"),
)


def _env_secret_values() -> list[str]:
    """Exact secret values from the environment, to redact verbatim wherever they appear."""
    values: list[str] = []
    for name, val in os.environ.items():
        if not val or len(val) < 8:
            continue
        upper = name.upper()
        if any(tok in upper for tok in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            values.append(val)
    return values


def scrub_secrets(text: str, extra_values: Optional[Iterable[str]] = None) -> str:
    """Remove credentials from ``text``; all other content is preserved verbatim.

    Replaces (1) exact secret values supplied by the caller (configured API keys) and
    from secret-ish env vars, and (2) common credential patterns (Authorization bearer,
    x-api-key, ``*_API_KEY=``, and ``sk-``/``xai-``/``AIza`` token shapes).
    """
    if not text:
        return text
    for value in list(extra_values or []) + _env_secret_values():
        if value and len(value) >= 8:
            text = text.replace(value, SECRET_PLACEHOLDER)
    for pattern in _PREFIX_PATTERNS:
        text = pattern.sub(lambda m: m.group(1) + SECRET_PLACEHOLDER, text)
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub(SECRET_PLACEHOLDER, text)
    return text


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


def write_crash_log(state_dir: Path, *, exc: BaseException, header: str = "") -> Optional[Path]:
    """Persist an unhandled-exception traceback (secrets scrubbed) for a true crash.

    Standalone (no app/DiagnosticsLog needed) so the top-level handler in main.py can call
    it even if the app never finished starting."""
    if os.environ.get(DIAGNOSTICS_DISABLED_ENV):
        return None
    try:
        directory = Path(state_dir) / "diagnostics"
        ensure_private_dir(directory)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = directory / f"crash-{stamp}.log"
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        path.write_text(scrub_secrets(f"{header}\n\n{tb}"), encoding="utf-8")
        ensure_private_file(path)
        return path
    except OSError:
        return None


def assemble_bug_bundle(diag: DiagnosticsLog, *, summary: str, session_json: Optional[Path]) -> Optional[Path]:
    """Build a shareable bundle dir: summary + diagnostics timeline + crash/stall dumps +
    the (full, unredacted-content) session JSON. Secrets are scrubbed throughout. Returns
    the bundle directory path, or None if diagnostics are disabled / on error."""
    if not diag.enabled:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = diag.dir / f"bug-{stamp}"
    try:
        ensure_private_dir(bundle)
        (bundle / "summary.md").write_text(scrub_secrets(summary, diag._secret_values), encoding="utf-8")
        # Diagnostics artifacts already scrubbed at write time; copy as-is.
        for artifact in list(diag.dir.glob("session-*.jsonl")) + list(diag.dir.glob("*.log")):
            try:
                shutil.copy2(artifact, bundle / artifact.name)
            except OSError:
                pass
        if session_json and session_json.exists():
            try:
                scrubbed = scrub_secrets(session_json.read_text(encoding="utf-8"), diag._secret_values)
                (bundle / "session.json").write_text(scrubbed, encoding="utf-8")
            except OSError:
                pass
        return bundle  # the bundle dir is already owner-only (0700) via ensure_private_dir
    except OSError:
        return None
