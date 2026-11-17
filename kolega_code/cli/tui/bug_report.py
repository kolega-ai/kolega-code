from __future__ import annotations

import asyncio
import threading
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..diagnostics import DiagnosticsLog, assemble_bug_bundle
from ..session_store import SessionStore
from ..updater import current_version

if TYPE_CHECKING:
    from textual.worker import Worker

    from .state import ConversationEntry


def diagnostics_summary_lines(
    header: dict[str, Any],
    diag: DiagnosticsLog | None,
    *,
    cancel_event: threading.Event | None = None,
) -> list[str]:
    lines = [
        f"Kolega Code {header.get('kolega_version') or current_version()}",
        f"Platform: {header.get('platform', '?')}  |  terminal: {header.get('term_program') or header.get('term') or '?'}",
        f"Python: {header.get('python', '?')}",
    ]
    if header.get("provider"):
        lines.append(f"Model: {header['provider']}/{header.get('model')} (effort: {header.get('thinking_effort')})")
    gigacode = "on" if header.get("gigacode_enabled") else "off"
    lines.append(
        f"Permission: {header.get('permission_mode')}  |  "
        f"mode: {header.get('interaction_mode')}  |  gigacode: {gigacode}"
    )
    if header.get("providers_with_keys"):
        lines.append(f"Providers with keys: {', '.join(header['providers_with_keys'])}")
    if diag is not None and diag.enabled:
        lines.append(f"Diagnostics log: {diag.path}")
        stalls = errors = 0
        try:
            if diag.path.exists():
                with diag.path.open(encoding="utf-8") as source:
                    for line in source:
                        if cancel_event is not None and cancel_event.is_set():
                            raise CancelledError
                        stalls += line.count('"kind": "event_loop_stalled"')
                        errors += line.count('"kind": "llm_error"')
            lines.append(f"This session: {stalls} loop stall(s), {errors} LLM error(s) recorded")
        except OSError:
            pass
    else:
        lines.append("Diagnostics: disabled (KOLEGA_CODE_NO_DIAGNOSTICS)")
    return lines


@dataclass(eq=False)
class BugReportJob:
    session_id: str
    session_name: str
    store: SessionStore
    diag: DiagnosticsLog
    header: dict[str, Any]
    entry: ConversationEntry
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: asyncio.Future[Path | None] | None = None
    worker: Worker[None] | None = None

    def prepare(self, progress: Callable[[str], None]) -> Path | None:
        try:
            if self.cancel_event.is_set():
                raise CancelledError
            summary = "\n".join(diagnostics_summary_lines(self.header, self.diag, cancel_event=self.cancel_event))
            progress("Preparing bug report: reading session…")
            export = self.store.bug_export(self.session_id, cancel_event=self.cancel_event)
            if self.cancel_event.is_set():
                raise CancelledError
            progress("Preparing bug report: redacting and writing archive…")
            bundle = assemble_bug_bundle(
                self.diag, summary=summary, session_export=export, cancel_event=self.cancel_event
            )
            if bundle is None:
                self.diag.record("bug_export_failed", session_id=self.session_id, error_type="BundleWriteError")
            return bundle
        except CancelledError:
            raise
        except Exception as exc:
            self.diag.record(
                "bug_export_failed", session_id=self.session_id, error_type=type(exc).__name__, message=str(exc)
            )
            return None

    async def wait(self) -> Path | None:
        assert self.future is not None
        # Executor futures survive runner-wide Task cancellation; a to_thread Task does not.
        while not self.future.done():
            try:
                await asyncio.shield(self.future)
            except asyncio.CancelledError:
                self.cancel_event.set()
        return self.future.result()
