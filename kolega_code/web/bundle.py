"""Export a session as a self-contained, shareable replay bundle.

The output is static: a directory (or zip) of JSON and assets with no server and
no runtime dependency on this package. That is deliberate — the simplest way to
send someone a replay is a link to static files, which any host will serve and
which keeps working long after the machine that produced it is gone.

Everything written here has passed through :mod:`kolega_code.web.redaction`.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from kolega_code.cli import theme as cli_theme
from kolega_code.events import AgentEvent
from kolega_code.session.store import ArtifactStore

from .redaction import RedactionReport, redact_event, shareable_artifacts
from .theme_css import stylesheet

#: Bumped when the on-disk bundle layout changes so a player can refuse politely.
BUNDLE_FORMAT_VERSION = 1

_ASSET_DIR = Path(__file__).parent / "assets"
_PLAYER_ASSETS = ("player.html", "player.js", "player.css", "fold.js")


@dataclass
class BundleResult:
    """Where the bundle went and what was removed on the way."""

    path: Path
    event_count: int
    artifact_count: int
    duration_ms: int
    report: RedactionReport


async def export_bundle(
    events: Sequence[AgentEvent],
    destination: Path,
    *,
    session_id: str,
    title: str = "",
    theme_slug: Optional[str] = None,
    artifact_store: Optional[ArtifactStore] = None,
    extra_secrets: Optional[Iterable[str]] = None,
    as_zip: bool = False,
) -> BundleResult:
    """Write a replay bundle for ``events`` to ``destination``."""
    report = RedactionReport()
    safe_events = [redact_event(event, report=report, extra_secrets=extra_secrets) for event in events]

    target = destination.with_suffix("") if as_zip else destination
    target.mkdir(parents=True, exist_ok=True)

    _write_events(target / "events.jsonl", safe_events)
    artifact_count = await _write_artifacts(target / "artifacts", safe_events, artifact_store)
    _write_assets(target)
    (target / "theme.css").write_text(stylesheet(), encoding="utf-8")
    (target / "themes.json").write_text(
        json.dumps(cli_theme.all_web_tokens(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    duration_ms = max((event.elapsed_ms for event in safe_events), default=0)
    manifest = {
        "format": BUNDLE_FORMAT_VERSION,
        "session_id": session_id,
        "title": title,
        "event_count": len(safe_events),
        "artifact_count": artifact_count,
        "duration_ms": duration_ms,
        "started_at": safe_events[0].timestamp if safe_events else None,
        "ended_at": safe_events[-1].timestamp if safe_events else None,
        # Baked in so a shared replay opens in the theme it was recorded with,
        # while still letting the viewer switch.
        "theme": theme_slug or cli_theme.default_theme_slug(),
        "themes": list(cli_theme.all_web_tokens().keys()),
        "turns": _turn_index(safe_events),
        "redaction": {
            "artifacts_kept": len(report.artifacts_kept),
            "artifacts_dropped": report.artifacts_dropped,
            "strings_redacted": report.strings_redacted,
        },
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    final_path = target
    if as_zip:
        final_path = destination if destination.suffix == ".zip" else destination.with_suffix(".zip")
        _write_zip(target, final_path)
        shutil.rmtree(target, ignore_errors=True)

    return BundleResult(
        path=final_path,
        event_count=len(safe_events),
        artifact_count=artifact_count,
        duration_ms=duration_ms,
        report=report,
    )


def _write_events(path: Path, events: Sequence[AgentEvent]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")


async def _write_artifacts(
    directory: Path,
    events: Sequence[AgentEvent],
    artifact_store: Optional[ArtifactStore],
) -> int:
    refs = shareable_artifacts(events)
    if not refs or artifact_store is None:
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for digest, ref in refs.items():
        try:
            data = await artifact_store.open(ref)
        except Exception:
            # A missing artifact must not abort an export; the player renders the
            # preview text that the event still carries.
            continue
        (directory / digest).write_bytes(data)
        written += 1
    return written


def _write_assets(target: Path) -> None:
    for name in _PLAYER_ASSETS:
        source = _ASSET_DIR / name
        if source.exists():
            shutil.copyfile(source, target / name)
    # index.html is the entry point a static host serves by default.
    player = target / "player.html"
    if player.exists():
        shutil.copyfile(player, target / "index.html")


def _turn_index(events: Sequence[AgentEvent]) -> list[dict]:
    """Seek targets, so the player can offer "jump to turn" without folding first."""
    turns: list[dict] = []
    for event in events:
        if event.event_type == "turn_started":
            turns.append(
                {
                    "turn_id": str(event.content.get("turn_id") or ""),
                    "user_text": str(event.content.get("user_text") or "")[:200],
                    "elapsed_ms": event.elapsed_ms,
                    "seq": event.seq,
                }
            )
        elif event.event_type == "turn_ended" and turns:
            turns[-1]["status"] = str(event.content.get("status") or "completed")
            turns[-1]["ended_ms"] = event.elapsed_ms
    return turns


def _write_zip(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for item in sorted(source.rglob("*")):
            if item.is_file():
                bundle.write(item, item.relative_to(source).as_posix())
