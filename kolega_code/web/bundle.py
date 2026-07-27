"""Export a session as a self-contained, shareable replay.

The output is static, with no server and no runtime dependency on this package,
so a replay keeps working long after the machine that produced it is gone.

Two shapes, for two different jobs:

* **One HTML file** (the default). Everything is embedded, so it opens with a
  double-click and can be emailed, dropped in a chat, or attached to a bug
  report. See :mod:`kolega_code.web.singlefile` for why a folder cannot.
* **A directory or zip** (``--dir`` / ``--zip``), for publishing a replay on a
  static host where separate, cacheable files are the better shape.

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
from kolega_code.events import AgentEvent, ArtifactRef
from kolega_code.session.store import ArtifactStore

from .redaction import RedactionReport, redact_event, shareable_artifacts
from .singlefile import build_single_file
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
    #: True when the whole replay is one HTML document rather than a directory.
    single_file: bool = False


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
    single_file: bool = False,
) -> BundleResult:
    """Write a replay for ``events`` to ``destination``.

    With ``single_file`` the destination is one HTML document; otherwise it is a
    directory, or a zip of that directory when ``as_zip`` is set.
    """
    report = RedactionReport()
    safe_events = [redact_event(event, report=report, extra_secrets=extra_secrets) for event in events]
    loaded_artifacts = await _load_artifacts(safe_events, artifact_store)
    events_jsonl = _events_jsonl(safe_events)
    duration_ms = max((event.elapsed_ms for event in safe_events), default=0)

    if single_file:
        return _write_single_file(
            destination,
            safe_events,
            events_jsonl=events_jsonl,
            artifacts=loaded_artifacts,
            session_id=session_id,
            title=title,
            theme_slug=theme_slug,
            duration_ms=duration_ms,
            report=report,
        )

    target = destination.with_suffix("") if as_zip else destination
    target.mkdir(parents=True, exist_ok=True)

    (target / "events.jsonl").write_bytes(events_jsonl)
    artifact_count = _write_artifacts(target / "artifacts", loaded_artifacts)
    _write_assets(target)
    (target / "theme.css").write_text(stylesheet(), encoding="utf-8")
    (target / "themes.json").write_text(
        json.dumps(cli_theme.all_web_tokens(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = _manifest(
        safe_events,
        session_id=session_id,
        title=title,
        theme_slug=theme_slug,
        duration_ms=duration_ms,
        artifact_count=artifact_count,
        report=report,
    )
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


def _write_single_file(
    destination: Path,
    events: Sequence[AgentEvent],
    *,
    events_jsonl: bytes,
    artifacts: dict[str, tuple[ArtifactRef, bytes]],
    session_id: str,
    title: str,
    theme_slug: Optional[str],
    duration_ms: int,
    report: RedactionReport,
) -> BundleResult:
    """Write the whole replay as one HTML document."""
    # Only images earn their place inline. A text artifact is already described
    # by the preview its event carries, and embedding it would just inflate the
    # file with bytes nothing renders.
    inlinable = {
        digest: (ref.media_type, data)
        for digest, (ref, data) in artifacts.items()
        if str(ref.media_type or "").startswith("image/")
    }
    manifest = _manifest(
        events,
        session_id=session_id,
        title=title,
        theme_slug=theme_slug,
        duration_ms=duration_ms,
        artifact_count=len(inlinable),
        report=report,
    )
    # Tells the player its artifacts are either inline or unreachable, so it
    # never falls back to a sibling path that cannot exist in a single file.
    manifest["artifacts_inline"] = True

    path = destination if destination.suffix.lower() == ".html" else destination.with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_single_file(
            manifest=manifest,
            events_jsonl=events_jsonl,
            theme_css=stylesheet(),
            artifacts=inlinable,
        ),
        encoding="utf-8",
    )
    return BundleResult(
        path=path,
        event_count=len(events),
        artifact_count=len(inlinable),
        duration_ms=duration_ms,
        report=report,
        single_file=True,
    )


def _manifest(
    events: Sequence[AgentEvent],
    *,
    session_id: str,
    title: str,
    theme_slug: Optional[str],
    duration_ms: int,
    artifact_count: int,
    report: RedactionReport,
) -> dict:
    return {
        "format": BUNDLE_FORMAT_VERSION,
        "session_id": session_id,
        "title": title,
        "event_count": len(events),
        "artifact_count": artifact_count,
        "duration_ms": duration_ms,
        "started_at": events[0].timestamp if events else None,
        "ended_at": events[-1].timestamp if events else None,
        # Baked in so a shared replay opens in the theme it was recorded with,
        # while still letting the viewer switch.
        "theme": theme_slug or cli_theme.default_theme_slug(),
        "themes": list(cli_theme.all_web_tokens().keys()),
        "turns": _turn_index(events),
        "redaction": {
            "artifacts_kept": len(report.artifacts_kept),
            "artifacts_dropped": report.artifacts_dropped,
            "strings_redacted": report.strings_redacted,
        },
    }


def _events_jsonl(events: Sequence[AgentEvent]) -> bytes:
    lines = [json.dumps(event.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False) for event in events]
    return ("\n".join(lines) + "\n" if lines else "").encode("utf-8")


async def _load_artifacts(
    events: Sequence[AgentEvent],
    artifact_store: Optional[ArtifactStore],
) -> dict[str, tuple[ArtifactRef, bytes]]:
    """Read every shareable artifact's bytes once, for either output shape."""
    refs = shareable_artifacts(events)
    if not refs or artifact_store is None:
        return {}
    loaded: dict[str, tuple[ArtifactRef, bytes]] = {}
    for digest, ref in refs.items():
        try:
            data = await artifact_store.open(ref)
        except Exception:
            # A missing artifact must not abort an export; the player renders the
            # preview text that the event still carries.
            continue
        loaded[digest] = (ref, data)
    return loaded


def _write_artifacts(directory: Path, artifacts: dict[str, tuple[ArtifactRef, bytes]]) -> int:
    if not artifacts:
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    for digest, (_, data) in artifacts.items():
        (directory / digest).write_bytes(data)
    return len(artifacts)


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
