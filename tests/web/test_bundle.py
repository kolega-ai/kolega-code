"""Share bundles: redaction, completeness, and offline playability.

Export is the only path where session content crosses a trust boundary, so the
redaction assertions here are the security tests for this feature. They are
written as "nothing forbidden is present" rather than "something expected is
absent", because the failure mode that matters is a leak nobody looked for.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from kolega_code.events import AgentEvent, ArtifactPurpose, ArtifactRef, KnownEventType
from kolega_code.session.inmemory import InMemoryArtifactStore
from kolega_code.session.projection import replay
from kolega_code.web.bundle import export_bundle
from kolega_code.web.redaction import RedactionReport, redact_event, shareable_artifacts

SECRET = "sk-live-51H9xTOTALLYSECRETvalue0000"
ALL_PURPOSES = (
    ArtifactPurpose.TOOL_RESULT,
    ArtifactPurpose.IMAGE,
    ArtifactPurpose.PROVIDER_SIGNATURE,
    ArtifactPurpose.REDACTED_REASONING,
    ArtifactPurpose.ENCRYPTED_REASONING,
    ArtifactPurpose.THOUGHT_SIGNATURE,
)


def _event(event_type: str, seq: int, *, elapsed_ms: int = 0, **content) -> AgentEvent:
    return AgentEvent(
        session_id="s1",
        sender="agent",
        event_type=event_type,
        content=dict(content),
        seq=seq,
        elapsed_ms=elapsed_ms,
    )


async def _seeded_session(artifacts: InMemoryArtifactStore) -> list[AgentEvent]:
    """A session containing a secret and one artifact of every purpose."""
    refs = []
    for index, purpose in enumerate(ALL_PURPOSES):
        refs.append(
            await artifacts.put(
                f"payload for {purpose} {index}".encode(),
                media_type="application/octet-stream",
                purpose=purpose,
                encoding="utf-8",
                chars=10,
            )
        )
    tool_event = _event(
        KnownEventType.CHAT_MESSAGE,
        3,
        elapsed_ms=300,
        message_type="tool_result",
        text=f"exported API_KEY={SECRET} to the environment",
        tool_call_id="c1",
    )
    tool_event.artifacts = refs
    # Local paths must not survive export either.
    tool_event.artifacts[0].path = "/Users/someone/.local/state/kolega/artifacts/abc"
    return [
        _event(KnownEventType.TURN_STARTED, 1, turn_id="t1", user_text=f"use {SECRET} please"),
        _event(KnownEventType.TERMINAL_OUTPUT, 2, elapsed_ms=200, output=f"echo {SECRET}\n"),
        tool_event,
        _event(KnownEventType.TURN_ENDED, 4, elapsed_ms=400, turn_id="t1", status="completed"),
    ]


@pytest.mark.asyncio
async def test_export_contains_no_secrets_anywhere(tmp_path: Path) -> None:
    artifacts = InMemoryArtifactStore()
    events = await _seeded_session(artifacts)

    result = await export_bundle(
        events,
        tmp_path / "bundle",
        session_id="s1",
        artifact_store=artifacts,
    )

    for path in sorted(result.path.rglob("*")):
        if not path.is_file():
            continue
        blob = path.read_bytes()
        assert SECRET.encode() not in blob, f"secret leaked into {path.name}"
    assert result.report.strings_redacted >= 3, "expected the seeded secret to be redacted in several places"


@pytest.mark.asyncio
async def test_export_keeps_only_shareable_artifact_purposes(tmp_path: Path) -> None:
    artifacts = InMemoryArtifactStore()
    events = await _seeded_session(artifacts)

    result = await export_bundle(
        events,
        tmp_path / "bundle",
        session_id="s1",
        artifact_store=artifacts,
    )

    exported = [
        json.loads(line) for line in (result.path / "events.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    purposes = {ref["purpose"] for event in exported for ref in event.get("artifacts", [])}
    assert purposes == {ArtifactPurpose.TOOL_RESULT, ArtifactPurpose.IMAGE}, (
        f"non-shareable artifact purposes survived export: {purposes}"
    )
    # Four opaque provider payloads must have been dropped, and reported.
    assert sum(result.report.artifacts_dropped.values()) == 4
    assert result.artifact_count == 2, "only the two shareable blobs may be written"
    written = {path.name for path in (result.path / "artifacts").iterdir()}
    assert len(written) == 2


@pytest.mark.asyncio
async def test_export_strips_local_filesystem_paths(tmp_path: Path) -> None:
    artifacts = InMemoryArtifactStore()
    events = await _seeded_session(artifacts)

    result = await export_bundle(events, tmp_path / "bundle", session_id="s1", artifact_store=artifacts)

    text = (result.path / "events.jsonl").read_text(encoding="utf-8")
    assert "/Users/someone" not in text, "an absolute local path leaked into the bundle"
    assert result.report.paths_stripped >= 1


def test_redaction_never_mutates_the_live_event() -> None:
    """The running session must keep full fidelity; only the copy is reduced."""
    event = _event(KnownEventType.CHAT_MESSAGE, 1, text=f"token {SECRET}")
    event.artifacts = [
        ArtifactRef(
            sha256="c" * 64,
            bytes=1,
            media_type="application/octet-stream",
            purpose=ArtifactPurpose.ENCRYPTED_REASONING,
            encoding="utf-8",
        )
    ]
    report = RedactionReport()

    redacted = redact_event(event, report=report)

    assert SECRET in event.content["text"], "the original event was mutated"
    assert len(event.artifacts) == 1, "the original artifact list was mutated"
    assert SECRET not in redacted.content["text"]
    assert redacted.artifacts == []


def test_shareable_artifacts_ignores_opaque_purposes() -> None:
    event = _event(KnownEventType.CHAT_MESSAGE, 1, text="x")
    event.artifacts = [
        # Distinct digests: identical content would legitimately dedupe.
        ArtifactRef(sha256=f"{index:064d}", bytes=1, media_type="m", purpose=purpose, encoding="utf-8")
        for index, purpose in enumerate(ALL_PURPOSES)
    ]

    collected = shareable_artifacts([event])

    assert {ref.purpose for ref in collected.values()} == {ArtifactPurpose.TOOL_RESULT, ArtifactPurpose.IMAGE}


def test_shareable_artifacts_dedupes_identical_content() -> None:
    first = _event(KnownEventType.CHAT_MESSAGE, 1, text="x")
    second = _event(KnownEventType.CHAT_MESSAGE, 2, text="y")
    ref = ArtifactRef(
        sha256="a" * 64,
        bytes=1,
        media_type="m",
        purpose=ArtifactPurpose.TOOL_RESULT,
        encoding="utf-8",
    )
    first.artifacts = [ref]
    second.artifacts = [ref.model_copy(deep=True)]

    assert len(shareable_artifacts([first, second])) == 1, "content-addressed artifacts must be written once"


@pytest.mark.asyncio
async def test_bundle_is_self_contained_and_offline_playable(tmp_path: Path) -> None:
    artifacts = InMemoryArtifactStore()
    events = await _seeded_session(artifacts)

    result = await export_bundle(events, tmp_path / "bundle", session_id="s1", artifact_store=artifacts)

    for required in ("index.html", "player.html", "player.js", "fold.js", "player.css", "theme.css", "manifest.json"):
        assert (result.path / required).is_file(), f"bundle is missing {required}"

    html = (result.path / "index.html").read_text(encoding="utf-8")
    assert "https://" not in html, "the player must not depend on a network resource to render"
    assert 'href="theme.css"' in html and 'src="player.js"' in html

    manifest = json.loads((result.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["session_id"] == "s1"
    assert manifest["event_count"] == 4
    assert manifest["duration_ms"] == 400
    assert manifest["theme"], "a bundle must record the theme it was captured in"
    assert manifest["themes"], "a viewer must be able to switch themes"
    assert [turn["turn_id"] for turn in manifest["turns"]] == ["t1"]
    assert manifest["turns"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_exported_events_still_fold_into_a_transcript(tmp_path: Path) -> None:
    """Redaction must not break the recording it protects."""
    artifacts = InMemoryArtifactStore()
    events = await _seeded_session(artifacts)

    result = await export_bundle(events, tmp_path / "bundle", session_id="s1", artifact_store=artifacts)

    restored = [
        AgentEvent.model_validate(json.loads(line))
        for line in (result.path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    state = replay(restored)
    assert [item.kind for item in state.conversation] == ["user", "tool"]
    assert state.turns[0].status == "completed"
    assert "echo" in state.terminal


@pytest.mark.asyncio
async def test_zip_export_produces_a_single_archive(tmp_path: Path) -> None:
    artifacts = InMemoryArtifactStore()
    events = await _seeded_session(artifacts)

    result = await export_bundle(
        events,
        tmp_path / "share",
        session_id="s1",
        artifact_store=artifacts,
        as_zip=True,
    )

    assert result.path.suffix == ".zip" and result.path.is_file()
    assert not (tmp_path / "share").is_dir(), "the staging directory must be cleaned up"
    with zipfile.ZipFile(result.path) as bundle:
        names = set(bundle.namelist())
        assert {"index.html", "events.jsonl", "manifest.json", "theme.css"} <= names
        assert SECRET.encode() not in bundle.read("events.jsonl")


@pytest.mark.asyncio
async def test_export_tolerates_a_missing_artifact_blob(tmp_path: Path) -> None:
    """A pruned artifact must degrade to preview text, not abort the export."""
    events = [_event(KnownEventType.CHAT_MESSAGE, 1, message_type="tool_result", text="preview only")]
    events[0].artifacts = [
        ArtifactRef(
            sha256="e" * 64,
            bytes=10,
            media_type="text/plain",
            purpose=ArtifactPurpose.TOOL_RESULT,
            encoding="utf-8",
            chars=10,
        )
    ]

    result = await export_bundle(
        events,
        tmp_path / "bundle",
        session_id="s1",
        artifact_store=InMemoryArtifactStore(),
    )

    assert result.event_count == 1
    assert result.artifact_count == 0
    assert "preview only" in (result.path / "events.jsonl").read_text(encoding="utf-8")
