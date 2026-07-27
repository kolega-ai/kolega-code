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


@pytest.mark.asyncio
async def test_zip_export_never_deletes_a_path_the_caller_named(tmp_path: Path) -> None:
    """Exporting must not be able to destroy data.

    The staging directory used to be the destination with its suffix stripped,
    and it was removed afterwards, so ``--out backup.zip`` deleted an unrelated
    ``backup/`` — recursively, silently, and with a zero exit code.
    """
    artifacts = InMemoryArtifactStore()
    events = await _seeded_session(artifacts)
    victim = tmp_path / "backup"
    (victim / "nested").mkdir(parents=True)
    (victim / "nested" / "keep.txt").write_text("precious")

    result = await export_bundle(
        events,
        tmp_path / "backup.zip",
        session_id="s1",
        artifact_store=artifacts,
        as_zip=True,
    )

    assert (victim / "nested" / "keep.txt").read_text() == "precious", (
        "an export wrote over a directory the caller never named"
    )
    assert result.path == tmp_path / "backup.zip" and result.path.is_file()


@pytest.mark.asyncio
async def test_zip_export_appends_to_a_dotted_name_instead_of_replacing_it(tmp_path: Path) -> None:
    """``--out my.project`` must not silently become ``my.zip``."""
    artifacts = InMemoryArtifactStore()
    events = await _seeded_session(artifacts)

    result = await export_bundle(
        events,
        tmp_path / "my.project",
        session_id="s1",
        artifact_store=artifacts,
        as_zip=True,
    )

    assert result.path == tmp_path / "my.project.zip"
    assert not (tmp_path / "my.zip").exists()


@pytest.mark.asyncio
async def test_zip_export_leaves_no_staging_directory_behind(tmp_path: Path) -> None:
    artifacts = InMemoryArtifactStore()
    events = await _seeded_session(artifacts)

    await export_bundle(events, tmp_path / "out.zip", session_id="s1", artifact_store=artifacts, as_zip=True)

    leftovers = [path.name for path in tmp_path.iterdir() if path.name.startswith(".kc-replay-")]
    assert leftovers == [], f"staging directories were left behind: {leftovers}"


@pytest.mark.asyncio
async def test_directory_export_warns_before_mixing_with_existing_files(tmp_path: Path) -> None:
    """A directory export is additive, so say when it lands on top of something."""
    artifacts = InMemoryArtifactStore()
    events = await _seeded_session(artifacts)
    target = tmp_path / "site"
    target.mkdir()
    (target / "unrelated.txt").write_text("mine")

    result = await export_bundle(events, target, session_id="s1", artifact_store=artifacts)

    assert (target / "unrelated.txt").read_text() == "mine", "a directory export must never delete"
    assert any("not empty" in warning for warning in result.warnings), result.warnings


@pytest.mark.asyncio
async def test_an_image_a_tool_produced_reaches_the_single_file_replay(tmp_path: Path) -> None:
    """End to end from an emitted event, because the wiring is what broke.

    Image artifacts are only ever created for provider-facing history records,
    which the exporter does not read, so every image in every real session was
    silently absent from its replay while a hand-built fixture passed.
    """
    import base64

    from kolega_code.events import AgentConnectionManager
    from kolega_code.session.inmemory import InMemorySessionEventStore
    from kolega_code.session.recording import RecordingConnectionManager

    class _Null(AgentConnectionManager):
        async def connect(self, *a, **k): ...
        def disconnect(self, *a, **k): ...
        async def broadcast_event(self, *a, **k): ...
        def get_connection_count(self, *a, **k):
            return {}

    png = b"\x89PNG\r\n\x1a\n" + b"pretend pixels" * 8
    artifacts = InMemoryArtifactStore()
    store = InMemorySessionEventStore()
    manager = RecordingConnectionManager(_Null(), store, session_id="s1", artifact_store=artifacts)
    await manager.broadcast_event(
        AgentEvent(
            sender="agent",
            event_type=KnownEventType.CHAT_MESSAGE,
            content={
                "message_type": "tool_result",
                "text": "# marker.png",
                "tool_description": "read_image",
                "images": [{"media_type": "image/png", "data": base64.b64encode(png).decode("ascii")}],
            },
        ),
        "w",
        "t",
    )
    await manager.flush()

    result = await export_bundle(
        await store.read("s1"),
        tmp_path / "replay.html",
        session_id="s1",
        artifact_store=artifacts,
        single_file=True,
    )

    document = result.path.read_text(encoding="utf-8")
    assert result.artifact_count == 1, "the image should have been inlined"
    assert f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}" in document


@pytest.mark.asyncio
async def test_an_oversize_image_falls_back_to_a_badge_and_is_reported(tmp_path: Path) -> None:
    from kolega_code.web import bundle as bundle_module

    artifacts = InMemoryArtifactStore()
    big = b"\x89PNG" + b"x" * (bundle_module.MAX_INLINE_IMAGE_BYTES + 1)
    ref = await artifacts.put(big, media_type="image/png", purpose=ArtifactPurpose.IMAGE, encoding="base64")
    event = _event(KnownEventType.CHAT_MESSAGE, 1, message_type="tool_result", text="shot")
    event.artifacts = [ref]

    result = await export_bundle(
        [event], tmp_path / "replay.html", session_id="s1", artifact_store=artifacts, single_file=True
    )

    assert result.artifact_count == 0, "an image over the cap must not be embedded"
    assert any("too large to embed" in warning for warning in result.warnings), result.warnings


def test_home_directory_is_rewritten_to_a_tilde(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A bundle should not disclose the directory layout that produced it.

    Only artifact references had their paths stripped, so the home directory
    still appeared thousands of times in command output, tool results, and the
    turn titles the player renders in its rail.
    """
    from kolega_code.web import redaction as redaction_module

    home = tmp_path / "home" / "someone"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    redaction_module._prefixes_for_home.cache_clear()

    report = RedactionReport()
    event = _event(
        KnownEventType.TERMINAL_OUTPUT,
        1,
        output=f"cd {home}/git/project && ls {home}/notes.txt",
    )
    safe = redact_event(event, report=report)

    assert str(home) not in safe.content["output"]
    assert safe.content["output"] == "cd ~/git/project && ls ~/notes.txt"
    assert report.home_paths_rewritten == 1
    assert any("home directory" in line for line in report.summary_lines())


def test_the_macos_private_twin_of_a_home_path_is_rewritten_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Command output prints whichever spelling the tool happened to resolve."""
    from kolega_code.web import redaction as redaction_module

    home = tmp_path / "home" / "someone"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    redaction_module._prefixes_for_home.cache_clear()

    resolved = str(home.resolve())
    safe = redact_event(
        _event(KnownEventType.TERMINAL_OUTPUT, 1, output=f"{resolved}/x"),
        report=RedactionReport(),
    )

    assert resolved not in safe.content["output"]


def test_caller_supplied_secrets_are_redacted_even_without_a_known_shape() -> None:
    """Pattern matching does not recognise every provider's key format."""
    unmatched = "fw_3ZUftNOTAREALKEY"
    report = RedactionReport()

    safe = redact_event(
        _event(KnownEventType.CHAT_MESSAGE, 1, message_type="tool_result", text=f"key={unmatched} done"),
        report=report,
        extra_secrets=[unmatched],
    )

    assert unmatched not in safe.content["text"]
    assert report.strings_redacted == 1


def test_the_summary_always_says_detection_is_best_effort() -> None:
    """ "Redacted 3 strings" must not read as "this bundle is clean"."""
    lines = RedactionReport(events_scanned=1).summary_lines()

    assert any("best-effort" in line for line in lines), lines


def test_sub_agent_dispatch_metadata_is_scrubbed_too() -> None:
    """A dispatch task is free text, and it rides every event of that dispatch.

    Scrubbing only ``content`` left the task string — which routinely quotes
    absolute paths and whatever the caller pasted in — repeated verbatim across
    hundreds of events, making it the single largest source of leaked host paths
    in a real export.
    """
    report = RedactionReport()
    event = _event(KnownEventType.ASSISTANT_DELTA, 1, text="fine")
    event.sub_agent_info = {
        "dispatch_id": "d1",
        "agent_name": "investigation-agent",
        "task": f"Investigate {SECRET} in the repo",
    }

    safe = redact_event(event, report=report)

    assert safe.sub_agent_info is not None
    assert SECRET not in safe.sub_agent_info["task"]
    assert safe.sub_agent_info["agent_name"] == "investigation-agent"
    assert event.sub_agent_info["task"].count(SECRET) == 1, "the live event must not be mutated"


@pytest.mark.asyncio
async def test_artifact_payloads_are_redacted_not_just_their_references(tmp_path: Path) -> None:
    """The bytes behind an artifact are the likeliest place for a secret.

    Only the reference was redacted; the payload was copied into the bundle
    verbatim. Those payloads exist *because* they were too big to inline — a
    file dump, a command that printed a config — so this was the one place in an
    export nothing looked at.
    """
    artifacts = InMemoryArtifactStore()
    payload = f"config dump\nAPI_KEY={SECRET}\npath=/home/somebody/project\n".encode()
    ref = await artifacts.put(
        payload,
        media_type="text/plain; charset=utf-8",
        purpose=ArtifactPurpose.TOOL_RESULT,
        encoding="utf-8",
        chars=len(payload),
    )
    event = _event(KnownEventType.CHAT_MESSAGE, 1, message_type="tool_result", text="see artifact")
    event.artifacts = [ref]

    result = await export_bundle([event], tmp_path / "site", session_id="s1", artifact_store=artifacts)

    written = (result.path / "artifacts" / ref.sha256).read_bytes()
    assert SECRET.encode() not in written, "an artifact payload carried a secret into the bundle"
    assert b"config dump" in written, "redaction must not destroy the payload"


@pytest.mark.asyncio
async def test_image_artifact_bytes_are_never_rewritten(tmp_path: Path) -> None:
    """Scrubbing binary would corrupt it, and an image cannot hold a string secret."""
    artifacts = InMemoryArtifactStore()
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
    ref = await artifacts.put(png, media_type="image/png", purpose=ArtifactPurpose.IMAGE, encoding="base64")
    event = _event(KnownEventType.CHAT_MESSAGE, 1, message_type="tool_result", text="shot")
    event.artifacts = [ref]

    result = await export_bundle([event], tmp_path / "site", session_id="s1", artifact_store=artifacts)

    assert (result.path / "artifacts" / ref.sha256).read_bytes() == png
