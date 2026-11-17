"""Tests for local diagnostics: secret scrubbing, the JSONL log, and the responsiveness
watchdog (which captures *where* the event loop is blocked when the UI goes unresponsive)."""

import io
import json
import os
import threading
import time
import zipfile
from concurrent.futures import CancelledError
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Iterator
from unittest.mock import patch

import pytest

from kolega_code.cli.diagnostics import (
    SECRET_PLACEHOLDER,
    DiagnosticsLog,
    ResponsivenessWatchdog,
    assemble_bug_bundle,
    scrub_secrets,
    scrub_secrets_in_obj,
    write_crash_log,
)
from kolega_code.cli.session_journal import SessionEvent
from kolega_code.cli.session_store import SessionBugExport


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _event(payload: dict[str, Any], *, version: int = 1, seq: int = 1) -> SessionEvent:
    return SessionEvent(
        version=version,
        event_id=f"event-{seq}",
        session_id="session-test",
        seq=seq,
        epoch_id="epoch-test",
        turn_id="turn-test",
        timestamp="2026-11-17T12:00:00+00:00",
        actor="assistant",
        event_type="turn.started",
        payload=payload,
        artifacts=[{"digest": "digest-test", "path": "artifacts/missing.txt", "size_bytes": 999}],
        agent_id="child-test",
        agent_name="researcher",
        parent_agent_id="parent-test",
        parent_tool_call_id="call-test",
        depth=1,
    )


def test_scrub_secrets_removes_credentials_keeps_content():
    text = (
        "normal prompt text about main.py\n"
        "Authorization: Bearer abc123def456ghi\n"
        "DEEPSEEK_API_KEY=supersecretvalue123\n"
        "token sk-abcdef1234567890 and xai-zyxwvu9876543210\n"
        "my key is mysecretkey-1234\n"
    )
    out = scrub_secrets(text, extra_values=["mysecretkey-1234"])
    # Content preserved:
    assert "normal prompt text about main.py" in out
    # Credentials gone:
    assert "abc123def456ghi" not in out
    assert "supersecretvalue123" not in out
    assert "sk-abcdef1234567890" not in out
    assert "xai-zyxwvu9876543210" not in out
    assert "mysecretkey-1234" not in out
    assert SECRET_PLACEHOLDER in out
    # Labels kept (so the record stays readable):
    assert "Authorization: Bearer" in out
    assert "DEEPSEEK_API_KEY=" in out


def test_log_records_jsonl_and_scrubs_secret_fields(tmp_path: Path):
    diag = DiagnosticsLog(tmp_path, "sess1", secret_values=["topsecret-value-xyz"])
    diag.record(
        "llm_error",
        provider="deepseek",
        model="deepseek-v4-pro",
        http_status=400,
        message="bad request; key topsecret-value-xyz leaked",
    )
    rows = _read(diag.path)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "llm_error" and row["provider"] == "deepseek" and row["http_status"] == 400
    assert "topsecret-value-xyz" not in json.dumps(row)
    assert SECRET_PLACEHOLDER in row["message"]


def test_log_bounds_large_fields(tmp_path: Path):
    diag = DiagnosticsLog(tmp_path, "sess2")
    diag.record("blob", data="x" * (DiagnosticsLog.MAX_FIELD_CHARS + 5000))
    row = _read(diag.path)[0]
    assert len(row["data"]) < DiagnosticsLog.MAX_FIELD_CHARS + 100
    assert "chars]" in row["data"]


def test_disabled_log_writes_nothing(tmp_path: Path):
    diag = DiagnosticsLog(tmp_path, "sess3", enabled=False)
    diag.record("x", a=1)
    assert not diag.path.exists()


def _simulated_blocking_call() -> None:
    # The watchdog dumps stacks while the (main) thread sits here; its frame must show up.
    # A generous block (vs the ~0.2s detection) keeps the test robust under suite load.
    time.sleep(1.0)


def test_watchdog_captures_loop_stall_with_blocking_stack(tmp_path: Path):
    diag = DiagnosticsLog(tmp_path, "sess4")
    watchdog = ResponsivenessWatchdog(diag, stall_seconds=0.15, check_interval=0.03)
    watchdog.start()
    try:
        _simulated_blocking_call()  # never beat -> watchdog sees the stall and dumps stacks
    finally:
        watchdog.beat()  # simulate the loop recovering
        time.sleep(0.2)  # let the watchdog observe recovery
        watchdog.stop()

    rows = _read(diag.path)
    stalled = [r for r in rows if r["kind"] == "event_loop_stalled"]
    recovered = [r for r in rows if r["kind"] == "event_loop_recovered"]
    assert stalled, "watchdog did not record a stall"
    assert "_simulated_blocking_call" in stalled[0]["stacks"], "stall dump should name the blocking frame"
    assert recovered, "watchdog did not record recovery"
    # The stall dump is also written to a sidecar for the bug bundle.
    assert (diag.dir / "stalls.log").exists()


def test_watchdog_histograms_sub_second_gaps(tmp_path: Path):
    """Chop below the stack-capture threshold has to leave *some* trace."""
    diag = DiagnosticsLog(tmp_path, "sess5")
    watchdog = ResponsivenessWatchdog(diag, beat_interval=0.0, histogram_interval=3600.0)

    # Drive beat() with controlled lateness instead of real sleeps.
    now = [1000.0]
    with patch("kolega_code.cli.diagnostics.time.monotonic", lambda: now[0]):
        watchdog._last_beat = now[0]
        for gap in (0.01, 0.15, 0.3, 2.0, 7.0):
            now[0] += gap
            watchdog.beat()
        watchdog.flush_histogram("test")

    rows = [r for r in _read(diag.path) if r["kind"] == "loop_gap_histogram"]
    assert len(rows) == 1
    assert rows[0]["buckets"] == {"0.1-0.25s": 1, "0.25-1.0s": 1, "1.0-5.0s": 1, ">=5.0s": 1}
    assert rows[0]["beats"] == 5  # the 10ms gap is counted as a beat but bucketed nowhere
    assert rows[0]["max_gap_s"] == 7.0

    # Counters reset per window, and an empty window writes nothing.
    watchdog.flush_histogram("test")
    assert len([r for r in _read(diag.path) if r["kind"] == "loop_gap_histogram"]) == 1


def test_watchdog_caps_stack_captures_but_keeps_recording_stalls(tmp_path: Path):
    diag = DiagnosticsLog(tmp_path, "sess6")
    watchdog = ResponsivenessWatchdog(diag, max_stall_captures=1)

    watchdog._record_stall(2.0)
    watchdog._record_stall(3.0)

    stalls = [r for r in _read(diag.path) if r["kind"] == "event_loop_stalled"]
    assert len(stalls) == 2
    assert "stacks" in stalls[0] and "stacks_omitted" not in stalls[0]
    assert stalls[1]["stacks_omitted"] is True and "stacks" not in stalls[1]


def test_watchdog_stop_flushes_the_final_histogram(tmp_path: Path):
    diag = DiagnosticsLog(tmp_path, "sess7")
    watchdog = ResponsivenessWatchdog(diag, beat_interval=0.0, histogram_interval=3600.0)
    watchdog._last_beat = time.monotonic() - 0.4
    watchdog.beat()
    watchdog.stop()

    rows = [r for r in _read(diag.path) if r["kind"] == "loop_gap_histogram"]
    assert len(rows) == 1
    assert rows[0]["reason"] == "session_end"


def test_write_crash_log_captures_scrubbed_traceback(tmp_path: Path):
    try:
        raise RuntimeError("boom; leaked key sk-deadbeef12345678 here")
    except RuntimeError as exc:
        path = write_crash_log(tmp_path, exc=exc, header="crash test header")
    assert path is not None and path.exists()
    text = path.read_text(encoding="utf-8")
    assert "crash test header" in text
    assert "Traceback" in text and "RuntimeError" in text and "boom" in text
    assert "sk-deadbeef12345678" not in text  # secrets scrubbed even in crash logs


def test_write_crash_log_scrubs_configured_secret_values(tmp_path: Path):
    """Configured API keys passed via secret_values are scrubbed, not just pattern-matched ones."""
    custom_key = "my-custom-provider-key-abcdef123456"  # no built-in pattern matches this
    try:
        raise RuntimeError(f"auth failed for key {custom_key}")
    except RuntimeError as exc:
        path = write_crash_log(tmp_path, exc=exc, header="hdr", secret_values=[custom_key])
    assert path is not None and path.exists()
    text = path.read_text(encoding="utf-8")
    assert custom_key not in text
    assert "auth failed for key" in text  # content preserved, only the key removed


def test_assemble_bug_bundle_scrubs_secrets_keeps_content(tmp_path: Path):
    diag = DiagnosticsLog(tmp_path, "sess5", secret_values=["sk-deadbeef-secret-9999"])
    diag.record("session_start", version="0.11.1", provider="deepseek", term="xterm-ghostty")
    session_export = SessionBugExport(
        session={"history": [{"role": "user", "content": "fix my bug, key=sk-deadbeef-secret-9999"}]},
        events=[_event({"key": "sk-deadbeef-secret-9999"})],
        artifact_manifest={"artifacts_included": False, "artifacts": []},
    )
    bundle = assemble_bug_bundle(
        diag,
        summary="kolega diag\nkey sk-deadbeef-secret-9999",
        session_export=session_export,
    )
    assert bundle is not None and bundle.is_file()
    assert bundle.suffix == ".zip"

    with zipfile.ZipFile(bundle) as zf:
        assert zf.testzip() is None
        names = zf.namelist()
        summary = zf.read("summary.md").decode("utf-8")
        session = zf.read("session.json").decode("utf-8")
        events = zf.read("session-events.jsonl").decode("utf-8")
        assert json.loads(zf.read("session-artifacts.json")) == session_export.artifact_manifest
    assert "sk-deadbeef-secret-9999" not in summary and "sk-deadbeef-secret-9999" not in session
    assert "sk-deadbeef-secret-9999" not in events
    # Ordinary conversation content is preserved (unredacted):
    assert "fix my bug" in session
    assert "session-sess5.jsonl" in names
    assert "session-artifacts.json" in names


# ---------------------------------------------------------------------------
# JSON payloads must survive redaction intact
# ---------------------------------------------------------------------------

_FAKE_TOKEN = "apify_api_9RtQv3LmZx8KpWn2Yc7BdF4HsJ6Tg1Ae0Nu5"
_SOURCE_LINE = f'api_url = f"https://api.example.com/v2/runs?token={_FAKE_TOKEN}"'


def test_log_line_stays_parseable_when_a_secret_precedes_an_escaped_quote(tmp_path: Path):
    """Scrubbing the serialized form used to splice across the \\" escape and break the line."""
    diag = DiagnosticsLog(tmp_path, "sess-escape")
    diag.record("tool", name="read", output=_SOURCE_LINE)

    records = _read(diag.path)

    assert len(records) == 1
    assert _FAKE_TOKEN not in records[0]["output"]
    assert SECRET_PLACEHOLDER in records[0]["output"]


def test_log_record_redacts_values_reached_only_via_str_coercion(tmp_path: Path):
    """json.dumps(default=str) stringified after scrubbing, so these leaked."""

    class Failure:
        def __str__(self) -> str:
            return f"boom token={_FAKE_TOKEN}"

    diag = DiagnosticsLog(tmp_path, "sess-coerce")
    diag.record("llm_error", error=Failure())

    records = _read(diag.path)

    assert _FAKE_TOKEN not in json.dumps(records[0])


def test_bundle_session_json_stays_parseable_with_adversarial_content(tmp_path: Path):
    diag = DiagnosticsLog(tmp_path, "sess-bundle-json")
    session_export = SessionBugExport(
        session={"history": [{"role": "assistant", "content": _SOURCE_LINE}]},
        events=[_event({"text": _SOURCE_LINE})],
        artifact_manifest={"artifacts_included": False, "artifacts": []},
    )

    bundle = assemble_bug_bundle(diag, summary="kolega diag", session_export=session_export)

    assert bundle is not None
    with zipfile.ZipFile(bundle) as zf:
        session = zf.read("session.json").decode("utf-8")
        events = zf.read("session-events.jsonl").decode("utf-8")

    parsed = json.loads(session)  # would raise before the fix
    assert _FAKE_TOKEN not in session
    assert SECRET_PLACEHOLDER in parsed["history"][0]["content"]
    for line in events.splitlines():
        assert json.loads(line)
    assert _FAKE_TOKEN not in events


def test_bundle_matches_structural_redaction_for_all_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    configured = 'configured-"雪\\credential-123456'
    environment = "environment-only-credential-987654"
    monkeypatch.setenv("BUNDLE_TEST_API_KEY", environment)
    diag = DiagnosticsLog(tmp_path, "sess-bundle-values", secret_values=[configured])

    class Failure:
        def __str__(self) -> str:
            return f'failure "{configured}" \\ {environment} token={_FAKE_TOKEN}'

    payload = {
        configured: f'Unicode 雪 😀, quotes " and slash \\, {configured} {environment}',
        "nested": [None, True, False, 0, -10, 1.25, (), ("ordinary", Failure()), {"text": _SOURCE_LINE}],
        "not_serialized_json": "{not json at all",
        "nonnative": Failure(),
    }
    events = [_event(payload, version=version, seq=version) for version in (1, 2)]
    session_export = SessionBugExport(
        session=payload,
        events=events,
        artifact_manifest={"artifacts_included": False, "artifacts": [payload]},
    )

    bundle = assemble_bug_bundle(diag, summary=f"{configured} {environment}", session_export=session_export)

    assert bundle is not None
    with zipfile.ZipFile(bundle) as zf:
        assert zf.testzip() is None
        assert set(zf.namelist()) == {"summary.md", "session.json", "session-events.jsonl", "session-artifacts.json"}
        assert json.loads(zf.read("session.json")) == scrub_secrets_in_obj(payload, [configured])
        assert json.loads(zf.read("session-artifacts.json")) == scrub_secrets_in_obj(
            session_export.artifact_manifest, [configured]
        )
        rows = [json.loads(line) for line in zf.read("session-events.jsonl").decode().splitlines()]
        assert rows == [scrub_secrets_in_obj(event.to_dict(), [configured]) for event in events]
        assert zf.read("summary.md").decode() == f"{SECRET_PLACEHOLDER} {SECRET_PLACEHOLDER}"
    assert configured in rows[0]["payload"]  # Structural keys are deliberately unchanged.
    assert "schema" not in rows[0]
    assert rows[1]["agent_id"] == "child-test"
    assert rows[1]["parent_tool_call_id"] == "call-test"
    assert [row["id"] for row in rows] == ["event-1", "event-2"]


def test_bundle_preserves_readable_json_formatting(tmp_path: Path):
    """SessionStore writes these with indent=2/sort_keys=True; redaction must not flatten them."""
    diag = DiagnosticsLog(tmp_path, "sess-bundle-fmt")
    session_export = SessionBugExport(
        session={"b": 1, "a": {"nested": True}},
        events=[],
        artifact_manifest={"artifacts_included": False},
    )

    bundle = assemble_bug_bundle(diag, summary="kolega diag", session_export=session_export)

    assert bundle is not None
    with zipfile.ZipFile(bundle) as zf:
        session = zf.read("session.json").decode("utf-8")
        manifest = zf.read("session-artifacts.json").decode()
        assert zf.read("session-events.jsonl") == b""

    assert session == json.dumps(session_export.session, indent=2, sort_keys=True) + "\n"
    assert manifest == json.dumps(session_export.artifact_manifest, indent=2, sort_keys=True) + "\n"


def test_bundle_streams_huge_scalars_and_diagnostic_files_in_bounded_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "configured-streaming-credential-123456"
    diag = DiagnosticsLog(tmp_path, "sess-streaming", secret_values=[secret])
    text = ("x" * (64 * 1024 - 12)) + secret + ('雪😀\\"\n' * 50_000) + " ordinary suffix"
    diagnostic = ("already scrubbed diagnostic 雪\n" * 10_000).encode()
    source_path = diag.dir / "stalls.log"
    source_path.write_bytes(diagnostic)
    session_export = SessionBugExport(
        session={"history": [{"content": text}]},
        events=[_event({"content": text})],
        artifact_manifest={"artifacts_included": False, "artifacts": [{"path": text}]},
    )
    sizes: dict[str, list[int]] = {}
    reads: list[int] = []
    zip_write_file = getattr(zipfile, "_ZipWriteFile")
    original_write = zip_write_file.write
    original_dumps = json.dumps
    original_open = Path.open

    def observed_write(target: Any, data: Any) -> int:
        sizes.setdefault(target._zinfo.filename, []).append(len(data))
        assert len(data) <= 64 * 1024
        return original_write(target, data)

    class ObservedSource(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            assert 0 < size <= 64 * 1024
            reads.append(size)
            return super().read(size)

    def observed_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == source_path:
            return ObservedSource(diagnostic)
        return original_open(path, *args, **kwargs)

    def scalar_dumps(value: Any, *args: Any, **kwargs: Any) -> str:
        assert not isinstance(value, (dict, list, tuple))
        if isinstance(value, str):
            assert len(value) <= 8192
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(zip_write_file, "write", observed_write)
    monkeypatch.setattr(json, "dumps", scalar_dumps)
    monkeypatch.setattr(Path, "open", observed_open)
    monkeypatch.setattr(zipfile.ZipFile, "writestr", lambda *a, **kw: pytest.fail("full-entry writestr"))
    monkeypatch.setattr(zipfile.ZipFile, "write", lambda *a, **kw: pytest.fail("unbounded file copy"))
    monkeypatch.setattr(
        "kolega_code.cli.diagnostics.scrub_secrets_in_obj", lambda *a, **kw: pytest.fail("full scrubbed tree")
    )

    bundle = assemble_bug_bundle(diag, summary=text, session_export=session_export)

    assert bundle is not None
    assert len(reads) > 2
    for name in ("summary.md", "session.json", "session-events.jsonl", "session-artifacts.json", "stalls.log"):
        assert len(sizes[name]) > 2
    with zipfile.ZipFile(bundle) as zf:
        assert zf.testzip() is None
        assert zf.read("stalls.log") == diagnostic
        assert json.loads(zf.read("session.json")) == scrub_secrets_in_obj(session_export.session, [secret])
        assert zf.read("summary.md").decode() == scrub_secrets(text, [secret])
        assert secret.encode() not in zf.read("session-events.jsonl")
        assert secret.encode() not in zf.read("session-artifacts.json")


def test_bundle_skips_optional_log_that_disappears_before_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    diag = DiagnosticsLog(tmp_path, "sess-missing")
    diag.record("ordinary", text="retained")
    missing = diag.dir / "stalls.log"
    missing.touch()
    original_open = Path.open

    def disappeared(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == missing:
            raise FileNotFoundError(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", disappeared)

    bundle = assemble_bug_bundle(diag, summary="summary", session_export=None)

    assert bundle is not None
    with zipfile.ZipFile(bundle) as zf:
        assert zf.testzip() is None
        assert set(zf.namelist()) == {"summary.md", diag.path.name}
        assert json.loads(zf.read(diag.path.name))["text"] == "retained"


@pytest.mark.parametrize("failure", ["open", "read", "read_late"])
def test_bundle_skips_unreadable_optional_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    diag = DiagnosticsLog(tmp_path, "sess-unreadable")
    diag.record("ordinary", text="retained")
    unreadable = diag.dir / "old-crash.log"
    unreadable.touch()
    before = set(diag.dir.iterdir())
    original_open = Path.open

    class FailingSource(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            if failure == "read" or self.tell() >= 2 * 64 * 1024:
                raise OSError("source read failed")
            return super().read(size)

    def source_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == unreadable:
            if failure == "open":
                raise PermissionError(path)
            return FailingSource(b"partial optional data\n" * 16_384)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", source_open)
    export = SessionBugExport(session={"history": ["ordinary conversation"]}, events=[], artifact_manifest={})

    bundle = assemble_bug_bundle(diag, summary="summary", session_export=export)

    assert bundle is not None
    assert set(diag.dir.iterdir()) == before | {bundle}
    with zipfile.ZipFile(bundle) as zf:
        assert zf.testzip() is None
        assert set(zf.namelist()) == {
            "summary.md",
            diag.path.name,
            "session.json",
            "session-events.jsonl",
            "session-artifacts.json",
        }
        assert json.loads(zf.read(diag.path.name))["text"] == "retained"
        assert json.loads(zf.read("session.json")) == export.session


@pytest.mark.parametrize("failure", ["stage_write", "stage_read", "archive_write"])
def test_bundle_optional_log_destination_failure_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    diag = DiagnosticsLog(tmp_path, "sess-optional-write")
    diag.write_sidecar("stalls.log", "retained content")
    before = set(diag.dir.iterdir())
    zip_write_file = getattr(zipfile, "_ZipWriteFile")
    original_write = zip_write_file.write

    class FailingStage(io.BytesIO):
        def write(self, data: Any) -> int:
            if failure == "stage_write":
                raise OSError("staging write failed")
            return super().write(data)

        def read(self, size: int = -1) -> bytes:
            if failure == "stage_read":
                raise OSError("staging read failed")
            return super().read(size)

    def target_write(target: Any, data: Any) -> int:
        if failure == "archive_write" and target._zinfo.filename == "stalls.log":
            raise OSError("archive write failed")
        return original_write(target, data)

    monkeypatch.setattr("kolega_code.cli.diagnostics.tempfile.SpooledTemporaryFile", lambda **kwargs: FailingStage())
    monkeypatch.setattr(zip_write_file, "write", target_write)

    assert assemble_bug_bundle(diag, summary="summary", session_export=None) is None
    assert set(diag.dir.iterdir()) == before


def test_bundle_private_creation_closed_publication_and_same_second_unique_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diag = DiagnosticsLog(tmp_path, "sess-private")
    original_init = zipfile.ZipFile.__init__
    original_replace = Path.replace
    observed: list[Path] = []

    def inspect_creation(target: Any, file: IO[bytes], *args: Any, **kwargs: Any) -> None:
        if args and args[0] == "w":
            temp_paths = list(diag.dir.glob(".bug-*.tmp"))
            assert len(temp_paths) == 1
            observed.append(temp_paths[0])
            if os.name != "nt":
                assert os.fstat(file.fileno()).st_mode & 0o777 == 0o600
        original_init(target, file, *args, **kwargs)

    def inspect_publication(source: Path, target: Path) -> Path:
        assert not target.exists()
        with zipfile.ZipFile(source) as zf:
            assert zf.testzip() is None
            assert zf.read("summary.md") in (b"first", b"second")
        return original_replace(source, target)

    monkeypatch.setattr(zipfile.ZipFile, "__init__", inspect_creation)
    monkeypatch.setattr(Path, "replace", inspect_publication)
    with patch("kolega_code.cli.diagnostics.datetime") as clock:
        clock.now.return_value = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
        first = assemble_bug_bundle(diag, summary="first", session_export=None)
        second = assemble_bug_bundle(diag, summary="second", session_export=None)

    assert first is not None and second is not None and first != second
    assert len(observed) == 2
    for bundle in (first, second):
        assert bundle.name.startswith("bug-20261117T120000Z-")
        assert bundle.suffix == ".zip"
        if os.name != "nt":
            assert bundle.stat().st_mode & 0o777 == 0o600
    with zipfile.ZipFile(first) as zf:
        assert zf.read("summary.md") == b"first"
    assert not list(diag.dir.glob(".bug-*.tmp"))


@pytest.mark.parametrize("failure", ["write", "close", "publish", "stringify"])
def test_bundle_failure_removes_partial_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    diag = DiagnosticsLog(tmp_path, "sess-failure")
    diag.record("retained")
    before = set(diag.dir.iterdir())

    class Failure:
        def __str__(self) -> str:
            raise RuntimeError("stringify failed")

    session_export = SessionBugExport(
        session={"value": Failure() if failure == "stringify" else "ordinary"},
        events=[],
        artifact_manifest={},
    )

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("storage unavailable")

    if failure == "write":
        monkeypatch.setattr(getattr(zipfile, "_ZipWriteFile"), "write", fail)
    elif failure == "close":
        original_close = zipfile.ZipFile.close

        def fail_close(archive: zipfile.ZipFile) -> None:
            was_open = archive.fp is not None
            original_close(archive)
            if was_open:
                raise OSError("close failed")

        monkeypatch.setattr(zipfile.ZipFile, "close", fail_close)
    elif failure == "publish":
        monkeypatch.setattr(Path, "replace", fail)

    if failure == "stringify":
        with pytest.raises(RuntimeError, match="stringify failed"):
            assemble_bug_bundle(diag, summary="summary", session_export=session_export)
    else:
        assert assemble_bug_bundle(diag, summary="summary", session_export=session_export) is None
    assert set(diag.dir.iterdir()) == before


@pytest.mark.parametrize(
    "stage",
    ["before", "open", "read", "read_eof", "write", "event", "stringify", "traversal", "closed"],
)
def test_bundle_cancellation_removes_partial_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    diag = DiagnosticsLog(tmp_path, "sess-cancel")
    source_path = diag.dir / "stalls.log"
    source_path.write_bytes(b"already scrubbed\n")
    before = set(diag.dir.iterdir())
    cancel = threading.Event()
    original_open = Path.open
    zip_write_file = getattr(zipfile, "_ZipWriteFile")
    original_write = zip_write_file.write
    original_to_dict = SessionEvent.to_dict
    original_close = zipfile.ZipFile.close
    visited: list[int] = []

    class CancellingSource(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            data = super().read(size)
            if stage == "read" or (stage == "read_eof" and not data):
                cancel.set()
            return data

    def source_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == source_path:
            if stage == "open":
                cancel.set()
            return CancellingSource(b"already scrubbed\n")
        return original_open(path, *args, **kwargs)

    def target_write(target: Any, data: Any) -> int:
        result = original_write(target, data)
        if stage == "write" and target._zinfo.filename == "session-artifacts.json":
            cancel.set()
        return result

    def event_to_dict(event: SessionEvent) -> dict[str, Any]:
        visited.append(event.seq)
        result = original_to_dict(event)
        if stage == "event":
            cancel.set()
        return result

    def archive_close(archive: zipfile.ZipFile) -> None:
        original_close(archive)
        if stage == "closed":
            cancel.set()

    class CancellingValue:
        def __str__(self) -> str:
            cancel.set()
            return "ordinary"

    class CancellingList(list[Any]):
        def __iter__(self) -> Iterator[Any]:
            for item in super().__iter__():
                cancel.set()
                yield item

    value = CancellingValue() if stage == "stringify" else "ordinary"
    history = CancellingList([value]) if stage == "traversal" else [value]
    session_export = SessionBugExport(
        session={"history": history},
        events=[_event({}, seq=1), _event({}, seq=2)],
        artifact_manifest={"artifacts_included": False},
    )
    monkeypatch.setattr(Path, "open", source_open)
    monkeypatch.setattr(zip_write_file, "write", target_write)
    monkeypatch.setattr(SessionEvent, "to_dict", event_to_dict)
    monkeypatch.setattr(zipfile.ZipFile, "close", archive_close)
    if stage == "before":
        cancel.set()

    with pytest.raises(CancelledError):
        assemble_bug_bundle(diag, summary="summary", session_export=session_export, cancel_event=cancel)

    assert set(diag.dir.iterdir()) == before
    if stage == "event":
        assert visited == [1]


def test_bundle_published_before_cancellation_remains_a_completed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diag = DiagnosticsLog(tmp_path, "sess-published")
    cancel = threading.Event()
    original_replace = Path.replace

    def publish_then_cancel(source: Path, target: Path) -> Path:
        result = original_replace(source, target)
        cancel.set()
        return result

    monkeypatch.setattr(Path, "replace", publish_then_cancel)

    bundle = assemble_bug_bundle(diag, summary="completed", session_export=None, cancel_event=cancel)

    assert cancel.is_set()
    assert bundle is not None
    assert not list(diag.dir.glob(".bug-*.tmp"))
    with zipfile.ZipFile(bundle) as zf:
        assert zf.testzip() is None
        assert zf.read("summary.md") == b"completed"


def test_disabled_bundle_creates_no_directory(tmp_path: Path) -> None:
    diag = DiagnosticsLog(tmp_path, "sess-disabled", enabled=False)

    assert assemble_bug_bundle(diag, summary="summary", session_export=None) is None
    assert not diag.dir.exists()


def test_watchdog_recent_excess_peaks_and_decays(tmp_path: Path):
    diag = DiagnosticsLog(tmp_path, "sess5b")
    watchdog = ResponsivenessWatchdog(diag, beat_interval=0.0, histogram_interval=3600.0)

    now = [1000.0]
    with patch("kolega_code.cli.diagnostics.time.monotonic", lambda: now[0]):
        watchdog._last_beat = now[0]
        now[0] += 1.0
        watchdog.beat()
        # A single late beat registers at full strength immediately.
        assert watchdog.recent_excess() == pytest.approx(1.0)

        # The histogram window reset leaves the pacing signal alone.
        watchdog.flush_histogram("test")
        assert watchdog.recent_excess() == pytest.approx(1.0)

        # On-time beats decay the peak below the pacing deadband in ~15 beats.
        for _ in range(15):
            watchdog.beat()
        assert watchdog.recent_excess() == pytest.approx(0.85**15)
        assert watchdog.recent_excess() < 0.1


def test_watchdog_recent_excess_never_negative(tmp_path: Path):
    diag = DiagnosticsLog(tmp_path, "sess5c")
    watchdog = ResponsivenessWatchdog(diag, beat_interval=0.2, histogram_interval=3600.0)

    now = [1000.0]
    with patch("kolega_code.cli.diagnostics.time.monotonic", lambda: now[0]):
        watchdog._last_beat = now[0]
        now[0] += 0.05  # early-firing timer: excess is negative
        watchdog.beat()
        assert watchdog.recent_excess() == 0.0
