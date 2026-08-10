"""Tests for local diagnostics: secret scrubbing, the JSONL log, and the responsiveness
watchdog (which captures *where* the event loop is blocked when the UI goes unresponsive)."""

import json
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from kolega_code.cli.diagnostics import (
    SECRET_PLACEHOLDER,
    DiagnosticsLog,
    ResponsivenessWatchdog,
    assemble_bug_bundle,
    scrub_secrets,
    write_crash_log,
)
from kolega_code.cli.session_store import SessionBugExport


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        session_json='{"history": [{"role": "user", "content": "fix my bug, key=sk-deadbeef-secret-9999"}]}',
        events_jsonl='{"type":"turn.started","payload":{"key":"sk-deadbeef-secret-9999"}}\n',
        artifact_manifest_json='{"artifacts_included":false,"artifacts":[]}',
    )
    bundle = assemble_bug_bundle(
        diag,
        summary="kolega diag\nkey sk-deadbeef-secret-9999",
        session_export=session_export,
    )
    assert bundle is not None and bundle.is_file()
    assert bundle.suffix == ".zip"

    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        summary = zf.read("summary.md").decode("utf-8")
        session = zf.read("session.json").decode("utf-8")
        events = zf.read("session-events.jsonl").decode("utf-8")
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
        session_json=json.dumps({"history": [{"role": "assistant", "content": _SOURCE_LINE}]}),
        events_jsonl=json.dumps({"type": "turn.started", "payload": {"text": _SOURCE_LINE}}) + "\n",
        artifact_manifest_json=json.dumps({"artifacts_included": False, "artifacts": []}),
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


def test_bundle_falls_back_to_text_scrubbing_for_unparseable_json(tmp_path: Path):
    diag = DiagnosticsLog(tmp_path, "sess-bundle-bad")
    session_export = SessionBugExport(
        session_json="{not json at all, token=" + _FAKE_TOKEN,
        events_jsonl="",
        artifact_manifest_json="{}",
    )

    bundle = assemble_bug_bundle(diag, summary="kolega diag", session_export=session_export)

    assert bundle is not None
    with zipfile.ZipFile(bundle) as zf:
        session = zf.read("session.json").decode("utf-8")

    # Entry is still present and still scrubbed, just not reformatted.
    assert _FAKE_TOKEN not in session
    assert "not json at all" in session


def test_bundle_keeps_good_jsonl_lines_when_one_line_is_malformed(tmp_path: Path):
    diag = DiagnosticsLog(tmp_path, "sess-bundle-mixed")
    session_export = SessionBugExport(
        session_json="{}",
        events_jsonl="\n".join(
            [
                json.dumps({"seq": 1, "text": "fine"}),
                "{broken",
                json.dumps({"seq": 3, "text": _SOURCE_LINE}),
            ]
        )
        + "\n",
        artifact_manifest_json="{}",
    )

    bundle = assemble_bug_bundle(diag, summary="kolega diag", session_export=session_export)

    assert bundle is not None
    with zipfile.ZipFile(bundle) as zf:
        lines = zf.read("session-events.jsonl").decode("utf-8").splitlines()

    assert len(lines) == 3
    assert json.loads(lines[0])["seq"] == 1
    assert json.loads(lines[2])["seq"] == 3
    assert _FAKE_TOKEN not in "\n".join(lines)


def test_bundle_preserves_readable_json_formatting(tmp_path: Path):
    """SessionStore writes these with indent=2/sort_keys=True; redaction must not flatten them."""
    diag = DiagnosticsLog(tmp_path, "sess-bundle-fmt")
    session_export = SessionBugExport(
        session_json=json.dumps({"b": 1, "a": {"nested": True}}, indent=2, sort_keys=True) + "\n",
        events_jsonl="",
        artifact_manifest_json=json.dumps({"artifacts_included": False}, indent=2, sort_keys=True) + "\n",
    )

    bundle = assemble_bug_bundle(diag, summary="kolega diag", session_export=session_export)

    assert bundle is not None
    with zipfile.ZipFile(bundle) as zf:
        session = zf.read("session.json").decode("utf-8")

    assert session.startswith("{\n")
    assert '\n  "a": {' in session  # indented and key-sorted
    assert session.endswith("\n")


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
