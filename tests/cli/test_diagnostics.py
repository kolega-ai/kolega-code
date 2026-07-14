"""Tests for local diagnostics: secret scrubbing, the JSONL log, and the responsiveness
watchdog (which captures *where* the event loop is blocked when the UI goes unresponsive)."""

import json
import time
import zipfile
from pathlib import Path

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
