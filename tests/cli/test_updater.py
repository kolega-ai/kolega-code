import json
import subprocess

from kolega_code.cli import updater


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_check_for_update_detects_newer_version(monkeypatch) -> None:
    monkeypatch.setattr(updater, "current_version", lambda: "0.2.0")
    monkeypatch.setattr(updater.request, "urlopen", lambda request, timeout: FakeResponse({"info": {"version": "0.3.0"}}))

    result = updater.check_for_update()

    assert result.current_version == "0.2.0"
    assert result.latest_version == "0.3.0"
    assert result.update_available is True
    assert result.error is None


def test_check_for_update_accepts_current_version(monkeypatch) -> None:
    monkeypatch.setattr(updater, "current_version", lambda: "0.2.0")
    monkeypatch.setattr(updater.request, "urlopen", lambda request, timeout: FakeResponse({"info": {"version": "0.2.0"}}))

    result = updater.check_for_update()

    assert result.update_available is False
    assert result.error is None


def test_check_for_update_reports_invalid_latest_version(monkeypatch) -> None:
    monkeypatch.setattr(updater, "current_version", lambda: "0.2.0")
    monkeypatch.setattr(updater.request, "urlopen", lambda request, timeout: FakeResponse({"info": {"version": "not a version"}}))

    result = updater.check_for_update()

    assert result.update_available is False
    assert result.error


def test_update_status_message_formats_available_update() -> None:
    result = updater.UpdateCheckResult(current_version="0.2.0", latest_version="0.3.0", update_available=True)

    assert updater.update_status_message(result) == "Update available: 0.2.0 -> 0.3.0. Run `kolega-code update`."


def test_run_self_update_requires_uv(monkeypatch) -> None:
    monkeypatch.setattr(updater.shutil, "which", lambda command: None)

    result = updater.run_self_update(capture_output=True)

    assert result.returncode == 2
    assert "uv is required" in (result.error or "")


def test_run_self_update_runs_uv_tool_upgrade(monkeypatch) -> None:
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(updater.shutil, "which", lambda command: "/usr/bin/uv")
    monkeypatch.setattr(updater.subprocess, "run", fake_run)

    result = updater.run_self_update(capture_output=True)

    assert result.returncode == 0
    assert result.stdout == "ok\n"
    assert calls == [
        (
            ["uv", "tool", "install", "--force", "--upgrade", "kolega-code"],
            {"text": True, "capture_output": True, "check": False},
        )
    ]
