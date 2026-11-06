import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from kolega_code.cli import updater

INSTALLER = Path(__file__).resolve().parents[2] / "scripts" / "install-kolega-code.sh"


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
    monkeypatch.setattr(
        updater.request, "urlopen", lambda request, timeout: FakeResponse({"info": {"version": "0.3.0"}})
    )

    result = updater.check_for_update()

    assert result.current_version == "0.2.0"
    assert result.latest_version == "0.3.0"
    assert result.update_available is True
    assert result.error is None


def test_check_for_update_accepts_current_version(monkeypatch) -> None:
    monkeypatch.setattr(updater, "current_version", lambda: "0.2.0")
    monkeypatch.setattr(
        updater.request, "urlopen", lambda request, timeout: FakeResponse({"info": {"version": "0.2.0"}})
    )

    result = updater.check_for_update()

    assert result.update_available is False
    assert result.error is None


def test_check_for_update_reports_invalid_latest_version(monkeypatch) -> None:
    monkeypatch.setattr(updater, "current_version", lambda: "0.2.0")
    monkeypatch.setattr(
        updater.request, "urlopen", lambda request, timeout: FakeResponse({"info": {"version": "not a version"}})
    )

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
    # Run regardless of whether this machine has the gateway service installed.
    monkeypatch.setattr("kolega_code.gateway.service.is_service_installed", lambda: False)

    result = updater.run_self_update(capture_output=True)

    assert result.returncode == 0
    assert result.stdout == "ok\n"
    assert calls == [
        (
            [
                "uv",
                "tool",
                "install",
                "--force",
                "--upgrade",
                "--python",
                updater.tool_python(sys.platform, platform.machine()),
                "kolega-code",
            ],
            {"text": True, "capture_output": True, "check": False},
        )
    ]


def test_tool_python_narrows_only_for_intel_macs() -> None:
    assert updater.tool_python("darwin", "x86_64") == updater.INTEL_MAC_TOOL_PYTHON
    assert updater.tool_python("darwin", "arm64") == updater.TOOL_PYTHON
    assert updater.tool_python("linux", "x86_64") == updater.TOOL_PYTHON
    assert updater.tool_python("linux", "aarch64") == updater.TOOL_PYTHON


@pytest.mark.parametrize(
    "system,machine", [("Darwin", "arm64"), ("Darwin", "x86_64"), ("Linux", "x86_64"), ("Linux", "aarch64")]
)
def test_installer_requests_the_updater_python_range(tmp_path: Path, system: str, machine: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_args = tmp_path / "uv-tool-args"
    stubs = {
        "uname": f'case "$1" in -s) echo {system} ;; -m) echo {machine} ;; esac',
        "uv": f'if [ "$1" = tool ]; then printf "%s\\n" "$@" > "{uv_args}"; fi',
        "kolega-code": "exit 0",
        "rg": "echo ripgrep 15.0.0",
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        stub.chmod(0o755)

    subprocess.run(
        ["/bin/sh", str(INSTALLER)],
        env={"HOME": str(tmp_path), "PATH": f"{bin_dir}:/usr/bin:/bin"},
        capture_output=True,
        check=True,
    )

    assert uv_args.read_text(encoding="utf-8").splitlines() == [
        "tool",
        "install",
        "--force",
        "--upgrade",
        "--python",
        updater.tool_python(system.lower(), machine),
        "kolega-code",
    ]


def test_run_self_update_restarts_gateway_when_installed(monkeypatch) -> None:
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(updater.shutil, "which", lambda command: "/usr/bin/uv")
    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    monkeypatch.setattr("kolega_code.gateway.service.is_service_installed", lambda: True)
    monkeypatch.setattr(
        "kolega_code.gateway.service.restart_service",
        lambda: (True, ["launchd agent restarted"]),
    )

    result = updater.run_self_update(capture_output=True)

    assert result.returncode == 0
    assert result.gateway_restart_notes == ("launchd agent restarted",)


def test_run_self_update_skips_gateway_restart_when_not_installed(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(updater.shutil, "which", lambda command: "/usr/bin/uv")
    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    monkeypatch.setattr("kolega_code.gateway.service.is_service_installed", lambda: False)

    result = updater.run_self_update(capture_output=True)

    assert result.returncode == 0
    assert result.gateway_restart_notes == ()


def test_run_self_update_gateway_restart_failure_is_non_fatal(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(updater.shutil, "which", lambda command: "/usr/bin/uv")
    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    monkeypatch.setattr("kolega_code.gateway.service.is_service_installed", lambda: True)
    monkeypatch.setattr(
        "kolega_code.gateway.service.restart_service",
        lambda: (False, ["failed to restart systemd unit"]),
    )

    result = updater.run_self_update(capture_output=True)

    assert result.returncode == 0
    assert result.gateway_restart_notes == ("failed to restart systemd unit",)
