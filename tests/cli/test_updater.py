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


def test_run_self_update_runs_uv_tool_install_when_no_compatible_venv(monkeypatch) -> None:
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(updater.shutil, "which", lambda command: "/usr/bin/uv")
    monkeypatch.setattr(updater, "existing_tool_python_compatible", lambda **_kwargs: False)
    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    # Run regardless of whether this machine has the gateway service installed.
    monkeypatch.setattr("kolega_code.gateway.service.is_service_installed", lambda: False)

    result = updater.run_self_update(capture_output=True)

    assert result.returncode == 0
    assert result.stdout == "ok\n"
    expected_cmd = (
        "uv",
        "tool",
        "install",
        "--force",
        "--upgrade",
        "--python",
        updater.tool_python(sys.platform, platform.machine()),
        "kolega-code",
    )
    assert result.command == expected_cmd
    assert calls == [(list(expected_cmd), {"text": True, "capture_output": True, "check": False})]


def test_run_self_update_prefers_in_place_upgrade_when_existing_python_compatible(monkeypatch) -> None:
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="upgraded\n", stderr="")

    monkeypatch.setattr(updater.shutil, "which", lambda command: "/usr/bin/uv")
    monkeypatch.setattr(updater, "existing_tool_python_compatible", lambda **_kwargs: True)
    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    monkeypatch.setattr("kolega_code.gateway.service.is_service_installed", lambda: False)

    result = updater.run_self_update(capture_output=True)

    assert result.returncode == 0
    assert result.command == updater.UPGRADE_COMMAND
    assert calls == [(["uv", "tool", "upgrade", "kolega-code"], {"text": True, "capture_output": True, "check": False})]


def test_run_self_update_imports_gateway_helpers_before_running_uv(monkeypatch) -> None:
    order: list[str] = []

    def fake_run(args, **kwargs):
        order.append("uv")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(updater.shutil, "which", lambda command: "/usr/bin/uv")
    monkeypatch.setattr(updater, "existing_tool_python_compatible", lambda **_kwargs: False)
    monkeypatch.setattr(updater.subprocess, "run", fake_run)

    # Remove the gateway module from sys.modules after pre-import by tracking
    # when is_service_installed is called versus uv execution.
    def fake_is_installed() -> bool:
        order.append("gateway_check")
        return False

    monkeypatch.setattr("kolega_code.gateway.service.is_service_installed", fake_is_installed)
    # Remove the module during uv execution: if run_self_update waited to import
    # kolega_code.gateway.service until after uv ran, this would fail.
    orig_fake_run = fake_run

    def wipe_module_on_uv(args, **kwargs):
        res = orig_fake_run(args, **kwargs)
        monkeypatch.setitem(sys.modules, "kolega_code.gateway.service", None)
        return res

    monkeypatch.setattr(updater.subprocess, "run", wipe_module_on_uv)

    result = updater.run_self_update(capture_output=True)

    assert result.returncode == 0
    assert result.gateway_restart_notes == ()
    assert order == ["uv", "gateway_check"]


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


def _make_fake_tool_venv(
    tool_dir: Path,
    *,
    version_line: str = "version_info = 3.13.11",
    include_receipt: bool = True,
    include_python: bool = True,
    broken_python_symlink: bool = False,
) -> Path:
    env_dir = tool_dir / "kolega-code"
    (env_dir / "bin").mkdir(parents=True, exist_ok=True)
    if include_receipt:
        (env_dir / "uv-receipt.toml").write_text("[tool]\n", encoding="utf-8")
    (env_dir / "pyvenv.cfg").write_text(f"home = /usr/bin\n{version_line}\n", encoding="utf-8")
    py = env_dir / "bin" / "python"
    if broken_python_symlink:
        py.symlink_to(tool_dir / "does-not-exist")
    elif include_python:
        py.write_text("#!/bin/sh\n", encoding="utf-8")
        py.chmod(0o755)
    return env_dir


@pytest.mark.parametrize(
    "version_line,sys_platform,machine,expected",
    [
        ("version_info = 3.13.11", "darwin", "arm64", True),
        ("version_info = 3.14.2", "linux", "x86_64", True),
        ("version = 3.11.5.final.0", "linux", "x86_64", True),
        ("version_info = 3.11.4", "darwin", "arm64", False),
        ("version_info = 3.15.0", "darwin", "arm64", False),
        ("version_info = 3.12.10", "darwin", "x86_64", True),
        ("version_info = 3.13.1", "darwin", "x86_64", False),
    ],
)
def test_existing_tool_python_compatible_checks_version_range(
    tmp_path: Path, version_line: str, sys_platform: str, machine: str, expected: bool
) -> None:
    _make_fake_tool_venv(tmp_path, version_line=version_line)

    assert (
        updater.existing_tool_python_compatible(tool_dir=tmp_path, sys_platform=sys_platform, machine=machine)
        is expected
    )


def test_existing_tool_python_compatible_rejects_broken_or_missing_venv(tmp_path: Path) -> None:
    assert updater.existing_tool_python_compatible(tool_dir=tmp_path, sys_platform="darwin", machine="arm64") is False

    no_receipt = tmp_path / "no_receipt"
    _make_fake_tool_venv(no_receipt, include_receipt=False)
    assert updater.existing_tool_python_compatible(tool_dir=no_receipt, sys_platform="darwin", machine="arm64") is False

    broken_link = tmp_path / "broken_link"
    _make_fake_tool_venv(broken_link, broken_python_symlink=True)
    assert (
        updater.existing_tool_python_compatible(tool_dir=broken_link, sys_platform="darwin", machine="arm64") is False
    )


def test_installer_uses_uv_tool_upgrade_when_existing_venv_is_compatible(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool_dir = tmp_path / "tools"
    env_dir = tool_dir / "kolega-code"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "uv-receipt.toml").write_text("[tool]\n", encoding="utf-8")
    (env_dir / "bin" / "python").symlink_to(sys.executable)

    uv_args = tmp_path / "uv-tool-args"
    stubs = {
        "uname": 'case "$1" in -s) echo Darwin ;; -m) echo arm64 ;; esac',
        "uv": f'if [ "$1" = tool ] && [ "$2" != dir ]; then printf "%s\\n" "$@" > "{uv_args}"; fi',
        "kolega-code": "exit 0",
        "rg": "echo ripgrep 15.0.0",
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        stub.chmod(0o755)

    subprocess.run(
        ["/bin/sh", str(INSTALLER)],
        env={"HOME": str(tmp_path), "PATH": f"{bin_dir}:/usr/bin:/bin", "UV_TOOL_DIR": str(tool_dir)},
        capture_output=True,
        check=True,
    )
    assert uv_args.read_text(encoding="utf-8").splitlines() == ["tool", "upgrade", "kolega-code"]

    # An explicit KOLEGA_CODE_VERSION pin still uses `uv tool install --force`.
    subprocess.run(
        ["/bin/sh", str(INSTALLER)],
        env={
            "HOME": str(tmp_path),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "UV_TOOL_DIR": str(tool_dir),
            "KOLEGA_CODE_VERSION": "0.56.0",
        },
        capture_output=True,
        check=True,
    )
    assert uv_args.read_text(encoding="utf-8").splitlines() == [
        "tool",
        "install",
        "--force",
        "--upgrade",
        "--python",
        updater.TOOL_PYTHON,
        "kolega-code==0.56.0",
    ]

    # When the existing venv's Python fails the range check (e.g. Python 3.15),
    # the installer falls back to recreating the venv with --python.
    (env_dir / "bin" / "python").unlink()
    (env_dir / "bin" / "python").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    (env_dir / "bin" / "python").chmod(0o755)
    subprocess.run(
        ["/bin/sh", str(INSTALLER)],
        env={"HOME": str(tmp_path), "PATH": f"{bin_dir}:/usr/bin:/bin", "UV_TOOL_DIR": str(tool_dir)},
        capture_output=True,
        check=True,
    )
    assert uv_args.read_text(encoding="utf-8").splitlines() == [
        "tool",
        "install",
        "--force",
        "--upgrade",
        "--python",
        updater.TOOL_PYTHON,
        "kolega-code",
    ]


def test_install_fingerprint_and_drift_detection(tmp_path: Path) -> None:
    # Editable / source checkout layout has no sibling .dist-info -> disabled.
    checkout_init = tmp_path / "repo" / "kolega_code" / "__init__.py"
    checkout_init.parent.mkdir(parents=True)
    checkout_init.write_text('__version__ = "0.56.0"\n', encoding="utf-8")
    assert updater.capture_install_fingerprint(checkout_init) is None

    # Installed wheel layout: site-packages/kolega_code/__init__.py + sibling .dist-info.
    site = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
    pkg_init = site / "kolega_code" / "__init__.py"
    pkg_init.parent.mkdir(parents=True)
    pkg_init.write_text('__version__ = "0.55.0"\n', encoding="utf-8")
    dist_55 = site / "kolega_code-0.55.0.dist-info"
    dist_55.mkdir()

    fp = updater.capture_install_fingerprint(pkg_init)
    assert fp is not None
    assert fp.version == "0.55.0"
    assert updater.detect_install_drift(fp) is None

    # In-place upgrade replaces dist-info with newer version.
    dist_55.rmdir()
    (site / "kolega_code-0.56.0.dist-info").mkdir()
    drift = updater.detect_install_drift(fp)
    assert drift == updater.InstallDrift(reason="updated", previous_version="0.55.0", current_version="0.56.0")
    assert drift is not None
    warning = updater.install_drift_warning(drift)
    assert "0.55.0 -> 0.56.0" in warning
    assert "/restart" in warning

    # Deleting the package file (e.g. Python 3.13 -> 3.14 venv swap) reports missing.
    pkg_init.unlink()
    missing = updater.detect_install_drift(fp)
    assert missing == updater.InstallDrift(reason="missing", previous_version="0.55.0")
    assert missing is not None
    assert "/restart" in updater.install_drift_warning(missing)


def test_count_other_live_sessions_ignores_self_and_dead_pids(tmp_path: Path, monkeypatch) -> None:
    from kolega_code.session import inbox

    sock_dir = tmp_path / inbox.MESSAGING_DIR_NAME
    sock_dir.mkdir(parents=True)
    sid_a = "a" * 32
    sid_b = "b" * 32
    sid_c = "c" * 32
    inbox.socket_path_for(sock_dir, sid_a, 1001).touch()
    inbox.socket_path_for(sock_dir, sid_b, 1002).touch()
    inbox.socket_path_for(sock_dir, sid_c, 9999).touch()
    (sock_dir / "foreign.sock").touch()

    monkeypatch.setattr(inbox, "is_pid_alive", lambda pid: pid in {1001, 1002})

    assert updater.count_other_live_sessions(tmp_path, exclude_pid=1001) == 1
    assert updater.count_other_live_sessions(tmp_path, exclude_pid=2000) == 2
    assert updater.other_live_sessions_note(0) is None
    assert "/restart" in (updater.other_live_sessions_note(1) or "")
    assert "2 other running" in (updater.other_live_sessions_note(2) or "")
