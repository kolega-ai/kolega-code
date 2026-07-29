from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from kolega_code.browser_extension.installer import (
    CHROME_EXTENSION_ID,
    CHROME_WEB_STORE_URL,
    NativeHostStatus,
)
from kolega_code.cli.main import _run_browser, parse_args


def _args(command: str, tmp_path: Path, *, json_output: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        browser_command=command,
        channel="production",
        extension_id=None,
        state_dir=tmp_path,
        host_path=None,
        json=json_output,
    )


def _status(tmp_path: Path, *, valid: bool = True) -> NativeHostStatus:
    return NativeHostStatus(
        manifest_path=tmp_path / "ai.kolega.browser.json",
        installed=True,
        valid=valid,
        host_path=tmp_path / "kolega-code-browser-host",
        extension_id=CHROME_EXTENSION_ID,
        detail="valid" if valid else "stale",
    )


def test_browser_subcommands_parse_registration_options(tmp_path: Path) -> None:
    args = parse_args(
        [
            "browser",
            "install",
            "--state-dir",
            str(tmp_path),
            "--host-path",
            str(tmp_path / "host"),
            "--channel",
            "dev",
            "--extension-id",
            "a" * 32,
            "--json",
        ]
    )

    assert args.command == "browser"
    assert args.browser_command == "install"
    assert args.state_dir == tmp_path
    assert args.host_path == tmp_path / "host"
    assert args.channel == "dev"
    assert args.extension_id == "a" * 32
    assert args.json is True


def test_browser_install_reports_status_and_opens_the_production_listing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = _status(tmp_path)
    with (
        patch("kolega_code.cli.main.install_native_host", return_value=status) as install,
        patch("kolega_code.cli.main.webbrowser.open") as open_browser,
    ):
        result = _run_browser(_args("install", tmp_path))

    assert result == 0
    install.assert_called_once_with(
        host_path=None,
        channel="production",
        extension_id=None,
        state_dir=tmp_path,
    )
    open_browser.assert_called_once_with(CHROME_WEB_STORE_URL)
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["web_store_url"] == CHROME_WEB_STORE_URL


def test_browser_status_and_doctor_return_one_for_invalid_configuration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("kolega_code.cli.main.native_host_status", return_value=_status(tmp_path, valid=False)):
        assert _run_browser(_args("status", tmp_path)) == 1
        status_payload = json.loads(capsys.readouterr().out)
        assert status_payload["valid"] is False

        assert _run_browser(_args("doctor", tmp_path)) == 1
        doctor_payload = json.loads(capsys.readouterr().out)
        assert "browser install" in doctor_payload["remediation"]


def test_browser_uninstall_is_a_successful_noop_when_not_installed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("kolega_code.cli.main.uninstall_native_host", return_value=False) as uninstall:
        assert _run_browser(_args("uninstall", tmp_path)) == 0

    uninstall.assert_called_once_with(channel="production", extension_id=None)
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed"] is False


def test_browser_command_errors_return_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with patch("kolega_code.cli.main.native_host_status", side_effect=RuntimeError("unsafe manifest")):
        assert _run_browser(_args("status", tmp_path)) == 2

    assert json.loads(capsys.readouterr().out)["error"] == "unsafe manifest"


def test_native_host_console_entry_point_is_packaged() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert (
        metadata["project"]["scripts"]["kolega-code-browser-host"] == "kolega_code.browser_extension.native_host:main"
    )
