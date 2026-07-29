from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import kolega_code.browser_extension.installer as installer_module
from kolega_code.browser_extension.installer import (
    CHROME_EXTENSION_ID,
    NATIVE_HOST_NAME,
    NativeHostInstallError,
    build_native_host_manifest,
    chrome_extension_origin,
    chrome_native_host_manifest_path,
    install_native_host,
    native_host_status,
    resolve_channel_extension_id,
    uninstall_native_host,
)
from kolega_code.browser_extension.runtime import UnsupportedRuntimeTransportError


def executable(tmp_path: Path) -> Path:
    path = tmp_path / "bin" / "kolega-code-browser-host"
    path.parent.mkdir(parents=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def install_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    host = executable(tmp_path)
    config = tmp_path / "state" / "native-host.json"
    manifest = chrome_native_host_manifest_path(platform="darwin", home=tmp_path)
    return host, config, manifest


def test_install_status_and_uninstall_manage_owned_private_files(tmp_path: Path) -> None:
    host, config, manifest_path = install_paths(tmp_path)
    state_dir = tmp_path / "runtime-state"
    status = install_native_host(
        host_path=host,
        platform="darwin",
        home=tmp_path,
        state_dir=state_dir,
        host_config_path=config,
    )
    assert status.installed is True
    assert status.valid is True
    assert status.host_path == host.resolve()
    assert status.extension_id == CHROME_EXTENSION_ID
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == build_native_host_manifest(host)
    assert json.loads(config.read_text(encoding="utf-8")) == {
        "extension_origin": chrome_extension_origin(),
        "registry_dir": str((state_dir / "browser-extension" / "runtimes").resolve()),
        "max_pending_requests": 256,
        "max_runtimes": 16,
    }
    assert os.stat(manifest_path).st_mode & 0o077 == 0
    assert os.stat(config).st_mode & 0o077 == 0
    assert native_host_status(
        platform="darwin",
        home=tmp_path,
        state_dir=state_dir,
        host_config_path=config,
        expected_host_path=host,
    ).valid

    assert uninstall_native_host(platform="darwin", home=tmp_path, host_config_path=config) is True
    assert not manifest_path.exists()
    assert not config.exists()
    assert uninstall_native_host(platform="darwin", home=tmp_path, host_config_path=config) is False


def test_status_rejects_a_manifest_pointing_to_a_different_executable(tmp_path: Path) -> None:
    host, config, _ = install_paths(tmp_path)
    state_dir = tmp_path / "runtime-state"
    install_native_host(
        host_path=host,
        platform="darwin",
        home=tmp_path,
        state_dir=state_dir,
        host_config_path=config,
    )
    unrelated = executable(tmp_path / "unrelated")
    status = native_host_status(
        platform="darwin",
        home=tmp_path,
        state_dir=state_dir,
        host_config_path=config,
        expected_host_path=unrelated,
    )
    assert status.installed is True
    assert status.valid is False


def test_install_refuses_foreign_manifest_without_modifying_config(tmp_path: Path) -> None:
    host, config, manifest_path = install_paths(tmp_path)
    manifest_path.parent.mkdir(parents=True)
    original = '{"name":"some.other.host"}\n'
    manifest_path.write_text(original, encoding="utf-8")
    manifest_path.chmod(0o600)
    with pytest.raises(NativeHostInstallError, match="foreign"):
        install_native_host(
            host_path=host,
            platform="darwin",
            home=tmp_path,
            state_dir=tmp_path / "runtime-state",
            host_config_path=config,
        )
    assert manifest_path.read_text(encoding="utf-8") == original
    assert not config.exists()


def test_install_refuses_foreign_config_without_modifying_manifest(tmp_path: Path) -> None:
    host, config, manifest_path = install_paths(tmp_path)
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "extension_origin": chrome_extension_origin(),
                "registry_dir": str(tmp_path / "other-runtime"),
                "max_runtimes": 16,
                "max_pending_requests": 256,
            }
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    with pytest.raises(NativeHostInstallError, match="another runtime"):
        install_native_host(
            host_path=host,
            platform="darwin",
            home=tmp_path,
            state_dir=tmp_path / "runtime-state",
            host_config_path=config,
        )
    assert not manifest_path.exists()


def test_install_rolls_back_both_owned_files_after_a_mid_transaction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, manifest_path = install_paths(tmp_path)
    state_dir = tmp_path / "runtime-state"
    install_native_host(
        host_path=host,
        platform="darwin",
        home=tmp_path,
        state_dir=state_dir,
        host_config_path=config,
    )
    original_config = config.read_bytes()
    original_manifest = manifest_path.read_bytes()
    real_write = installer_module._write_private_json

    def fail_manifest(path: Path, value: dict[str, object]) -> None:
        if path == manifest_path:
            raise OSError("simulated manifest write failure")
        real_write(path, value)

    monkeypatch.setattr(installer_module, "_write_private_json", fail_manifest)
    with pytest.raises(OSError, match="simulated"):
        install_native_host(
            host_path=host,
            platform="darwin",
            home=tmp_path,
            state_dir=state_dir,
            host_config_path=config,
        )
    assert config.read_bytes() == original_config
    assert manifest_path.read_bytes() == original_manifest


def test_uninstall_refuses_foreign_artifacts(tmp_path: Path) -> None:
    _, config, manifest_path = install_paths(tmp_path)
    manifest_path.parent.mkdir(parents=True)
    original = '{"name":"some.other.host"}\n'
    manifest_path.write_text(original, encoding="utf-8")
    manifest_path.chmod(0o600)
    with pytest.raises(NativeHostInstallError, match="foreign"):
        uninstall_native_host(platform="darwin", home=tmp_path, host_config_path=config)
    assert manifest_path.read_text(encoding="utf-8") == original


def test_uninstall_validates_all_artifacts_before_removing_any(tmp_path: Path) -> None:
    host, config, manifest_path = install_paths(tmp_path)
    install_native_host(
        host_path=host,
        platform="darwin",
        home=tmp_path,
        state_dir=tmp_path / "runtime-state",
        host_config_path=config,
    )
    original_manifest = manifest_path.read_bytes()
    config.write_text("{malformed", encoding="utf-8")
    with pytest.raises(NativeHostInstallError, match="malformed"):
        uninstall_native_host(platform="darwin", home=tmp_path, host_config_path=config)
    assert manifest_path.read_bytes() == original_manifest
    assert config.read_text(encoding="utf-8") == "{malformed"


def test_uninstall_restores_files_after_a_mid_transaction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, manifest_path = install_paths(tmp_path)
    install_native_host(
        host_path=host,
        platform="darwin",
        home=tmp_path,
        state_dir=tmp_path / "runtime-state",
        host_config_path=config,
    )
    original_config = config.read_bytes()
    original_manifest = manifest_path.read_bytes()
    real_unlink = Path.unlink
    failed = False

    def fail_config_unlink(path: Path, missing_ok: bool = False) -> None:
        nonlocal failed
        if path == config and not failed:
            failed = True
            raise OSError("simulated config removal failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_config_unlink)
    with pytest.raises(OSError, match="simulated"):
        uninstall_native_host(platform="darwin", home=tmp_path, host_config_path=config)
    assert config.read_bytes() == original_config
    assert manifest_path.read_bytes() == original_manifest


def test_manifest_path_is_macos_only_and_channel_ids_are_exact(tmp_path: Path) -> None:
    darwin = chrome_native_host_manifest_path(platform="darwin", home=tmp_path)
    assert darwin == (
        tmp_path
        / "Library"
        / "Application Support"
        / "Google"
        / "Chrome"
        / "NativeMessagingHosts"
        / f"{NATIVE_HOST_NAME}.json"
    )
    for platform in ("linux", "win32"):
        with pytest.raises(UnsupportedRuntimeTransportError, match="only on macOS"):
            chrome_native_host_manifest_path(platform=platform, home=tmp_path)
    assert resolve_channel_extension_id() == ("production", CHROME_EXTENSION_ID)
    with pytest.raises(NativeHostInstallError, match="conflicting"):
        resolve_channel_extension_id(extension_id="b" * 32)
    with pytest.raises(NativeHostInstallError, match="explicit"):
        resolve_channel_extension_id(channel="dev")
    assert resolve_channel_extension_id(channel="dev", extension_id="b" * 32) == ("dev", "b" * 32)


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_install_rejects_non_macos_without_writing_files(tmp_path: Path, platform: str) -> None:
    host = executable(tmp_path)
    state_dir = tmp_path / "state"
    config = tmp_path / "native-host.json"

    with pytest.raises(UnsupportedRuntimeTransportError, match="only on macOS"):
        install_native_host(
            host_path=host,
            platform=platform,
            home=tmp_path,
            state_dir=state_dir,
            host_config_path=config,
        )

    assert not state_dir.exists()
    assert not config.exists()
