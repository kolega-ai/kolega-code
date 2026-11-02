"""Unit tests for the dedicated eval environment manager (hermetic, mocked I/O)."""

import asyncio
import json
import subprocess
import sys

import pytest

from kolega_code.agent.eval import env as env_module
from kolega_code.agent.eval.env import ENV_DIR_PREFIX, EvalEnvironmentError, EvalEnvironmentManager


def _ok(args, timeout, stdout="ok"):
    return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")


def _fail(args, timeout):
    return subprocess.CompletedProcess(args, 1, stdout="", stderr="boom")


def _probe_stdout(version="3.14.3"):
    return json.dumps(["CPython", version])


def _fake_installer(env_path, *, fail_install=False, fail_smoke=False):
    def run(cmd, timeout):
        if "venv" in cmd:
            python = env_module._venv_python(env_path)
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("#!/bin/sh\n")
            return _ok(cmd, timeout)
        if fail_install and "install" in cmd:
            return _fail(cmd, timeout)
        if fail_smoke and any("import numpy" in arg for arg in cmd):
            return _fail(cmd, timeout)
        if "-c" in cmd:
            return _ok(cmd, timeout, stdout=_probe_stdout())
        return _ok(cmd, timeout)

    return run


def _ensure(manager):
    return asyncio.run(manager.ensure())


def _make_valid_env(manager):
    python = env_module._venv_python(manager.env_path)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\n")
    manager._write_manifest("uv", "3.14.3")
    return python


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "state"


def test_override_python_skips_provisioning(state_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(env_module, "_run", lambda cmd, timeout: calls.append(cmd) or _ok(cmd, timeout))
    manager = EvalEnvironmentManager(state_dir=state_dir, override_python=sys.executable)

    info = _ensure(manager)

    assert info.python == sys.executable
    assert info.bundled is False
    assert "custom interpreter" in (info.note or "")
    assert calls == [[sys.executable, "--version"]]


def test_provisions_with_uv_and_writes_actual_version(state_dir, monkeypatch):
    calls = []
    manager = EvalEnvironmentManager(state_dir=state_dir)

    def run(cmd, timeout):
        calls.append(cmd)
        return _fake_installer(manager.env_path)(cmd, timeout)

    monkeypatch.setattr(env_module, "_run", run)
    monkeypatch.setattr(env_module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    info = _ensure(manager)

    assert info.bundled is True
    assert info.provisioned_now is True
    assert "CPython 3.14.3" in (info.note or "")
    assert manager.env_path.name.startswith(f"{ENV_DIR_PREFIX}-")
    assert calls[0] == ["/usr/bin/uv", "venv", str(manager.env_path), "--python", "3.14"]
    assert all("-I" in cmd for cmd in calls if "-c" in cmd)
    manifest = json.loads((manager.env_path / env_module.MANIFEST_NAME).read_text())
    assert manifest["python_version"] == "3.14"
    assert manifest["actual_python_version"] == "3.14.3"


def test_provisions_with_system_314_when_uv_missing(state_dir, monkeypatch):
    calls = []
    manager = EvalEnvironmentManager(state_dir=state_dir)

    def run(cmd, timeout):
        calls.append(cmd)
        return _fake_installer(manager.env_path)(cmd, timeout)

    monkeypatch.setattr(env_module, "_run", run)
    monkeypatch.setattr(
        env_module.shutil,
        "which",
        lambda name: "/usr/bin/python3.14" if name == "python3.14" else None,
    )

    _ensure(manager)

    assert calls[0] == ["/usr/bin/python3.14", "-m", "venv", str(manager.env_path)]
    assert calls[2][1:4] == ["-m", "pip", "install"]


def test_windows_python_launcher_fallback(state_dir, monkeypatch):
    calls = []
    manager = EvalEnvironmentManager(state_dir=state_dir)
    monkeypatch.setattr(env_module.sys, "platform", "win32")

    def run(cmd, timeout):
        calls.append(cmd)
        return _fake_installer(manager.env_path)(cmd, timeout)

    monkeypatch.setattr(env_module, "_run", run)
    monkeypatch.setattr(env_module.shutil, "which", lambda name: "py" if name == "py" else None)

    _ensure(manager)

    assert calls[0] == ["py", "-3.14", "-m", "venv", str(manager.env_path)]


def test_missing_uv_never_provisions_arbitrary_host_python(state_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(env_module, "_run", lambda cmd, timeout: calls.append(cmd) or _fail(cmd, timeout))
    monkeypatch.setattr(env_module.shutil, "which", lambda name: None)

    with pytest.raises(EvalEnvironmentError, match="no CPython 3.14"):
        _ensure(EvalEnvironmentManager(state_dir=state_dir))

    assert not any(cmd[:2] == [sys.executable, "-m"] for cmd in calls)


def test_failed_first_provision_leaves_legacy_env_untouched(state_dir, monkeypatch):
    legacy = state_dir / "eval-env"
    legacy.mkdir(parents=True)
    canary = legacy / "canary.txt"
    canary.write_text("working 3.12 environment")
    monkeypatch.setattr(env_module.shutil, "which", lambda name: None)

    with pytest.raises(EvalEnvironmentError):
        _ensure(EvalEnvironmentManager(state_dir=state_dir))

    assert canary.read_text() == "working 3.12 environment"


def test_failed_new_bundle_preserves_previous_environment(state_dir, monkeypatch):
    monkeypatch.setattr(EvalEnvironmentManager, "_bundle_source", lambda self: b"old")
    old_manager = EvalEnvironmentManager(state_dir=state_dir)
    monkeypatch.setattr(env_module, "_run", _fake_installer(old_manager.env_path))
    monkeypatch.setattr(env_module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    old_info = _ensure(old_manager)
    assert old_info.env_path is not None
    canary = old_info.env_path / "canary.txt"
    canary.write_text("preserve me")

    monkeypatch.setattr(EvalEnvironmentManager, "_bundle_source", lambda self: b"new")
    new_manager = EvalEnvironmentManager(state_dir=state_dir)
    monkeypatch.setattr(env_module, "_run", _fake_installer(new_manager.env_path, fail_install=True))

    with pytest.raises(EvalEnvironmentError, match="Previous environments were left unchanged"):
        _ensure(new_manager)

    assert new_manager.env_path != old_manager.env_path
    assert canary.read_text() == "preserve me"


def test_reuses_matching_manifest_with_isolated_verification(state_dir, monkeypatch):
    manager = EvalEnvironmentManager(state_dir=state_dir)
    python = _make_valid_env(manager)
    calls = []

    def run(cmd, timeout):
        calls.append(cmd)
        if "venv" in cmd or "install" in cmd:
            raise AssertionError("must not provision when the existing environment verifies")
        return _ok(cmd, timeout, stdout=_probe_stdout())

    monkeypatch.setattr(env_module, "_run", run)
    monkeypatch.setattr(env_module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    info = _ensure(manager)

    assert info.provisioned_now is False
    assert info.python == str(python)
    assert len(calls) == 2
    assert all("-I" in cmd for cmd in calls)


def test_wrong_existing_interpreter_is_an_actionable_error(state_dir, monkeypatch):
    manager = EvalEnvironmentManager(state_dir=state_dir)
    _make_valid_env(manager)
    canary = manager.env_path / "canary.txt"
    canary.write_text("preserve me")
    monkeypatch.setattr(env_module, "_run", lambda cmd, timeout: _ok(cmd, timeout, stdout=_probe_stdout("3.12.9")))

    with pytest.raises(EvalEnvironmentError, match="failed verification"):
        _ensure(manager)

    assert canary.read_text() == "preserve me"


def test_broken_existing_bundle_is_an_actionable_error(state_dir, monkeypatch):
    manager = EvalEnvironmentManager(state_dir=state_dir)
    _make_valid_env(manager)
    monkeypatch.setattr(env_module, "_run", _fake_installer(manager.env_path, fail_smoke=True))

    with pytest.raises(EvalEnvironmentError, match="bundled-library smoke test failed"):
        _ensure(manager)


def test_reuse_probe_os_error_is_an_actionable_error(state_dir, monkeypatch):
    manager = EvalEnvironmentManager(state_dir=state_dir)
    _make_valid_env(manager)

    def missing(cmd, timeout):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(env_module, "_run", missing)

    with pytest.raises(EvalEnvironmentError, match="interpreter probe could not run"):
        _ensure(manager)


def test_bundle_change_creates_a_separate_environment(state_dir, monkeypatch):
    monkeypatch.setattr(EvalEnvironmentManager, "_bundle_source", lambda self: b"old")
    old_manager = EvalEnvironmentManager(state_dir=state_dir)
    monkeypatch.setattr(env_module, "_run", _fake_installer(old_manager.env_path))
    monkeypatch.setattr(env_module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    old_info = _ensure(old_manager)
    assert old_info.env_path is not None

    monkeypatch.setattr(EvalEnvironmentManager, "_bundle_source", lambda self: b"new")
    new_manager = EvalEnvironmentManager(state_dir=state_dir)
    monkeypatch.setattr(env_module, "_run", _fake_installer(new_manager.env_path))
    new_info = _ensure(new_manager)

    assert new_info.provisioned_now is True
    assert new_info.env_path != old_info.env_path
    assert old_info.env_path.exists()


def test_extra_packages_extend_bundle_and_hash(state_dir):
    base = EvalEnvironmentManager(state_dir=state_dir)
    extended = EvalEnvironmentManager(state_dir=state_dir, extra_packages=["scipy==1.14.1"])

    assert base.env_path != extended.env_path
    assert "scipy==1.14.1" in extended._bundle_names()


def test_legacy_python_version_produces_deprecation_note(state_dir, monkeypatch):
    manager = EvalEnvironmentManager(state_dir=state_dir, legacy_python_version="3.12")
    _make_valid_env(manager)
    monkeypatch.setattr(env_module, "_run", lambda cmd, timeout: _ok(cmd, timeout, stdout=_probe_stdout()))
    monkeypatch.setattr(env_module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    info = _ensure(manager)

    assert "deprecated and ignored" in (info.note or "")
    assert "eval_python_path" in (info.note or "")
