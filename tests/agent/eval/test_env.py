"""Unit tests for the dedicated eval environment manager (hermetic, mocked I/O)."""

import asyncio
import json
import subprocess
import sys

import pytest

from kolega_code.agent.eval import env as env_module
from kolega_code.agent.eval.env import EvalEnvironmentManager


def _ok(args, timeout):
    return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")


def _fail(args, timeout):
    return subprocess.CompletedProcess(args, 1, stdout="", stderr="boom")


def _fake_installer(env_path):
    """A fake _run that 'creates' the venv python on the venv step."""

    def run(cmd, timeout):
        if "venv" in cmd:
            python = env_module._venv_python(env_path)
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("#!/bin/sh\n")
        return _ok(cmd, timeout)

    return run


def _ensure(manager):
    return asyncio.run(manager.ensure())


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "state"


def test_override_python_skips_provisioning(state_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(env_module, "_run", lambda cmd, timeout: calls.append(cmd) or _ok(cmd, timeout))
    monkeypatch.setattr(env_module.shutil, "which", lambda name: f"/usr/bin/{name}")

    manager = EvalEnvironmentManager(state_dir=state_dir, override_python=sys.executable)
    info = _ensure(manager)

    assert info.python == sys.executable
    assert info.degraded is False
    assert info.bundled is False
    assert "custom interpreter" in (info.note or "")
    assert info.pip_install_cmd == [sys.executable, "-m", "pip", "install"]
    # Only the --version probe ran; no venv/install commands.
    assert calls == [[sys.executable, "--version"]]


def test_provisions_with_uv_and_writes_manifest(state_dir, monkeypatch):
    calls = []

    manager = EvalEnvironmentManager(state_dir=state_dir, python_version="3.12")

    def run(cmd, timeout):
        calls.append(cmd)
        return _fake_installer(manager.env_path)(cmd, timeout)

    monkeypatch.setattr(env_module, "_run", run)
    monkeypatch.setattr(env_module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    info = _ensure(manager)

    assert info.bundled is True
    assert info.degraded is False
    assert info.provisioned_now is True
    assert "one-time setup" in (info.note or "")
    assert calls[0][:3] == ["/usr/bin/uv", "venv", str(manager.env_path)]
    assert "--python" in calls[0]
    assert calls[1][:3] == ["/usr/bin/uv", "pip", "install"]
    assert info.pip_install_cmd[0] == "/usr/bin/uv"
    assert "numpy" in " ".join(info.bundle)

    manifest = json.loads((manager.env_path / env_module.MANIFEST_NAME).read_text())
    assert manifest["installer"] == "uv"
    assert manifest["python_version"] == "3.12"


def test_provisions_with_venv_when_uv_missing(state_dir, monkeypatch):
    calls = []

    def run(cmd, timeout):
        calls.append(cmd)
        return _fake_installer(env_path)(cmd, timeout)

    manager = EvalEnvironmentManager(state_dir=state_dir)
    env_path = manager.env_path
    monkeypatch.setattr(env_module, "_run", run)
    monkeypatch.setattr(env_module.shutil, "which", lambda name: None)

    info = _ensure(manager)

    assert calls[0] == [sys.executable, "-m", "venv", str(manager.env_path)]
    assert calls[1][1:4] == ["-m", "pip", "install"]
    assert info.pip_install_cmd[1:] == ["-m", "pip", "install"]


def test_degrades_to_host_interpreter_on_failure(state_dir, monkeypatch):
    monkeypatch.setattr(env_module, "_run", _fail)
    monkeypatch.setattr(env_module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    manager = EvalEnvironmentManager(state_dir=state_dir)
    info = _ensure(manager)

    assert info.degraded is True
    assert info.python == sys.executable
    assert info.pip_install_cmd == []
    assert "host interpreter" in (info.note or "")
    assert not manager.env_path.exists()


def test_reuses_matching_manifest_without_provisioning(state_dir, monkeypatch):
    monkeypatch.setattr(env_module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    manager = EvalEnvironmentManager(state_dir=state_dir)

    python = env_module._venv_python(manager.env_path)
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    manager._write_manifest("uv")

    def no_run(cmd, timeout):
        raise AssertionError("must not provision when the manifest matches")

    monkeypatch.setattr(env_module, "_run", no_run)
    info = _ensure(manager)

    assert info.bundled is True
    assert info.provisioned_now is False
    assert info.python == str(python)


def test_reprovisions_on_bundle_hash_change(state_dir, monkeypatch):
    calls = []

    manager = EvalEnvironmentManager(state_dir=state_dir)

    def run(cmd, timeout):
        calls.append(cmd)
        return _fake_installer(manager.env_path)(cmd, timeout)

    monkeypatch.setattr(env_module, "_run", run)
    monkeypatch.setattr(env_module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    python = env_module._venv_python(manager.env_path)
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    # A manifest whose bundle hash does not match the current bundle.
    (manager.env_path / env_module.MANIFEST_NAME).write_text(
        json.dumps({"python_version": "3.12", "bundle_hash": "stale", "installer": "uv"})
    )

    info = _ensure(manager)
    assert info.provisioned_now is True
    assert any(cmd[:2] == ["/usr/bin/uv", "venv"] for cmd in calls)


def test_extra_packages_extend_bundle_and_hash(state_dir):
    base = EvalEnvironmentManager(state_dir=state_dir)
    extended = EvalEnvironmentManager(state_dir=state_dir, extra_packages=["scipy==1.14.1"])

    assert base._bundle_hash() != extended._bundle_hash()
    assert "scipy==1.14.1" in extended._bundle_names()
