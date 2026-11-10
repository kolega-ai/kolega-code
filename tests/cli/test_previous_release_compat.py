"""Offline failure-detection tests for the previous-release compatibility check."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest
from packaging.version import Version

from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import SettingsStore
from kolega_code.llm.models import Message

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture_module = _load("previous_release_compat_fixture")
check = _load("check_previous_release_compat")
worker = _load("previous_release_compat_worker")


def _wheel(version: str, *, yanked: bool = False) -> dict[str, Any]:
    filename = f"kolega_code-{version}-py3-none-any.whl"
    return {
        "filename": filename,
        "packagetype": "bdist_wheel",
        "yanked": yanked,
        "url": f"https://files.pythonhosted.org/packages/{filename}",
        "digests": {"sha256": "a" * 64},
    }


def _pypi(monkeypatch: pytest.MonkeyPatch, releases: dict[str, list[dict[str, Any]]]) -> None:
    data = json.dumps({"releases": releases}).encode()
    monkeypatch.setattr(check, "urlopen", lambda *args, **kwargs: io.BytesIO(data))


def test_discovery_uses_latest_non_yanked_stable_release(monkeypatch: pytest.MonkeyPatch) -> None:
    _pypi(
        monkeypatch,
        {
            "1.9": [_wheel("1.9")],
            "1.10": [_wheel("1.10")],
            "2.0rc1": [_wheel("2.0rc1")],
            "3.0": [_wheel("3.0", yanked=True)],
            "not-a-version": [_wheel("not-a-version")],
        },
    )
    version, wheel = check.discover_release()
    assert version == "1.10"
    assert wheel["filename"] == "kolega_code-1.10-py3-none-any.whl"


def test_release_discovery_excludes_candidate_and_newer_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    _pypi(
        monkeypatch,
        {
            "1.8": [_wheel("1.8")],
            "1.9": [_wheel("1.9")],
            "1.10": [_wheel("1.10")],
            "1.11": [_wheel("1.11")],
        },
    )
    # The candidate may already be on PyPI when a preflight is rerun.
    assert check.discover_release(before=Version("1.10"))[0] == "1.9"
    assert check.discover_release()[0] == "1.11"


def test_release_discovery_fails_when_no_older_release_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    _pypi(monkeypatch, {"1.0": [_wheel("1.0")], "2.0": [_wheel("2.0")]})
    with pytest.raises(ValueError, match="non-yanked stable release"):
        check.discover_release(before=Version("1.0"))


@pytest.mark.parametrize("version", ["1.0", "2.0"])
def test_release_preflight_rejects_not_older_installed_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, version: str
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.0"\n', encoding="utf-8")
    monkeypatch.setenv("KOLEGA_SCRATCHPAD", str(tmp_path))
    monkeypatch.setattr(check, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["check", "--previous-python", str(tmp_path / "python"), "--release-preflight"])
    monkeypatch.setattr(check, "installed_version", lambda *args: version)
    with pytest.raises(ValueError, match="must be older than candidate"):
        check.main()


def test_discovery_fails_if_latest_release_has_no_safe_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    source_only = {"filename": "kolega_code-2.0.tar.gz", "packagetype": "sdist", "yanked": False}
    _pypi(monkeypatch, {"1.0": [_wheel("1.0")], "2.0": [source_only]})
    with pytest.raises(ValueError, match="universal wheel"):
        check.discover_release()


def test_discovery_rejects_untrusted_wheel_host(monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = _wheel("1.0")
    wheel["url"] = "https://example.com/kolega_code-1.0-py3-none-any.whl"
    _pypi(monkeypatch, {"1.0": [wheel]})
    with pytest.raises(ValueError, match="download host"):
        check.discover_release()


def test_wheel_hash_mismatch_is_fatal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(check, "urlopen", lambda *args, **kwargs: io.BytesIO(b"not the wheel"))
    with pytest.raises(ValueError, match="SHA-256"):
        check.download_wheel(_wheel("1.0"), tmp_path)


def test_previous_process_environment_is_isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-secret")
    monkeypatch.setenv("UV_EXCLUDE_NEWER", "1 week")
    monkeypatch.setenv("PYTHONPATH", "/fake/checkout")
    monkeypatch.setenv("PIP_INDEX_URL", "https://example.com")
    env = check.clean_environment(tmp_path)
    assert "OPENAI_API_KEY" not in env
    assert "UV_EXCLUDE_NEWER" not in env
    assert "PYTHONPATH" not in env
    assert "PIP_INDEX_URL" not in env
    assert env["HOME"] == str(tmp_path)


def test_checkout_and_mismatched_distribution_imports_are_rejected(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    prefix = checkout / ".venv"
    installed = prefix / "site-packages/kolega_code/__init__.py"
    worker.verify_origin(installed, installed, prefix, checkout)
    source = checkout / "kolega_code/__init__.py"
    with pytest.raises(RuntimeError, match="not distribution module"):
        worker.verify_origin(source, installed, prefix, checkout)
    with pytest.raises(RuntimeError, match="outside isolated"):
        worker.verify_origin(source, source, prefix, checkout)
    with pytest.raises(RuntimeError, match="checkout"):
        worker.verify_origin(source, source, checkout, checkout)


def test_previous_reader_must_replay_original_history_in_order() -> None:
    expected = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    worker.verify_original_history(expected, expected)
    for actual in ([], list(reversed(expected))):
        with pytest.raises(RuntimeError, match="did not replay"):
            worker.verify_original_history(actual, expected)


def _simulate_writer(root: Path, manifest: dict[str, Any]) -> None:
    settings_store = SettingsStore(root)
    settings = settings_store.load()
    settings.active_theme = "compat-previous-theme"
    settings.agent_models["building"]["model"] = "compat-previous-model"
    settings_store.save(settings)
    sessions = SessionStore(root)
    recorder = sessions.recorder(manifest["session_id"], recover=False)
    recorder.start_turn(Message(role="user", content="previous-release user"))
    recorder.record_assistant(Message(role="assistant", content="previous-release assistant"))
    recorder.finish_turn("completed")
    sessions.release_session_locks()


@pytest.fixture(scope="module")
def producer_inventory() -> frozenset[str]:
    """Parse the package once: the producer inventory does not change during a run.

    ``_producer_inventory`` AST-scans every module in the package (~3.5 s here),
    and ``create_fixture`` calls it for each roundtrip test.
    """
    return frozenset(fixture_module._producer_inventory())


@pytest.fixture
def completed_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, producer_inventory: frozenset[str]
) -> tuple[Path, dict[str, Any]]:
    monkeypatch.setattr(fixture_module, "_producer_inventory", lambda: producer_inventory)
    manifest = fixture_module.create_fixture(tmp_path)
    _simulate_writer(tmp_path, manifest)
    return tmp_path, manifest


def test_roundtrip_checks_current_event_coverage(completed_roundtrip: tuple[Path, dict[str, Any]]) -> None:
    root, manifest = completed_roundtrip
    assert fixture_module._producer_inventory() <= set(manifest["event_types"])
    assert {f"ui.{name}" for name in fixture_module._ui_types()} <= set(manifest["event_types"])
    fixture_module.verify_after_previous(root, manifest)


def test_rewritten_journal_bytes_fail(completed_roundtrip: tuple[Path, dict[str, Any]]) -> None:
    root, manifest = completed_roundtrip
    path = SessionStore(root).events_path_for(manifest["session_id"])
    data = path.read_bytes()
    path.write_bytes(b" " + data)  # Still JSON-decodable, but no longer the original bytes.
    with pytest.raises(AssertionError, match="rewritten or truncated"):
        fixture_module.verify_after_previous(root, manifest)


def test_lost_unknown_settings_fail(completed_roundtrip: tuple[Path, dict[str, Any]]) -> None:
    root, manifest = completed_roundtrip
    path = root / "settings.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    del saved["agent_models"]["building"]["compat_future_nested"]
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(AssertionError, match="Settings changed"):
        fixture_module.verify_after_previous(root, manifest)


def test_missing_previous_turn_completion_fails(completed_roundtrip: tuple[Path, dict[str, Any]]) -> None:
    root, manifest = completed_roundtrip
    path = SessionStore(root).events_path_for(manifest["session_id"])
    lines = path.read_bytes().splitlines(keepends=True)
    assert json.loads(lines[-1])["type"] == "turn.completed"
    path.write_bytes(b"".join(lines[:-1]))
    with pytest.raises(AssertionError, match="ordered user, assistant and completion"):
        fixture_module.verify_after_previous(root, manifest)


def test_new_unreviewed_semantic_type_fails_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fixture_module, "_SEMANTIC_TYPES", fixture_module._SEMANTIC_TYPES | {"new.event"})
    with pytest.raises(AssertionError, match="semantic event inventory"):
        fixture_module._producer_inventory()


def test_old_process_failure_is_not_silenced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KOLEGA_SCRATCHPAD", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["check", "--previous-python", str(tmp_path / "python")])
    monkeypatch.setattr(check, "installed_version", lambda *args: "1.0")
    monkeypatch.setattr(check, "create_fixture", lambda root: {"session_id": "fixture"})

    def fail(*args: Any, **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, "previous-release worker")

    monkeypatch.setattr(check.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        check.main()
