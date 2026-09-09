"""Guard the end-user install against dependencies that drop platform wheels.

CI resolves and syncs on Linux/arm64 runners, so a dependency that stops
publishing wheels for another supported platform never fails here — it fails on
a user's machine as a source build that needs Rust, OpenSSL, or a C toolchain
(cryptography >= 49 dropping Intel-mac wheels broke the curl installer exactly
this way). This test walks the runtime dependency graph in uv.lock as it
resolves for such a platform and fails when a wheel-shipping package has no
installable wheel there, so the gap surfaces at lock-bump time instead.
"""

from __future__ import annotations

import tomllib
from collections import deque
from pathlib import Path
from typing import Any

import pytest
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.tags import Tag, compatible_tags, cpython_tags, mac_platforms
from packaging.utils import parse_wheel_filename

LOCK_PATH = Path(__file__).resolve().parents[1] / "uv.lock"
PROJECT_PATH = LOCK_PATH.with_name("pyproject.toml")

# Interpreters uv realistically selects for `uv tool install kolega-code`.
# 3.14 is excluded: onnxruntime publishes no sdist and no Intel-mac wheels for
# it, so that combination fails resolution outright and predates this guard.
# ONNX 1.19.2 supports Monterey only through Python 3.12. Python 3.13
# uses newer ONNX wheels requiring macOS 13+, so it is covered on macOS 15.
INTEL_MAC_TARGETS = [
    pytest.param((3, 11), (12, 0), id="macos12.0-cp311"),
    pytest.param((3, 12), (12, 0), id="macos12.0-cp312"),
    pytest.param((3, 11), (15, 0), id="macos15.0-cp311"),
    pytest.param((3, 12), (15, 0), id="macos15.0-cp312"),
    pytest.param((3, 13), (15, 0), id="macos15.0-cp313"),
]


def _intel_mac_env(python: tuple[int, int], macos: tuple[int, int]) -> dict[str, str]:
    version = f"{python[0]}.{python[1]}"
    return {
        "implementation_name": "cpython",
        "implementation_version": f"{version}.0",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_release": f"{macos[0] + 9}.0.0",
        "platform_system": "Darwin",
        "python_full_version": f"{version}.0",
        "python_version": version,
        "sys_platform": "darwin",
        "extra": "",
    }


def _intel_mac_tags(python: tuple[int, int], macos: tuple[int, int]) -> set[Tag]:
    platforms = list(mac_platforms(macos, "x86_64"))
    return set(cpython_tags(python, None, platforms)) | set(compatible_tags(python, None, platforms))


def _entry_applies(entry: dict[str, Any], env: dict[str, str]) -> bool:
    markers = entry.get("resolution-markers")
    if not markers:
        return True
    return any(Marker(marker).evaluate(env) for marker in markers)


def _reachable_runtime_entries(env: dict[str, str]) -> list[dict[str, Any]]:
    """Entries reachable from kolega-code's runtime dependencies under env."""
    with LOCK_PATH.open("rb") as fh:
        lock = tomllib.load(fh)

    entries_by_name: dict[str, list[dict[str, Any]]] = {}
    for entry in lock["package"]:
        entries_by_name.setdefault(entry["name"], []).append(entry)

    reached: dict[tuple[str, str], dict[str, Any]] = {}
    seen: set[tuple[str, tuple[str, ...]]] = set()
    queue: deque[tuple[str, tuple[str, ...]]] = deque([("kolega-code", ())])
    while queue:
        name, extras = queue.popleft()
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        for entry in entries_by_name.get(name, []):
            if not _entry_applies(entry, env):
                continue
            if "editable" not in entry["source"]:
                reached[(entry["name"], entry["version"])] = entry
            edges = list(entry.get("dependencies", []))
            optional = entry.get("optional-dependencies", {})
            for extra in extras:
                edges.extend(optional.get(extra, []))
            for edge in edges:
                marker = edge.get("marker")
                if marker and not Marker(marker).evaluate(env):
                    continue
                queue.append((edge["name"], tuple(edge.get("extra", ()))))
    return list(reached.values())


def _has_installable_wheel(entry: dict[str, Any], supported: set[Tag]) -> bool:
    for wheel in entry["wheels"]:
        filename = wheel["url"].rsplit("/", 1)[-1]
        _, _, _, tags = parse_wheel_filename(filename)
        if not tags.isdisjoint(supported):
            return True
    return False


@pytest.mark.parametrize("python,macos", INTEL_MAC_TARGETS)
def test_runtime_dependencies_have_intel_mac_wheels(python: tuple[int, int], macos: tuple[int, int]) -> None:
    env = _intel_mac_env(python, macos)
    supported = _intel_mac_tags(python, macos)

    broken = [
        f"{entry['name']}=={entry['version']}"
        for entry in _reachable_runtime_entries(env)
        # Sdist-only packages are pure source for every platform; a native one
        # would already fail everywhere, including CI.
        if entry.get("wheels") and not _has_installable_wheel(entry, supported)
    ]

    assert not broken, (
        "These locked runtime dependencies ship wheels, but none installable on "
        f"Intel macOS {macos[0]}.{macos[1]} (CPython {python[0]}.{python[1]}), so `uv tool install "
        f"kolega-code` there falls back to a native source build: {broken}. "
        "Pin an older wheel-bearing release behind a platform marker in "
        "pyproject.toml, like the cryptography pin."
    )


def test_pdfium_pin_is_published_and_locked() -> None:
    """A transitive lock pin alone does not constrain fresh PyPI installs."""
    with PROJECT_PATH.open("rb") as fh:
        project = tomllib.load(fh)
    requirements = [Requirement(value) for value in project["project"]["dependencies"]]
    pdfium = [requirement for requirement in requirements if requirement.name == "pypdfium2"]
    assert len(pdfium) == 1
    assert str(pdfium[0].specifier) == "==5.11.0"
    assert pdfium[0].marker is None
    assert pdfium[0].url is None

    with LOCK_PATH.open("rb") as fh:
        lock = tomllib.load(fh)
    root = next(entry for entry in lock["package"] if entry["name"] == "kolega-code")
    assert {"name": "pypdfium2"} in root["dependencies"]
    assert {"name": "pypdfium2", "specifier": "==5.11.0"} in root["metadata"]["requires-dist"]
    assert [entry["version"] for entry in lock["package"] if entry["name"] == "pypdfium2"] == ["5.11.0"]


@pytest.mark.parametrize("minimum_macos,expected", [(12, True), (13, False)])
def test_monterey_wheel_baseline(minimum_macos: int, expected: bool) -> None:
    entry = {"wheels": [{"url": f"example-1.0-py3-none-macosx_{minimum_macos}_0_x86_64.whl"}]}
    assert _has_installable_wheel(entry, _intel_mac_tags((3, 11), (12, 0))) is expected


@pytest.mark.parametrize(
    "python,system,machine,expected",
    [
        ((3, 11), "darwin", "x86_64", True),
        ((3, 12), "darwin", "x86_64", True),
        ((3, 13), "darwin", "x86_64", False),
        ((3, 14), "darwin", "x86_64", False),
        ((3, 11), "darwin", "arm64", False),
        ((3, 12), "linux", "x86_64", False),
        ((3, 12), "win32", "AMD64", False),
    ],
)
def test_legacy_onnx_pin_scope(python: tuple[int, int], system: str, machine: str, expected: bool) -> None:
    with PROJECT_PATH.open("rb") as fh:
        project = tomllib.load(fh)
    requirements = [Requirement(value) for value in project["project"]["dependencies"]]
    legacy = [req for req in requirements if req.name == "onnxruntime" and str(req.specifier) == "==1.19.2"]
    assert len(legacy) == 1
    assert legacy[0].marker is not None
    env = _intel_mac_env(python, (12, 0))
    env.update(sys_platform=system, platform_machine=machine)
    assert legacy[0].marker.evaluate(env) is expected


@pytest.mark.parametrize("python", [(3, 11), (3, 12)])
def test_monterey_locks_compatible_onnx(python: tuple[int, int]) -> None:
    entries = _reachable_runtime_entries(_intel_mac_env(python, (12, 0)))
    assert [entry["version"] for entry in entries if entry["name"] == "onnxruntime"] == ["1.19.2"]
