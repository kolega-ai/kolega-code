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
from packaging.tags import Tag, compatible_tags, cpython_tags, mac_platforms
from packaging.utils import parse_wheel_filename

LOCK_PATH = Path(__file__).resolve().parents[1] / "uv.lock"

# Interpreters uv realistically selects for `uv tool install kolega-code`.
# 3.14 is excluded: onnxruntime publishes no sdist and no Intel-mac wheels for
# it, so that combination fails resolution outright and predates this guard.
PYTHONS = [(3, 11), (3, 12), (3, 13)]


def _intel_mac_env(python: tuple[int, int]) -> dict[str, str]:
    version = f"{python[0]}.{python[1]}"
    return {
        "implementation_name": "cpython",
        "implementation_version": f"{version}.0",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_release": "24.6.0",
        "platform_system": "Darwin",
        "python_full_version": f"{version}.0",
        "python_version": version,
        "sys_platform": "darwin",
        "extra": "",
    }


def _intel_mac_tags(python: tuple[int, int]) -> set[Tag]:
    platforms = list(mac_platforms((15, 0), "x86_64"))
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


@pytest.mark.parametrize("python", PYTHONS, ids=lambda p: f"cp{p[0]}{p[1]}")
def test_runtime_dependencies_have_intel_mac_wheels(python: tuple[int, int]) -> None:
    env = _intel_mac_env(python)
    supported = _intel_mac_tags(python)

    broken = [
        f"{entry['name']}=={entry['version']}"
        for entry in _reachable_runtime_entries(env)
        # Sdist-only packages are pure source for every platform; a native one
        # would already fail everywhere, including CI.
        if entry.get("wheels") and not _has_installable_wheel(entry, supported)
    ]

    assert not broken, (
        "These locked runtime dependencies ship wheels, but none installable on "
        f"Intel macOS (CPython {python[0]}.{python[1]}), so `uv tool install "
        f"kolega-code` there falls back to a native source build: {broken}. "
        "Pin an older wheel-bearing release behind a platform marker in "
        "pyproject.toml, like the cryptography pin."
    )
