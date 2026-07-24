"""Consistency checks for the managed eval environment's dependency bundle."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PROJECT_ROOT = Path(__file__).parents[3]
BUNDLE_INPUT = PROJECT_ROOT / "kolega_code" / "agent" / "eval" / "bundle-requirements.in"
BUNDLE_LOCK = PROJECT_ROOT / "kolega_code" / "agent" / "eval" / "bundle-requirements.txt"


def _exact_pins(requirements: list[str], *, source: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for value in requirements:
        requirement = Requirement(value)
        specifiers = list(requirement.specifier)
        assert len(specifiers) == 1, f"{source}: {requirement.name} must have one exact pin"
        specifier = specifiers[0]
        assert specifier.operator == "==" and not specifier.version.endswith(".*"), (
            f"{source}: {requirement.name} must use an exact == pin"
        )
        pins[canonicalize_name(requirement.name)] = specifier.version
    return pins


def _requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_bundle_lock_contains_exact_direct_pins() -> None:
    direct = _exact_pins(_requirement_lines(BUNDLE_INPUT), source=BUNDLE_INPUT.name)
    locked = _exact_pins(_requirement_lines(BUNDLE_LOCK), source=BUNDLE_LOCK.name)

    assert direct.items() <= locked.items()


def test_bundle_pins_match_shared_project_dependencies() -> None:
    direct = _exact_pins(_requirement_lines(BUNDLE_INPUT), source=BUNDLE_INPUT.name)
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = _exact_pins(pyproject["project"]["dependencies"], source="pyproject.toml")

    shared = direct.keys() & project.keys()
    assert shared, "expected at least one dependency shared by the project and eval bundle"
    assert {name: direct[name] for name in shared} == {name: project[name] for name in shared}
