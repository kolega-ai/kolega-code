"""Regression test: the model-facing tool-definition wire bytes are frozen.

Any change to a tool's name, description, input schema, input kind, freeform
format, or registry order fails here. That is deliberate — tool definitions
are tuned prompt surface, so a change must be an explicit decision recorded in
the committed snapshot, never a side effect. To accept an intentional change::

    KOLEGA_UPDATE_TOOL_DEFINITION_SNAPSHOT=1 uv run pytest tests/agent/test_tool_definition_freeze.py

and commit the regenerated fixture together with the change that caused it.
"""

import difflib
import itertools
import os

from .tool_definition_freeze import SNAPSHOT_PATH, build_snapshot, render_snapshot

_UPDATE_ENV = "KOLEGA_UPDATE_TOOL_DEFINITION_SNAPSHOT"


def test_wire_definitions_match_snapshot(tmp_path):
    rendered = render_snapshot(build_snapshot(tmp_path / "freeze"))

    if os.environ.get(_UPDATE_ENV) == "1":
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(rendered, encoding="utf-8")

    assert SNAPSHOT_PATH.is_file(), f"Missing snapshot fixture {SNAPSHOT_PATH}; regenerate with {_UPDATE_ENV}=1."
    expected = SNAPSHOT_PATH.read_text(encoding="utf-8")
    if rendered != expected:
        diff = "\n".join(
            itertools.islice(
                difflib.unified_diff(
                    expected.splitlines(),
                    rendered.splitlines(),
                    fromfile="committed snapshot",
                    tofile="current build",
                    lineterm="",
                ),
                200,
            )
        )
        raise AssertionError(
            "Tool-definition wire bytes changed. If intentional, regenerate the snapshot with "
            f"{_UPDATE_ENV}=1 and commit it with this change.\n{diff}"
        )


def test_snapshot_is_independent_of_project_path(tmp_path):
    """No environment or filesystem detail may leak into the wire definitions."""
    first = render_snapshot(build_snapshot(tmp_path / "one"))
    second = render_snapshot(build_snapshot(tmp_path / "two" / "nested"))
    assert first == second
