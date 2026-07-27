#!/usr/bin/env python3
"""Fail if tracked files name closed-source host internals.

This repository is public. The host applications that embed this package are not,
and their internal package names and architecture must not appear in anything we
publish — docs, migration guides, changelogs, docstrings, or tests.

Guidance about integrating a host is written generically instead: "hosts
implementing ``AgentConnectionManager``", "a database-backed
``SessionEventStore``", "an atomic sequence increment". Host-specific mapping
belongs in an untracked companion document.

Product names that already appear in the published public API surface are not in
scope; this checks for internal identifiers only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

#: Internal package names of the closed-source host applications.
FORBIDDEN = ("kolega_studio", "kolega_dev")

#: This file necessarily contains the strings it forbids.
SELF = Path(__file__).name

#: Generated lock content can legitimately mention arbitrary package names.
SKIP_SUFFIXES = (".lock",)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        check=True,
        text=True,
    )
    return [Path(name) for name in result.stdout.split("\0") if name]


def main(argv: list[str]) -> int:
    candidates = [Path(arg) for arg in argv] or tracked_files()
    violations: list[str] = []
    for path in candidates:
        if path.name == SELF or path.suffix in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for needle in FORBIDDEN:
            for number, line in enumerate(text.splitlines(), start=1):
                if needle in line:
                    violations.append(f"{path}:{number}: mentions {needle}")

    if violations:
        print("Tracked files must not name closed-source host internals:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        print(
            "\nWrite host guidance generically, and keep host-specific detail in an untracked document.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
