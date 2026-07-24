#!/usr/bin/env python3
"""Count lines of code for kolega-code.

Reproducible LOC metric: only git-tracked files are counted, so .venv/,
dist/, caches, and other untracked artifacts never leak into the numbers.

Python files get a blank/comment/code breakdown. A "comment" is a line
whose first non-whitespace character is '#'; docstrings count as code
(same convention as `cloc --docstring-as-code`). All other tracked files
(markdown, YAML, benchmark fixture corpora, ...) contribute raw lines
under "other lines" with no comment analysis.

Usage:
    uv run scripts/count_loc.py
    python3 scripts/count_loc.py
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Top-level areas reported individually; anything else lands in "other".
AREAS = ("kolega_code", "tests", "scripts", "benchmarks", "docs")


@dataclass
class AreaStats:
    files: int = 0
    py_files: int = 0
    blank: int = 0
    comment: int = 0
    code: int = 0
    other_lines: int = 0
    per_ext: dict[str, int] = field(default_factory=dict)
    binary_files: list[str] = field(default_factory=list)

    @property
    def total_lines(self) -> int:
        return self.blank + self.comment + self.code + self.other_lines


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return [p for p in result.stdout.decode("utf-8").split("\0") if p]


def count_python(data: bytes) -> tuple[int, int, int]:
    """Return (blank, comment, code) line counts for a Python file."""
    text = data.decode("utf-8", errors="replace")
    blank = comment = code = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            blank += 1
        elif stripped.startswith("#"):
            comment += 1
        else:
            code += 1
    return blank, comment, code


def is_binary(data: bytes) -> bool:
    """Heuristic: a file is binary if its first 8 KiB contain a NUL byte."""
    return b"\0" in data[:8192]


def main() -> int:
    stats = {area: AreaStats() for area in AREAS}
    stats["other"] = AreaStats()

    for rel in tracked_files():
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        area = rel.split("/", 1)[0]
        bucket = stats[area] if area in stats else stats["other"]
        bucket.files += 1

        data = path.read_bytes()
        if is_binary(data):
            bucket.binary_files.append(rel)
            continue

        ext = path.suffix.lstrip(".").lower() or "(none)"
        if ext == "py":
            bucket.py_files += 1
            blank, comment, code = count_python(data)
            bucket.blank += blank
            bucket.comment += comment
            bucket.code += code
        else:
            lines = len(data.decode("utf-8", errors="replace").splitlines())
            bucket.other_lines += lines
            bucket.per_ext[ext] = bucket.per_ext.get(ext, 0) + lines

    header = (
        f"{'area':<14} {'files':>6} {'py files':>9} {'blank':>8} {'comment':>8} {'code':>9} {'other':>8} {'total':>9}"
    )
    print("kolega-code lines of code (git-tracked files only)\n")
    print(header)
    print("-" * len(header))
    totals = AreaStats()
    for area, s in stats.items():
        if s.files == 0:
            continue
        print(
            f"{area:<14} {s.files:>6} {s.py_files:>9} {s.blank:>8} {s.comment:>8} "
            f"{s.code:>9} {s.other_lines:>8} {s.total_lines:>9}"
        )
        totals.files += s.files
        totals.py_files += s.py_files
        totals.blank += s.blank
        totals.comment += s.comment
        totals.code += s.code
        totals.other_lines += s.other_lines
        for ext, lines in s.per_ext.items():
            totals.per_ext[ext] = totals.per_ext.get(ext, 0) + lines
        totals.binary_files.extend(s.binary_files)
    print("-" * len(header))
    print(
        f"{'TOTAL':<14} {totals.files:>6} {totals.py_files:>9} {totals.blank:>8} {totals.comment:>8} "
        f"{totals.code:>9} {totals.other_lines:>8} {totals.total_lines:>9}"
    )

    py_total = totals.blank + totals.comment + totals.code
    print(f"\nPython code (excl. blank/comment): {totals.code:,}")
    print(f"Python total lines:                {py_total:,}")
    print(f"All tracked lines:                 {totals.total_lines:,}")

    if totals.per_ext:
        top = sorted(totals.per_ext.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("\nTop non-Python extensions by lines:")
        for ext, lines in top:
            print(f"  .{ext:<10} {lines:>8,}")

    if totals.binary_files:
        print(f"\nSkipped {len(totals.binary_files)} binary file(s) (no line counts):")
        for rel in totals.binary_files:
            print(f"  {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
