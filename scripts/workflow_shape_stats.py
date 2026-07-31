#!/usr/bin/env python3
"""Classify on-disk gigacode workflow scripts by orchestration shape.

Produced the "What models actually choose" table in how-gigacode-works.md.
Keep the method versioned so the numbers are re-derivable (and diffable) when
the corpus grows or the markers change.

Method and caveats, exactly as disclosed in the doc:
  - Scripts are read from <state-dir>/workflows/<run_id>/script.py and deduped
    by content hash; "substantial" means >=3 journaled agent calls and >1.5KB.
  - Shape detection is keyword-marker matching over the source, so shape
    percentages are approximate and categories OVERLAP (one script can count
    toward several rows). Primitive usage is exact substring matching.
  - The corpus is whatever machine this runs on — usage, not fleet telemetry.

Usage: python scripts/workflow_shape_stats.py [state-dir]
       (default state dir: platform default, honoring KOLEGA_CODE_STATE_DIR)
"""

from __future__ import annotations

import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

MIN_CALLS = 3
MIN_BYTES = 1500

SHAPE_MARKERS = {
    "adversarial-verify": r"refut|skeptic|adversar|challeng|disprove",
    "judge-panel": r"judge|panel|score.*attempt|tournament",
    "loop-until-dry": r"dry|no.?new|consecutive|stops? growing|nothing new",  # also requires a while loop
    "budget-gated-loop": r"budget\.remaining|budget\.total",
    "shard-and-sweep": r"shard|workstream|file.?sets?|disjoint",
    "cross-cut-matrix": r"dimension|cross.?cut|matrix|lens",
    "surface-map": r"surface|invariant|map the|mapping",
    "synthesis-gate": r"synthes|final report|merge.*finding|dedup",
    "uses-coder-agents": r"agent_type\s*=\s*[\"']cod",
}

PRIMITIVE_MARKERS = {
    "parallel": "parallel(",
    "pipeline": "pipeline(",
    "schema": "schema=",
    "while-loop": None,  # regex below
    "model_override": "model_override",
    "label": "label=",
    "phase": "phase(",
}


def default_state_dir() -> Path:
    from kolega_code.cli.session_store import default_state_dir as _dsd

    return _dsd()


def collect_scripts(state_dir: Path) -> dict[str, dict]:
    scripts: dict[str, dict] = {}
    root = state_dir / "workflows"
    if not root.exists():
        return scripts
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        source_path = run_dir / "script.py"
        if not source_path.exists():
            continue
        source = source_path.read_text(errors="replace")
        journal = run_dir / "journal.jsonl"
        calls = 0
        if journal.exists():
            calls = sum(1 for line in journal.read_text().splitlines() if line.strip())
        digest = hashlib.sha1(source.encode()).hexdigest()
        record = scripts.setdefault(digest, {"source": source, "max_calls": 0, "runs": 0})
        record["runs"] += 1
        record["max_calls"] = max(record["max_calls"], calls)
    return scripts


def classify(source: str) -> set[str]:
    lowered = source.lower()
    shapes = set()
    for shape, pattern in SHAPE_MARKERS.items():
        if not re.search(pattern, lowered):
            continue
        if shape == "loop-until-dry" and not re.search(r"\bwhile\b", lowered):
            continue
        shapes.add(shape)
    return shapes


def main() -> None:
    state_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_state_dir()
    scripts = collect_scripts(state_dir)
    substantial = {
        digest: record
        for digest, record in scripts.items()
        if record["max_calls"] >= MIN_CALLS and len(record["source"]) > MIN_BYTES
    }
    if not substantial:
        print(f"No substantial workflow scripts under {state_dir}/workflows")
        return

    shape_counts: Counter[str] = Counter()
    primitive_counts: Counter[str] = Counter()
    for record in substantial.values():
        source = record["source"]
        for shape in classify(source):
            shape_counts[shape] += 1
        for name, needle in PRIMITIVE_MARKERS.items():
            if name == "while-loop":
                if re.search(r"\bwhile\b", source):
                    primitive_counts[name] += 1
            elif needle in source:
                primitive_counts[name] += 1

    total = len(substantial)
    print(f"unique scripts: {len(scripts)}; substantial (>= {MIN_CALLS} calls, > {MIN_BYTES}B): {total}")
    print("\nshapes (keyword-approximate, overlapping):")
    for shape, count in shape_counts.most_common():
        print(f"  {shape:<20} {count:>4}  ({100 * count // total}%)")
    print("\nprimitives (exact):")
    for name, count in primitive_counts.most_common():
        print(f"  {name:<20} {count:>4}  ({100 * count // total}%)")


if __name__ == "__main__":
    main()
