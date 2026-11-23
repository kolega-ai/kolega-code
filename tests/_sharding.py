"""Deterministic test shard selection for the ``--shard`` pytest option."""

from __future__ import annotations

import zlib

import pytest


def parse_shard(spec: str) -> tuple[int, int]:
    """Turn an ``INDEX/TOTAL`` spec into a 1-based index and the shard count."""
    try:
        index_text, total_text = spec.split("/", 1)
        index, total = int(index_text), int(total_text)
    except ValueError:
        raise pytest.UsageError(f"--shard must look like INDEX/TOTAL, got {spec!r}") from None
    if total < 1 or not 1 <= index <= total:
        raise pytest.UsageError(f"--shard={spec} has an index outside 1..{total}")
    return index, total


def owns_nodeid(nodeid: str, index: int, total: int) -> bool:
    """True when the shard at 1-based ``index`` is responsible for ``nodeid``."""
    return zlib.crc32(nodeid.encode()) % total == index - 1
