"""Trace record for native Tinker sampling.

A deliberately dependency-free leaf module: the record is part of the package's
public export surface (extension authors type their trace sinks against it), so
it must be importable without pulling in the optional native Tinker stack —
``providers/tinker.py`` imports tinker-cookbook (and therefore torch) at module
load whenever the ``[tinker]`` extra is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TinkerTraceRecord:
    """Structured record of one native Tinker sample, for on-policy RL.

    Emitted only when a trace sink is attached; the record carries everything
    the Anthropic-compatible endpoint cannot provide: the exact rendered
    prompt tokens, the sampled token ids, per-token behavior logprobs, the stop
    reason, prefix-cache usage, and the model/checkpoint identity.
    """

    model: str
    base_model: str
    request_role: Optional[Dict[str, Any]]
    prompt_tokens: List[int]
    sampled_tokens: List[int]
    sampled_text: str
    logprobs: List[Optional[float]]
    stop_reason: str
    termination: str
    cache_hit_tokens: int
    temperature: Optional[float]
    max_tokens: Optional[int]
