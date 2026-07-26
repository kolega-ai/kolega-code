"""Export redaction: what may leave this machine, and what may not.

A shared session is the only place agent output crosses a trust boundary, so the
policy here is allowlist-based rather than denylist-based. Two independent gates:

1. **Artifact purpose.** Only ``tool_result`` and ``image`` payloads are content a
   viewer needs. The other four purposes are opaque provider state — reasoning
   signatures and encrypted reasoning — with no display value whatsoever. They
   exist so a conversation can be replayed to the model that produced it, and
   sharing them would leak provider-internal material for no benefit.
2. **Secret scrubbing.** Every string in every exported payload passes through
   :func:`kolega_code.security.redact_secrets`, which also catches values the
   agent happened to print.

Local file paths are stripped from artifact references as well: a bundle handed
to someone else should not disclose the directory layout of the machine that
produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from kolega_code.events import AgentEvent, ArtifactPurpose, ArtifactRef
from kolega_code.security.secrets import redact_secrets


@dataclass
class RedactionReport:
    """What the redaction pass changed, so an operator can see it before sharing."""

    events_scanned: int = 0
    strings_redacted: int = 0
    artifacts_kept: set[str] = field(default_factory=set)
    artifacts_dropped: dict[str, int] = field(default_factory=dict)
    paths_stripped: int = 0

    def summary_lines(self) -> list[str]:
        lines = [
            f"Scanned {self.events_scanned} events.",
            f"Redacted {self.strings_redacted} string(s) containing possible secrets.",
            f"Kept {len(self.artifacts_kept)} shareable artifact(s).",
        ]
        if self.artifacts_dropped:
            detail = ", ".join(f"{purpose} x{count}" for purpose, count in sorted(self.artifacts_dropped.items()))
            lines.append(f"Dropped non-shareable artifacts: {detail}.")
        if self.paths_stripped:
            lines.append(f"Stripped {self.paths_stripped} local filesystem path(s).")
        return lines


def redact_event(
    event: AgentEvent,
    *,
    report: RedactionReport,
    extra_secrets: Optional[Iterable[str]] = None,
) -> AgentEvent:
    """Return an export-safe copy of ``event``.

    The original is never mutated: the live session keeps its full fidelity and
    only the exported copy is reduced.
    """
    clone = event.model_copy(deep=True)
    report.events_scanned += 1

    secrets = list(extra_secrets or ())
    clone.content = _scrub(clone.content, report=report, extra=secrets)

    kept: list[ArtifactRef] = []
    for ref in clone.artifacts:
        if ref.purpose not in ArtifactPurpose.SHAREABLE:
            report.artifacts_dropped[ref.purpose] = report.artifacts_dropped.get(ref.purpose, 0) + 1
            continue
        if ref.path is not None:
            ref.path = None
            report.paths_stripped += 1
        report.artifacts_kept.add(ref.sha256)
        kept.append(ref)
    clone.artifacts = kept
    return clone


def _scrub(value: Any, *, report: RedactionReport, extra: list[str]) -> Any:
    """Recursively redact strings inside an arbitrary JSON-shaped payload."""
    if isinstance(value, str):
        cleaned = redact_secrets(value, extra)
        if cleaned != value:
            report.strings_redacted += 1
        return cleaned
    if isinstance(value, dict):
        return {key: _scrub(item, report=report, extra=extra) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, report=report, extra=extra) for item in value]
    return value


def shareable_artifacts(events: Iterable[AgentEvent]) -> dict[str, ArtifactRef]:
    """Collect the artifacts an export must include, keyed by digest.

    Only references that survived redaction appear here, so a bundle can never
    contain a blob that no exported event is allowed to point at.
    """
    refs: dict[str, ArtifactRef] = {}
    for event in events:
        for ref in event.artifacts:
            if ref.purpose in ArtifactPurpose.SHAREABLE:
                refs.setdefault(ref.sha256, ref)
    return refs
