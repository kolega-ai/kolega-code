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
   caller names explicitly via ``extra_secrets``.

   This is pattern matching, not proof. It recognises the common credential
   shapes and any value the caller hands it; a secret in a shape nobody
   anticipated can still get through. Callers that know their own secrets — the
   CLI knows every configured API key — must pass them rather than trusting the
   patterns.

3. **Host paths.** The home directory is rewritten to ``~`` in every exported
   string, so a bundle does not disclose the directory layout, and local paths
   are dropped from artifact references entirely. Usernames appearing outside a
   home path are deliberately left alone: a short or common username would
   corrupt unrelated prose if replaced globally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
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
    #: Local paths removed from artifact *references*.
    paths_stripped: int = 0
    #: Strings in which a home directory prefix was rewritten to ``~``.
    home_paths_rewritten: int = 0

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
        if self.home_paths_rewritten:
            lines.append(f"Rewrote your home directory to ~ in {self.home_paths_rewritten} string(s).")
        # Said every time, not only when something matched. "Redacted 3 strings"
        # reads as "this bundle is clean", and it is not that.
        lines.append("Secret detection is best-effort pattern matching — check the output before sharing it.")
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
    # sub_agent_info is free text the model wrote — a dispatch task, quoting file
    # paths and whatever else the caller pasted in — and it rides *every* event
    # of that dispatch. Scrubbing only content left one task string repeated
    # across hundreds of events as the largest single source of leaked paths.
    if clone.sub_agent_info is not None:
        clone.sub_agent_info = _scrub(clone.sub_agent_info, report=report, extra=secrets)

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
        rehomed = scrub_home_paths(cleaned)
        if rehomed != cleaned:
            report.home_paths_rewritten += 1
        return rehomed
    if isinstance(value, dict):
        return {key: _scrub(item, report=report, extra=extra) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, report=report, extra=extra) for item in value]
    return value


@lru_cache(maxsize=8)
def _prefixes_for_home(home: str) -> tuple[str, ...]:
    """Spellings of ``home`` to rewrite, longest first.

    Both the raw and the resolved form are needed: on macOS a home under the
    temporary directory resolves to a ``/private``-prefixed twin, and command
    output routinely contains whichever one the tool happened to print.

    Keyed on ``home`` rather than cached outright so that changing ``HOME`` —
    which every isolated test does — recomputes instead of reusing a previous
    machine's answer. ``resolve()`` touches the filesystem, which is why this is
    not simply recomputed for each of the thousands of strings in an export.
    """
    candidates: set[str] = set()
    for form in (Path(home), Path(home).resolve()):
        text = str(form).rstrip("/")
        # A home of "/" would rewrite every absolute path in the session.
        if len(text) > 1:
            candidates.add(text)
    return tuple(sorted(candidates, key=len, reverse=True))


def scrub_home_paths(text: str) -> str:
    """Rewrite this machine's home directory to ``~`` inside ``text``."""
    for prefix in _prefixes_for_home(os.path.expanduser("~")):
        if prefix in text:
            text = text.replace(prefix, "~")
    return text


def redact_artifact_bytes(
    ref: ArtifactRef,
    data: bytes,
    *,
    report: RedactionReport,
    extra_secrets: Optional[Iterable[str]] = None,
) -> bytes:
    """Scrub a text artifact's payload the same way its event's strings are.

    Only the *references* used to be redacted. The payload behind them was
    written to the bundle verbatim, and those payloads are the long tool outputs
    that got offloaded precisely because they were big — a file dump, a command
    that printed a config. That is the likeliest place for a secret in the whole
    export, and it was the one place nothing looked.

    Images are returned untouched: they are binary, and scrubbing bytes that are
    not text would corrupt them for no benefit.

    The reference keeps its original digest, which is the bundle's lookup key. It
    identifies the payload this artifact came from, not the redacted bytes.
    """
    if not str(ref.media_type or "").startswith("text/"):
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    cleaned = redact_secrets(text, list(extra_secrets or ()))
    if cleaned != text:
        report.strings_redacted += 1
    rehomed = scrub_home_paths(cleaned)
    if rehomed != cleaned:
        report.home_paths_rewritten += 1
    return rehomed.encode("utf-8")


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
