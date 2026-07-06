"""Format LSP diagnostics for agent consumption and UI presentation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .client import LspDiagnostic

# LSP severity constants
_SEVERITY_ERROR = 1
_SEVERITY_WARNING = 2
_SEVERITY_INFO = 3
_SEVERITY_HINT = 4

_SEVERITY_EMOJI = {
    _SEVERITY_ERROR: "\U0001f534",  # 🔴
    _SEVERITY_WARNING: "\U0001f7e1",  # 🟡
    _SEVERITY_INFO: "\U0001f535",  # 🔵
    _SEVERITY_HINT: "\U0001f535",  # 🔵
}

_SEVERITY_LABEL = {
    _SEVERITY_ERROR: "error",
    _SEVERITY_WARNING: "warning",
    _SEVERITY_INFO: "info",
    _SEVERITY_HINT: "hint",
}


@dataclass
class MissingServer:
    """Information about a language server that was detected as needed but not found."""

    language_id: str
    display_name: str
    detection_reason: str  # e.g. "pyproject.toml + 42 .py files"
    server_name: str
    server_bin: str
    install_commands: list[str]
    alternatives: list[str]  # other available server names for this language


def format_diagnostics(diagnostics: list[LspDiagnostic], path: str, source: str = "", max_diags: int = 20) -> str:
    """Format a list of LSP diagnostics as plain text.

    The output begins with a header line of the form
    ``LSP diagnostics (<summary label>):`` (e.g. ``LSP diagnostics (2 warnings):``)
    followed by one line per diagnostic. ``extract_lsp_label`` parses that header so
    the TUI can surface a glanceable badge without re-deriving the counts.

    Args:
        diagnostics: Raw diagnostics from the language server.
        path: The file path (reserved for future use; not currently rendered).
        source: Language server name shown as attribution, e.g. ``(pyright)``.
        max_diags: Maximum diagnostics to include; excess is truncated with a note.

    Returns:
        Plain text suitable for appending to a tool result. No leading newline —
        the caller adds any separator it wants between the result and this block.
    """
    if not diagnostics:
        return ""

    label = summary_label(severity_counts(diagnostics))
    lines: list[str] = [f"LSP diagnostics ({label}):"]

    shown = diagnostics[:max_diags]
    for diag in shown:
        line_no = _line_number(diag)
        emoji = _SEVERITY_EMOJI.get(diag.severity or 0, "\u26a0\ufe0f")  # ⚠️ fallback
        msg = diag.message.strip()
        code_str = f" [{diag.code}]" if diag.code else ""
        source_label = diag.source or source
        source_str = f" ({source_label})" if source_label else ""

        if line_no is not None:
            lines.append(f"{emoji} Line {line_no}: {msg}{code_str}{source_str}")
        else:
            lines.append(f"{emoji} {msg}{code_str}{source_str}")

    if len(diagnostics) > max_diags:
        lines.append(f"... and {len(diagnostics) - max_diags} more (capped at {max_diags})")

    return "\n".join(lines)


def severity_counts(diagnostics: list[LspDiagnostic]) -> dict:
    """Tally diagnostics by LSP severity.

    Returns a dict with keys ``total``, ``errors``, ``warnings``, ``infos``,
    ``hints``. Unknown severities count toward ``total`` only.
    """
    counts = {"total": 0, "errors": 0, "warnings": 0, "infos": 0, "hints": 0}
    for diag in diagnostics:
        counts["total"] += 1
        severity = diag.severity
        if severity == _SEVERITY_ERROR:
            counts["errors"] += 1
        elif severity == _SEVERITY_WARNING:
            counts["warnings"] += 1
        elif severity == _SEVERITY_INFO:
            counts["infos"] += 1
        elif severity == _SEVERITY_HINT:
            counts["hints"] += 1
    return counts


def summary_label(counts: dict) -> str:
    """Compact, pluralization-correct severity label for a ``severity_counts`` dict.

    Examples: ``"2 warnings"``, ``"1 error, 2 warnings"``, ``"3 diagnostics"``.
    """
    parts: list[str] = []
    errors = int(counts.get("errors", 0))
    warnings = int(counts.get("warnings", 0))
    notes = int(counts.get("infos", 0)) + int(counts.get("hints", 0))
    if errors:
        parts.append(f"{errors} error{'s' if errors != 1 else ''}")
    if warnings:
        parts.append(f"{warnings} warning{'s' if warnings != 1 else ''}")
    if notes:
        parts.append(f"{notes} note{'s' if notes != 1 else ''}")
    if not parts:
        total = int(counts.get("total", 0))
        return f"{total} diagnostic{'s' if total != 1 else ''}"
    return ", ".join(parts)


# Matches the ``LSP diagnostics (<label>):`` header emitted by format_diagnostics.
_LSP_HEADER_RE = re.compile(r"LSP diagnostics \(([^)]+)\)")


def extract_lsp_label(text: str) -> Optional[str]:
    """Recover the severity summary label from a tool-result text, or ``None``.

    Scans for the ``LSP diagnostics (<label>):`` header produced by
    :func:`format_diagnostics` and returns the captured label (e.g. ``"2 warnings"``).
    Returns ``None`` when the text carries no diagnostics block, so callers can
    treat the absence as "no badge".
    """
    if not text:
        return None
    match = _LSP_HEADER_RE.search(text)
    return match.group(1) if match else None


def format_no_diagnostics() -> str:
    """Return a positive confirmation when no diagnostics were found."""
    return "\n✅ No LSP diagnostics."


def format_missing_prompt(missing: list[MissingServer]) -> str:
    """Format a user-facing prompt listing missing language servers with install commands.

    Intended for stderr / status display, not for agent tool results.
    """
    if not missing:
        return ""

    lines: list[str] = [
        "\n⚠  Language servers missing:",
        "",
    ]

    for ms in missing:
        install = ms.install_commands[0] if ms.install_commands else "See docs for install instructions"
        lines.append(f"  {ms.display_name:12s} →  {ms.server_name:20s} install: {install}")

    # Alternatives
    all_alternatives: list[str] = []
    for ms in missing:
        for alt in ms.alternatives:
            all_alternatives.append(f"{alt} ({ms.display_name})")

    if all_alternatives:
        lines.append(f"\n  Alternatives available: {', '.join(all_alternatives)}")

    lines.append("\n  → Configure in .kolega/lsp.json or set KOLEGA_LSP_DISABLED_LANGUAGES to suppress.")
    lines.append("  → Run 'kolega install-lsp' to attempt installing all missing servers.")

    return "\n".join(lines)


def format_detected_summary(detected: list[tuple[str, str, str]]) -> str:
    """Format the detected-languages summary shown at startup.

    Args:
        detected: List of ``(language_id, display_name, detection_reason)`` tuples.
    """
    if not detected:
        return ""

    lines: list[str] = [
        f"\n🔍 Detected {len(detected)} language(s) from your project:",
    ]
    for lang_id, display_name, reason in detected:
        lines.append(f"  • {display_name:12s} ({reason})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _line_number(diag: LspDiagnostic) -> Optional[int]:
    """Extract the 1-based line number from a diagnostic range."""
    start = diag.range.get("start", {})
    if isinstance(start, dict):
        raw = start.get("line")
        if isinstance(raw, int):
            return raw + 1  # LSP uses 0-based lines; convert to 1-based
    return None


def _diag_sort_key(diag: LspDiagnostic) -> tuple:
    """Sort key: severity (errors first), then line, then character."""
    start = diag.range.get("start", {})
    line = start.get("line", 0) if isinstance(start, dict) else 0
    char = start.get("character", 0) if isinstance(start, dict) else 0
    severity = diag.severity if diag.severity is not None else 99
    return (severity, line, char)


def _diag_dedup_key(diag: LspDiagnostic) -> tuple:
    """Dedup key: (start_line, start_char, message, source, code)."""
    start = diag.range.get("start", {})
    line = start.get("line", 0) if isinstance(start, dict) else 0
    char = start.get("character", 0) if isinstance(start, dict) else 0
    return (line, char, diag.message, diag.source, diag.code)


def dedupe_and_sort(diagnostics: list[LspDiagnostic], max_count: int = 20) -> list[LspDiagnostic]:
    """Deduplicate, sort by severity/location, and cap diagnostics.

    - **Dedup** by ``(start_line, start_character, message, source, code)``.
    - **Sort** by ``(severity, start_line, start_character)`` — errors first.
    - **Cap** at *max_count*.
    """
    seen: set[tuple] = set()
    deduped: list[LspDiagnostic] = []
    for d in diagnostics:
        key = _diag_dedup_key(d)
        if key not in seen:
            seen.add(key)
            deduped.append(d)
    deduped.sort(key=_diag_sort_key)
    return deduped[:max_count]
