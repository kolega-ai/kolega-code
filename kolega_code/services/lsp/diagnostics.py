"""Format LSP diagnostics as markdown for agent consumption."""

from __future__ import annotations

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
    """Format a list of LSP diagnostics as markdown.

    Args:
        diagnostics: Raw diagnostics from the language server.
        path: The file path (for the header).
        source: Language server name shown as attribution, e.g. ``"(pyright)"``.
        max_diags: Maximum diagnostics to include; excess is truncated with a note.

    Returns:
        A markdown string suitable for appending to a tool result.
    """
    if not diagnostics:
        return ""

    lines: list[str] = []
    lines.append("\n### LSP Diagnostics")

    shown = diagnostics[:max_diags]
    for diag in shown:
        line_no = _line_number(diag)
        emoji = _SEVERITY_EMOJI.get(diag.severity or 0, "\u26a0\ufe0f")  # ⚠️ fallback
        msg = diag.message.strip()
        code_str = f" [{diag.code}]" if diag.code else ""
        source_str = f" ({source})" if source else ""

        if line_no is not None:
            lines.append(f"{emoji} Line {line_no}: {msg}{code_str}{source_str}")
        else:
            lines.append(f"{emoji} {msg}{code_str}{source_str}")

    if len(diagnostics) > max_diags:
        lines.append(f"\n... and {len(diagnostics) - max_diags} more diagnostics (capped at {max_diags})")

    return "\n".join(lines)


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
