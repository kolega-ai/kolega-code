"""LSP configuration dataclasses: language specs, server specs, and user-facing config."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class LanguageServerSpec:
    """Describes a single language server binary and how to install it."""

    name: str
    """Human-readable server name, e.g. ``"pyright"``."""

    bin: str
    """Executable name (resolved via ``shutil.which``)."""

    args: list[str] = field(default_factory=list)
    """Default arguments passed when spawning the server (e.g. ``["--stdio"]``)."""

    install_commands: dict[str, list[str]] = field(default_factory=dict)
    """Platform-keyed install commands, e.g. ``{"darwin": [...], "linux": [...]}``."""

    env: dict[str, str] = field(default_factory=dict)
    """Optional environment variables to set for the server process.

    These are merged onto a minimal allowlisted base environment (PATH, HOME, …);
    provider API keys and other inherited variables are withheld. Loader-injection
    variables (``LD_PRELOAD``, ``PYTHONPATH``, ``NODE_OPTIONS``, …) are always
    stripped, even when declared here.
    """


@dataclass(frozen=True)
class LanguageSpec:
    """Describes one programming / markup language for auto-detection and LSP routing."""

    id: str
    """Stable identifier, e.g. ``"python"``."""

    display_name: str
    """Human-readable name, e.g. ``"Python"``."""

    config_files: list[str] = field(default_factory=list)
    """Filenames at the project root that strongly indicate this language.

    May include globs like ``"*.csproj"``. Checked non-recursively.
    """

    extensions: list[str] = field(default_factory=list)
    """File extensions (dot-prefixed), e.g. ``[".py", ".pyi"]``."""

    filename_map: dict[str, str] = field(default_factory=dict)
    """Exact filenames that map to this language (e.g. ``"Dockerfile"``)."""

    language_servers: list[LanguageServerSpec] = field(default_factory=list)
    """Ordered list of available servers (first match on PATH wins)."""

    family: Optional[str] = None
    """When set, diagnostics for this language reuse the server of ``family``.

    Example: ``"typescript"`` sets ``family = "javascript"`` so both share
    ``typescript-language-server``.
    """


@dataclass
class LspConfig:
    """User-facing LSP configuration (wired into ``AgentConfig``)."""

    enabled: bool = True
    """Master kill-switch. Set ``False`` to disable all LSP activity."""

    auto_diagnostics_on_edit: bool = True
    """Append LSP diagnostics to supported post-edit tool results."""

    max_diagnostics: int = 20
    """Maximum diagnostics returned per file per query."""

    auto_fallback: bool = True
    """If the preferred server is missing, silently use an available alternative."""

    prompt_on_missing: bool = True
    """Show install prompts when a detected language has no server binary on PATH."""

    disabled_languages: list[str] = field(default_factory=list)
    """Language IDs to skip entirely (no detection, no prompting)."""

    preferences: dict[str, str] = field(default_factory=dict)
    """Per-language preferred server name, e.g. ``{"python": "basedpyright"}``."""

    custom_servers: dict[str, dict] = field(default_factory=dict)
    """User-defined server definitions keyed by server name."""

    initialization_options: dict[str, dict] = field(default_factory=dict)
    """Per-server initialization options keyed by server name.

    Example: ``{"pyright": {"python": {"analysis": {"typeCheckingMode": "basic"}}}}``.
    Passed as ``initializationOptions`` in the LSP ``initialize`` request.
    """

    diagnostic_servers: list[str] = field(default_factory=list)
    """Names of *additional* servers to start alongside the primary for extra
    linting coverage.  Default: empty (one server per language).
    """

    workspace_configuration: dict[str, dict] = field(default_factory=dict)
    """Per-server responses for ``workspace/configuration`` requests.

    Example: ``{"pyright": {"python": {"analysis": {"typeCheckingMode": "basic"}}}}``.
    """


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _platform() -> str:
    p = sys.platform
    if p == "darwin":
        return "darwin"
    if p.startswith("linux"):
        return "linux"
    if p in ("win32", "cygwin"):
        return "windows"
    return p  # unknown — callers handle gracefully


def platform_key() -> str:
    """Return ``"darwin"``, ``"linux"``, or ``"windows"`` for the current host."""
    return _platform()
