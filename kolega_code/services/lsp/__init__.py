"""Language Server Protocol integration for Kolega Code.

Provides auto-detection of project languages, management of language server
subprocesses, and diagnostic queries wired into the agent's edit/write tools.

Public API
----------

- ``LspManager``: lifecycle manager for language server processes.
- ``LspConfig``: user-facing configuration (wired into ``AgentConfig``).
- ``LspRegistry``: loads and merges preset + user language definitions.
- ``LanguageSpec``, ``LanguageServerSpec``: dataclasses describing a language
  and its available servers.
- ``LspDiagnostic``: mirrors the LSP ``Diagnostic`` struct.
- ``format_diagnostics``: format a list of ``LspDiagnostic`` as markdown
  (appended to edit/write tool results).
- ``format_no_diagnostics``: "no diagnostics" confirmation message.
- ``format_missing_prompt``: user-facing prompt about missing language servers.
- ``format_detected_summary``: summary of detected languages shown at startup.
"""

from .client import LspClientError, LspDiagnostic, PublishDiagnosticsParams, parse_publish_diagnostics
from .config import LanguageServerSpec, LanguageSpec, LspConfig, platform_key
from .diagnostics import (
    MissingServer,
    dedupe_and_sort,
    extract_lsp_label,
    format_detected_summary,
    format_diagnostics,
    format_missing_prompt,
    format_no_diagnostics,
)
from .manager import LspManager
from .registry import LspRegistry, load_project_lsp_config

__all__ = [
    "LanguageServerSpec",
    "LanguageSpec",
    "LspClientError",
    "LspConfig",
    "LspDiagnostic",
    "LspManager",
    "LspRegistry",
    "MissingServer",
    "PublishDiagnosticsParams",
    "dedupe_and_sort",
    "extract_lsp_label",
    "format_detected_summary",
    "format_diagnostics",
    "format_missing_prompt",
    "format_no_diagnostics",
    "load_project_lsp_config",
    "parse_publish_diagnostics",
    "platform_key",
]
