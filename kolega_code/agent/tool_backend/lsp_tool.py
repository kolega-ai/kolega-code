"""``lsp_diagnostics`` tool — explicit LSP diagnostic queries for agents.

Provides a read-only tool that agents call to check for errors, warnings, and
hints from language servers after editing files.
"""

from __future__ import annotations

from typing import Optional

from .base_tool import BaseTool
from kolega_code.services.lsp import LspManager, format_diagnostics, format_no_diagnostics


class LspTool(BaseTool):
    """Exposes LSP diagnostics as an agent-callable tool.

    Requires an ``LspManager`` instance; diagnostics are delegated to the
    appropriate language server for the file's detected language.
    """

    def __init__(self, *args, lsp_manager: Optional[LspManager] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lsp_manager = lsp_manager

    async def lsp_diagnostics(self, path: str) -> str:
        """Get language server diagnostics (errors, warnings, hints) for a file.

        Use this when you want to verify that a file you just edited or created is
        free of syntax errors, type errors, or other code quality issues. The
        diagnostics come from the project's language servers (e.g. pyright for
        Python, typescript-language-server for TypeScript).

        When to use this tool:
        - After editing or creating a file to verify correctness
        - When you suspect a file may have issues but aren't sure
        - Before proposing changes to verify the baseline
        - When a previous edit produced unexpected behavior

        Usage notes:
        1. The path should be relative to the project root (or absolute).
        2. Diagnostics are returned as markdown with severity indicators
           (🔴 error, 🟡 warning, 🔵 info/hint).
        3. If no language server is available for the file's language, a message
           is returned noting that.
        4. Results are capped (default: 20 diagnostics per file).

        Args:
            path: Path to the file. Relative to the project root is preferred;
                  an absolute path is also accepted.

        Returns:
            A markdown-formatted list of diagnostics, or a confirmation message
            if no issues were found.
        """
        if self._lsp_manager is None or not self._lsp_manager.enabled:
            return "LSP diagnostics are not available (LSP is disabled or not configured)."

        if not self._lsp_manager._initialized:
            await self._lsp_manager.initialize()

        server_name = self._lsp_manager.server_for_path(path)
        if server_name is None:
            # LSP may not have a server for this file type
            return f"No language server configured for {path}."

        try:
            diagnostics = await self._lsp_manager.get_diagnostics(path)
        except Exception as exc:
            await self.log_warning(f"LSP diagnostic query failed for {path}: {exc}", sender=self.caller.agent_name)
            return f"LSP diagnostic query failed for {path}: {exc}"

        if not diagnostics:
            return format_no_diagnostics()

        return format_diagnostics(
            diagnostics,
            path,
            source=server_name,
        )
