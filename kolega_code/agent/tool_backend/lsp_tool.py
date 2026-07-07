"""Read-only LSP tools for agents.

Provides ``lsp_diagnostics`` (compatibility wrapper) and the generic ``lsp``
tool for diagnostics, go-to-definition, references, hover, symbols, and status.
"""

from __future__ import annotations

import json
from typing import Any, Optional, cast
from urllib.parse import unquote, urlparse

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

    # -- generic lsp tool ----------------------------------------

    _POSITION_OPS = {"definition", "type_definition", "implementation", "references", "hover"}
    _PATH_OPS = {
        "diagnostics",
        "document_symbols",
        "definition",
        "type_definition",
        "implementation",
        "references",
        "hover",
        "call_hierarchy",
        "code_actions",
        "capabilities",
    }
    _ALL_OPS = _POSITION_OPS | {
        "call_hierarchy",
        "code_actions",
        "diagnostics",
        "document_symbols",
        "workspace_symbols",
        "status",
        "capabilities",
        "reload",
    }

    async def lsp(
        self,
        operation: str,
        path: Optional[str] = None,
        line: Optional[int] = None,
        symbol: Optional[str] = None,
        query: Optional[str] = None,
        end_line: Optional[int] = None,
        kind: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Query language server intelligence (diagnostics, definition, references, hover, symbols, status).

        This is a versatile read-only tool for interacting with the project's language
        servers. Different operations require different arguments.

        Operations:
        - **diagnostics**: Get errors/warnings/hints for a file. Requires ``path``.
        - **definition**: Go to definition. Requires ``path``, ``line`` (1-based), ``symbol``.
        - **type_definition**: Go to type definition. Requires ``path``, ``line``, ``symbol``.
        - **implementation**: Find implementations. Requires ``path``, ``line``, ``symbol``.
        - **references**: Find all references. Requires ``path``, ``line``, ``symbol``.
        - **hover**: Get hover/type info. Requires ``path``, ``line``, ``symbol``.
        - **call_hierarchy**: Show incoming/outgoing calls. Requires ``path``, ``line``, ``symbol``.
        - **code_actions**: List available fixes/refactors. Requires ``path``, ``line``, ``symbol``.
        - **document_symbols**: List symbols in a file. Requires ``path``.
        - **workspace_symbols**: Search project-wide symbols. Requires ``query``.
        - **status**: Show LSP server status (no args required).
        - **capabilities**: Show server capabilities. Optional ``path``.
        - **reload**: Restart language servers and re-detect languages (no args required).

        Position resolution: For position operations, provide a 1-based ``line`` number
        and a ``symbol`` name. The tool finds the symbol on that line and computes the
        exact character position. Use ``name#N`` to target the Nth occurrence on the line.

        Args:
            operation: One of the operations listed above.
            path: File path (relative to project root preferred).
            line: 1-based line number for position operations.
            symbol: Symbol name to resolve on the line (supports ``name#N``).
            query: Search query for workspace_symbols.
            end_line: Optional 1-based end line for code_actions.
            kind: Optional code action kind filter, e.g. ``quickfix`` or ``refactor``.
            timeout: Per-call timeout in seconds (default: 30).

        Returns:
            Markdown-formatted results for the requested operation.
        """
        if self._lsp_manager is None or not self._lsp_manager.enabled:
            return "LSP is not available (disabled or not configured)."

        if operation not in self._ALL_OPS:
            return f"Unknown operation '{operation}'. Valid operations: {', '.join(sorted(self._ALL_OPS))}."

        if not self._lsp_manager._initialized:
            await self._lsp_manager.initialize()

        kw_timeout: dict[str, Any] = {"timeout": timeout} if timeout is not None else {}

        try:
            if operation == "diagnostics":
                return await self._op_diagnostics(path)
            elif operation in self._POSITION_OPS:
                return await self._op_position(operation, path, line, symbol, kw_timeout)
            elif operation == "call_hierarchy":
                return await self._op_call_hierarchy(path, line, symbol, kw_timeout)
            elif operation == "code_actions":
                return await self._op_code_actions(path, line, symbol, end_line, kind, kw_timeout)
            elif operation == "document_symbols":
                return await self._op_document_symbols(path, kw_timeout)
            elif operation == "workspace_symbols":
                return await self._op_workspace_symbols(query, kw_timeout)
            elif operation == "status":
                return self._op_status()
            elif operation == "capabilities":
                return self._op_capabilities(path)
            elif operation == "reload":
                return await self._op_reload()
            else:
                return f"Operation '{operation}' is not yet implemented."
        except ValueError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            await self.log_warning(f"LSP operation '{operation}' failed: {exc}", sender=self.caller.agent_name)
            return f"LSP operation '{operation}' failed: {exc}"

    async def _op_diagnostics(self, path: Optional[str]) -> str:
        assert self._lsp_manager is not None
        if not path:
            return "Error: 'path' is required for the 'diagnostics' operation."
        server_name = self._lsp_manager.server_for_path(path)
        if server_name is None:
            return f"No language server configured for {path}."
        diagnostics = await self._lsp_manager.get_diagnostics(path)
        if not diagnostics:
            return format_no_diagnostics()
        return format_diagnostics(diagnostics, path, source=server_name)

    async def _op_position(
        self,
        operation: str,
        path: Optional[str],
        line: Optional[int],
        symbol: Optional[str],
        kw_timeout: dict[str, Any],
    ) -> str:
        assert self._lsp_manager is not None
        if not path:
            return f"Error: 'path' is required for the '{operation}' operation."
        if line is None:
            return f"Error: 'line' (1-based) is required for the '{operation}' operation."
        if not symbol:
            return f"Error: 'symbol' is required for the '{operation}' operation."

        server_name = self._lsp_manager.server_for_path(path)
        if server_name is None:
            return f"No language server configured for {path}."

        lsp_line, character = self._lsp_manager._resolve_position(path, line, symbol)

        method_map = {
            "definition": self._lsp_manager.get_definition,
            "type_definition": self._lsp_manager.get_type_definition,
            "implementation": self._lsp_manager.get_implementation,
            "references": self._lsp_manager.get_references,
            "hover": self._lsp_manager.get_hover,
        }
        handler = method_map[operation]
        result = await handler(path, lsp_line, character, **kw_timeout)

        if result is None:
            return f"The '{operation}' operation is not supported by {server_name}, or no results were found."

        return self._format_position_result(operation, result, server_name)

    async def _op_call_hierarchy(
        self,
        path: Optional[str],
        line: Optional[int],
        symbol: Optional[str],
        kw_timeout: dict[str, Any],
    ) -> str:
        assert self._lsp_manager is not None
        if not path:
            return "Error: 'path' is required for the 'call_hierarchy' operation."
        if line is None:
            return "Error: 'line' (1-based) is required for the 'call_hierarchy' operation."
        if not symbol:
            return "Error: 'symbol' is required for the 'call_hierarchy' operation."

        server_name = self._lsp_manager.server_for_path(path)
        if server_name is None:
            return f"No language server configured for {path}."

        lsp_line, character = self._lsp_manager._resolve_position(path, line, symbol)
        result = await self._lsp_manager.get_call_hierarchy(path, lsp_line, character, **kw_timeout)
        if result is None:
            return f"Call hierarchy is not supported by {server_name}, or no results were found."
        return self._format_call_hierarchy(result)

    async def _op_code_actions(
        self,
        path: Optional[str],
        line: Optional[int],
        symbol: Optional[str],
        end_line: Optional[int],
        kind: Optional[str],
        kw_timeout: dict[str, Any],
    ) -> str:
        assert self._lsp_manager is not None
        if not path:
            return "Error: 'path' is required for the 'code_actions' operation."
        if line is None:
            return "Error: 'line' (1-based) is required for the 'code_actions' operation."
        if not symbol:
            return "Error: 'symbol' is required for the 'code_actions' operation."

        server_name = self._lsp_manager.server_for_path(path)
        if server_name is None:
            return f"No language server configured for {path}."

        lsp_line, character = self._lsp_manager._resolve_position(path, line, symbol)
        lsp_end_line = end_line - 1 if end_line is not None else None
        result = await self._lsp_manager.get_code_actions(
            path,
            lsp_line,
            character,
            end_line=lsp_end_line,
            kind=kind,
            **kw_timeout,
        )
        if result is None:
            return f"Code actions are not supported by {server_name}, or no actions were found."
        return self._format_code_actions(result)

    async def _op_document_symbols(self, path: Optional[str], kw_timeout: dict[str, Any]) -> str:
        assert self._lsp_manager is not None
        if not path:
            return "Error: 'path' is required for the 'document_symbols' operation."
        server_name = self._lsp_manager.server_for_path(path)
        if server_name is None:
            return f"No language server configured for {path}."
        result = await self._lsp_manager.get_document_symbols(path, **kw_timeout)
        if result is None:
            return f"Document symbols are not supported by {server_name}, or the file has no symbols."
        return self._format_symbols(result)

    async def _op_workspace_symbols(self, query: Optional[str], kw_timeout: dict[str, Any]) -> str:
        assert self._lsp_manager is not None
        if not query:
            return "Error: 'query' is required for the 'workspace_symbols' operation."
        result = await self._lsp_manager.get_workspace_symbols(query, **kw_timeout)
        if result is None:
            return "Workspace symbol search is not supported, or no symbols were found."
        if not result:
            return f"No symbols found matching '{query}'."
        return self._format_symbol_info_list(result)

    def _op_status(self) -> str:
        assert self._lsp_manager is not None
        status = self._lsp_manager.status()
        return self._format_status(status)

    def _op_capabilities(self, path: Optional[str]) -> str:
        assert self._lsp_manager is not None
        if path:
            caps = self._lsp_manager.get_capabilities(path)
            if not caps:
                return f"No active language server for {path}, or capabilities not yet available."
            return f"Server capabilities for {path}:\n\n```json\n{json.dumps(caps, indent=2)}\n```"
        # No path — show all sessions' capabilities summary
        lines = ["## LSP Capabilities"]
        for lang_id, client in self._lsp_manager._sessions.items():
            caps = client.server_capabilities or {}
            providers = sorted(k for k in caps if k.endswith("Provider"))
            lines.append(f"\n**{lang_id}** ({client.status}): {', '.join(providers) if providers else 'none'}")
        return "\n".join(lines)

    async def _op_reload(self) -> str:
        assert self._lsp_manager is not None
        messages = await self._lsp_manager.reload()
        if messages:
            return "LSP reloaded.\n" + "\n".join(messages)
        return "LSP reloaded."

    # -- formatting helpers -------------------------------------------------

    @staticmethod
    def _format_position_result(operation: str, result: Any, server_name: str) -> str:
        """Format definition/references/hover results as readable text."""
        op_label = operation.replace("_", " ").title()

        if operation == "hover":
            return LspTool._format_hover(result, server_name)

        # Definition, typeDefinition, implementation, references → Location | Location[] | LocationLink[] | null
        locations = result
        if isinstance(result, dict) and ("uri" in result or "targetUri" in result):
            locations = [result]
        elif isinstance(result, list):
            locations = result
        else:
            locations = []

        if not locations:
            return f"No {op_label.lower()} results."

        lines = [f"## {op_label} ({len(locations)} result{'s' if len(locations) != 1 else ''})"]
        for loc in locations[:50]:
            lines.append(f"  - `{LspTool._format_location(loc)}`")

        if len(locations) > 50:
            lines.append(f"  ... and {len(locations) - 50} more")
        return "\n".join(lines)

    @staticmethod
    def _format_hover(result: Any, server_name: str) -> str:
        if not result:
            return "No hover information available."
        contents = result.get("contents") if isinstance(result, dict) else None
        if contents is None:
            return "No hover information available."

        if isinstance(contents, dict):
            value = contents.get("value", "")
            language = contents.get("language", "")
            if language:
                return f"```{language}\n{value}\n```"
            return value
        elif isinstance(contents, list):
            parts = []
            for item in contents:
                if isinstance(item, dict):
                    value = item.get("value", "")
                    language = item.get("language", "")
                    if language:
                        parts.append(f"```{language}\n{value}\n```")
                    else:
                        parts.append(value)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n\n".join(parts) if parts else "No hover information available."
        elif isinstance(contents, str):
            return contents
        return "No hover information available."

    @staticmethod
    def _format_symbols(symbols: Any) -> str:
        """Format documentSymbol results."""
        if not symbols:
            return "No symbols found in this file."

        lines = ["## Document Symbols"]
        count = 0

        def append_symbol(sym: dict[str, Any], depth: int = 0) -> None:
            nonlocal count
            if count >= 100:
                return
            name = sym.get("name", "?")
            kind = sym.get("kind", 0)
            detail = sym.get("detail", "")
            kind_str = _SYMBOL_KINDS.get(kind, "Symbol")
            detail_str = f" — {detail}" if detail else ""
            rng = sym.get("range") or sym.get("selectionRange") or {}
            start = rng.get("start", {}) if isinstance(rng, dict) else {}
            line_no = start.get("line", 0) + 1 if isinstance(start, dict) else 0
            indent = "  " + ("  " * depth)
            lines.append(f"{indent}- `{name}` ({kind_str}){detail_str} — line {line_no}")
            count += 1
            for child in sym.get("children", []) or []:
                if isinstance(child, dict):
                    append_symbol(child, depth + 1)

        for sym in symbols[:100]:
            if isinstance(sym, dict):
                append_symbol(sym)

        if len(symbols) > 100 or count >= 100:
            lines.append("  ... additional nested symbols omitted")
        return "\n".join(lines)

    @staticmethod
    def _format_symbol_info_list(symbols: Any) -> str:
        """Format workspace/symbol (SymbolInformation[]) results."""
        if not symbols:
            return "No symbols found."

        lines = [f"## Workspace Symbols ({len(symbols)} found)"]
        for sym in symbols[:50]:
            if isinstance(sym, dict):
                name = sym.get("name", "?")
                kind = sym.get("kind", 0)
                container = sym.get("containerName", "")
                loc = sym.get("location", {})
                kind_str = _SYMBOL_KINDS.get(kind, "Symbol")
                container_str = f" in {container}" if container else ""
                location = LspTool._format_location(loc) if isinstance(loc, dict) else "?"
                lines.append(f"  - `{name}` ({kind_str}){container_str} — `{location}`")

        if len(symbols) > 50:
            lines.append(f"  ... and {len(symbols) - 50} more")
        return "\n".join(lines)

    @staticmethod
    def _format_code_actions(actions: Any) -> str:
        if not actions:
            return "No code actions found."
        if not isinstance(actions, list):
            return "No code actions found."

        lines = [f"## Code Actions ({len(actions)} found)"]
        for index, action in enumerate(actions[:50], 1):
            if not isinstance(action, dict):
                continue
            title = action.get("title") or action.get("command") or f"Action {index}"
            kind = action.get("kind")
            command = action.get("command")
            diagnostics = action.get("diagnostics") or []
            raw_edit = action.get("edit")
            edit = cast(dict[str, Any], raw_edit) if isinstance(raw_edit, dict) else {}
            edit_summary = LspTool._summarize_workspace_edit(edit)
            parts = [f"  {index}. `{title}`"]
            if kind:
                parts.append(f"kind={kind}")
            if command:
                parts.append(f"command={command}")
            if diagnostics:
                parts.append(f"diagnostics={len(diagnostics)}")
            if edit_summary:
                parts.append(edit_summary)
            lines.append(" — ".join(parts))
        if len(actions) > 50:
            lines.append(f"  ... and {len(actions) - 50} more")
        return "\n".join(lines)

    @staticmethod
    def _summarize_workspace_edit(edit: dict[str, Any]) -> str:
        if not edit:
            return ""
        changes = edit.get("changes")
        doc_changes = edit.get("documentChanges")
        parts = []
        if isinstance(changes, dict):
            parts.append(f"{sum(len(v) for v in changes.values() if isinstance(v, list))} text edits")
        if isinstance(doc_changes, list):
            parts.append(f"{len(doc_changes)} document changes")
        return ", ".join(parts)

    @staticmethod
    def _format_call_hierarchy(result: Any) -> str:
        if not isinstance(result, dict):
            return "No call hierarchy found."
        items = result.get("items") or []
        incoming = result.get("incoming") or []
        outgoing = result.get("outgoing") or []
        if not items and not incoming and not outgoing:
            return "No call hierarchy found."

        lines = ["## Call Hierarchy"]
        if items:
            lines.append("\n**Prepared items:**")
            for item in items[:10]:
                if isinstance(item, dict):
                    lines.append(f"  - {LspTool._format_call_item(item)}")
        if incoming:
            lines.append(f"\n**Incoming calls ({len(incoming)}):**")
            for call in incoming[:50]:
                if isinstance(call, dict):
                    caller = call.get("from", {})
                    lines.append(f"  - {LspTool._format_call_item(caller)}")
        if outgoing:
            lines.append(f"\n**Outgoing calls ({len(outgoing)}):**")
            for call in outgoing[:50]:
                if isinstance(call, dict):
                    callee = call.get("to", {})
                    lines.append(f"  - {LspTool._format_call_item(callee)}")
        return "\n".join(lines)

    @staticmethod
    def _format_call_item(item: Any) -> str:
        if not isinstance(item, dict):
            return "`?`"
        name = item.get("name", "?")
        kind = _SYMBOL_KINDS.get(item.get("kind", 0), "Symbol")
        uri = item.get("uri", "?")
        rng = item.get("selectionRange") or item.get("range") or {}
        loc = LspTool._format_location({"uri": uri, "range": rng})
        return f"`{name}` ({kind}) — `{loc}`"

    @staticmethod
    def _format_location(loc: dict[str, Any]) -> str:
        uri = loc.get("targetUri") or loc.get("uri") or "?"
        rng = loc.get("targetSelectionRange") or loc.get("targetRange") or loc.get("range") or {}
        start = rng.get("start", {}) if isinstance(rng, dict) else {}
        line_no = start.get("line", 0) + 1 if isinstance(start, dict) else 1
        char_no = start.get("character", 0) if isinstance(start, dict) else 0
        return f"{_uri_to_display_path(uri)}:{line_no}:{char_no}"

    @staticmethod
    def _format_status(status: dict) -> str:
        """Format the manager status dict as readable text."""
        lines = ["## 🔍 LSP Status"]

        if not status.get("enabled"):
            lines.append("\nLSP is disabled.")
            return "\n".join(lines)

        detected = status.get("detected", [])
        if detected:
            lines.append(f"\n**Detected {len(detected)} language(s):**")
            for d in detected:
                lines.append(f"  - {d['display_name']} ({d['detection_reason']})")
        else:
            lines.append("\nNo languages detected.")

        sessions = status.get("sessions", [])
        if sessions:
            lines.append(f"\n**Active sessions ({len(sessions)}):**")
            for s in sessions:
                status_icon = "✅" if s.get("connected") else "❌"
                pid_str = f" pid={s['pid']}" if s.get("pid") else ""
                error_str = f" — {s['last_error']}" if s.get("last_error") else ""
                lines.append(f"  {status_icon} {s['server_name']} ({s['language_id']}){pid_str}{error_str}")

        missing = status.get("missing", [])
        if missing:
            lines.append(f"\n**⚠️ Missing servers ({len(missing)}):**")
            for m in missing:
                lines.append(f"  - {m['display_name']} → {m['server_name']}")

        diag_counts = status.get("diagnostic_counts", {})
        if diag_counts:
            lines.append("\n**Last diagnostic counts:**")
            for uri, count in list(diag_counts.items())[:10]:
                path_str = uri.replace("file://", "") if uri.startswith("file://") else uri
                lines.append(f"  - {path_str}: {count}")

        return "\n".join(lines)


# LSP SymbolKind constants (subset)
_SYMBOL_KINDS: dict[int, str] = {
    1: "File",
    2: "Module",
    3: "Namespace",
    4: "Package",
    5: "Class",
    6: "Method",
    7: "Property",
    8: "Field",
    9: "Constructor",
    10: "Enum",
    11: "Interface",
    12: "Function",
    13: "Variable",
    14: "Constant",
    15: "String",
    16: "Number",
    17: "Boolean",
    18: "Array",
    19: "Object",
    20: "Key",
    21: "Null",
    22: "EnumMember",
    23: "Struct",
    24: "Event",
    25: "Operator",
    26: "TypeParameter",
}


def _uri_to_display_path(uri: str) -> str:
    if not uri.startswith("file://"):
        return uri
    parsed = urlparse(uri)
    path = unquote(parsed.path)
    if parsed.netloc:
        return f"//{parsed.netloc}{path}"
    return path
