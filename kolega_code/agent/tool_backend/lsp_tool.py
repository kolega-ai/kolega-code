"""Read-only LSP tool for agents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, cast
from urllib.parse import unquote, urlparse

from .base_tool import BaseTool
from .edit_preview import build_diff_preview
from kolega_code.services.lsp import LspManager, format_diagnostics, format_no_diagnostics
from kolega_code.services.lsp.edits import WorkspaceEditApplier, WorkspaceEditError, WorkspaceEditResult


class LspTool(BaseTool):
    """Exposes read-only LSP operations as an agent-callable tool."""

    def __init__(self, *args, lsp_manager: Optional[LspManager] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lsp_manager = lsp_manager

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
            parts = [f"  {index}. `{title}`", f"action_id={LspTool._action_id(action, index - 1)}"]
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
    def _action_id(action: dict[str, Any], index: int) -> str:
        payload = {
            "index": index,
            "title": action.get("title"),
            "kind": action.get("kind"),
            "command": action.get("command"),
            "edit": action.get("edit"),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]

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


class LspEditTool(BaseTool):
    """Trusted mutating LSP operations."""

    _ALL_OPS = {"rename", "rename_file", "format_document", "format_range", "apply_code_action"}

    def __init__(self, *args, lsp_manager: Optional[LspManager] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lsp_manager = lsp_manager

    async def lsp_edit(
        self,
        operation: str,
        path: Optional[str] = None,
        line: Optional[int] = None,
        symbol: Optional[str] = None,
        new_name: Optional[str] = None,
        new_path: Optional[str] = None,
        query: Optional[str] = None,
        action_id: Optional[str] = None,
        end_line: Optional[int] = None,
        kind: Optional[str] = None,
        apply: bool = True,
        timeout: Optional[float] = None,
    ) -> str:
        """Apply trusted LSP edits such as rename, formatting, and code actions."""
        if self._lsp_manager is None or not self._lsp_manager.enabled:
            return "LSP edits are not available (LSP is disabled or not configured)."

        if operation not in self._ALL_OPS:
            return f"Unknown operation '{operation}'. Valid operations: {', '.join(sorted(self._ALL_OPS))}."

        if not self._lsp_manager._initialized:
            await self._lsp_manager.initialize()

        kw_timeout: dict[str, Any] = {"timeout": timeout} if timeout is not None else {}

        try:
            if operation == "rename":
                return await self._op_rename(path, line, symbol, new_name, apply, kw_timeout)
            if operation == "rename_file":
                return await self._op_rename_file(path, new_path, apply, kw_timeout)
            if operation == "format_document":
                return await self._op_format_document(path, apply, kw_timeout)
            if operation == "format_range":
                return await self._op_format_range(path, line, end_line, apply, kw_timeout)
            if operation == "apply_code_action":
                return await self._op_apply_code_action(
                    path, line, symbol, query, action_id, end_line, kind, apply, kw_timeout
                )
            return f"Operation '{operation}' is not yet implemented."
        except WorkspaceEditError as exc:
            return f"LSP edit was rejected: {exc}"
        except ValueError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            await self.log_warning(f"LSP edit operation '{operation}' failed: {exc}", sender=self.caller.agent_name)
            return f"LSP edit operation '{operation}' failed: {exc}"

    async def _op_rename(
        self,
        path: Optional[str],
        line: Optional[int],
        symbol: Optional[str],
        new_name: Optional[str],
        should_apply: bool,
        kw_timeout: dict[str, Any],
    ) -> str:
        assert self._lsp_manager is not None
        if not path:
            return "Error: 'path' is required for the 'rename' operation."
        if line is None:
            return "Error: 'line' (1-based) is required for the 'rename' operation."
        if not symbol:
            return "Error: 'symbol' is required for the 'rename' operation."
        if not new_name:
            return "Error: 'new_name' is required for the 'rename' operation."

        server_name = self._lsp_manager.server_for_path(path)
        if server_name is None:
            return f"No language server configured for {path}."

        lsp_line, character = self._lsp_manager._resolve_position(path, line, symbol)
        edit = await self._lsp_manager.get_rename(path, lsp_line, character, new_name, **kw_timeout)
        if edit is None:
            return f"Rename is not supported by {server_name}, or no rename edits were returned."
        return await self._apply_or_preview_workspace_edit("rename", edit, should_apply)

    async def _op_rename_file(
        self,
        path: Optional[str],
        new_path: Optional[str],
        should_apply: bool,
        kw_timeout: dict[str, Any],
    ) -> str:
        assert self._lsp_manager is not None
        if not path:
            return "Error: 'path' is required for the 'rename_file' operation."
        if not new_path:
            return "Error: 'new_path' is required for the 'rename_file' operation."

        will_rename_edits = await self._lsp_manager.will_rename_files(path, new_path, **kw_timeout)
        result_lines: list[str] = []
        touched_paths: list[str] = []

        for edit in will_rename_edits:
            preview_result = self._preview_workspace_edit(edit)
            result = await self._apply_or_preview_workspace_edit(
                "rename_file:willRenameFiles",
                edit,
                should_apply,
                include_diagnostics=False,
            )
            result_lines.append(result)
            touched_paths.extend(preview_result.touched_paths)

        rename_edit = {
            "documentChanges": [
                {
                    "kind": "rename",
                    "oldUri": self._file_uri(path),
                    "newUri": self._file_uri(new_path),
                    "options": {"overwrite": False, "ignoreIfExists": False},
                }
            ]
        }
        result = await self._apply_or_preview_workspace_edit(
            "rename_file",
            rename_edit,
            should_apply,
            include_diagnostics=False,
        )
        result_lines.append(result)
        touched_paths.extend([path, new_path])

        if should_apply:
            await self._lsp_manager.did_rename_files(path, new_path)
            diagnostics = await self._diagnostics_for_paths([new_path, *touched_paths])
            if diagnostics:
                result_lines.append(diagnostics)

        return "\n\n".join(line for line in result_lines if line)

    async def _op_format_document(
        self,
        path: Optional[str],
        should_apply: bool,
        kw_timeout: dict[str, Any],
    ) -> str:
        assert self._lsp_manager is not None
        if not path:
            return "Error: 'path' is required for the 'format_document' operation."

        server_name = self._lsp_manager.server_for_path(path)
        if server_name is None:
            return f"No language server configured for {path}."

        edits = await self._lsp_manager.get_document_formatting(path, **kw_timeout)
        if edits is None:
            return f"Document formatting is not supported by {server_name}."
        if not edits:
            return "Document formatting returned no edits."
        return await self._apply_or_preview_workspace_edit(
            "format_document",
            self._text_edits_to_workspace_edit(path, edits),
            should_apply,
        )

    async def _op_format_range(
        self,
        path: Optional[str],
        line: Optional[int],
        end_line: Optional[int],
        should_apply: bool,
        kw_timeout: dict[str, Any],
    ) -> str:
        assert self._lsp_manager is not None
        if not path:
            return "Error: 'path' is required for the 'format_range' operation."
        if line is None:
            return "Error: 'line' (1-based) is required for the 'format_range' operation."

        server_name = self._lsp_manager.server_for_path(path)
        if server_name is None:
            return f"No language server configured for {path}."

        end_line_1based = end_line if end_line is not None else line
        end_character = self._line_end_character(path, end_line_1based)
        edits = await self._lsp_manager.get_range_formatting(
            path,
            line - 1,
            0,
            end_line_1based - 1,
            end_character,
            **kw_timeout,
        )
        if edits is None:
            return f"Range formatting is not supported by {server_name}."
        if not edits:
            return "Range formatting returned no edits."
        return await self._apply_or_preview_workspace_edit(
            "format_range",
            self._text_edits_to_workspace_edit(path, edits),
            should_apply,
        )

    async def _op_apply_code_action(
        self,
        path: Optional[str],
        line: Optional[int],
        symbol: Optional[str],
        query: Optional[str],
        action_id: Optional[str],
        end_line: Optional[int],
        kind: Optional[str],
        should_apply: bool,
        kw_timeout: dict[str, Any],
    ) -> str:
        assert self._lsp_manager is not None
        if not path:
            return "Error: 'path' is required for the 'apply_code_action' operation."
        if line is None:
            return "Error: 'line' (1-based) is required for the 'apply_code_action' operation."
        if not symbol:
            return "Error: 'symbol' is required for the 'apply_code_action' operation."

        server_name = self._lsp_manager.server_for_path(path)
        if server_name is None:
            return f"No language server configured for {path}."

        lsp_line, character = self._lsp_manager._resolve_position(path, line, symbol)
        lsp_end_line = end_line - 1 if end_line is not None else None
        actions = await self._lsp_manager.get_code_actions(
            path,
            lsp_line,
            character,
            end_line=lsp_end_line,
            kind=kind,
            **kw_timeout,
        )
        if actions is None:
            return f"Code actions are not supported by {server_name}, or no actions were found."
        if not isinstance(actions, list) or not actions:
            return "No code actions found."

        selected = self._select_code_action(actions, action_id=action_id, query=query)
        if selected is None:
            return "No matching code action found. Use lsp code_actions to list action_id values."

        disabled = selected.get("disabled")
        if isinstance(disabled, dict):
            reason = disabled.get("reason")
            return f"Code action is disabled: {reason or 'no reason provided'}."

        selected = await self._lsp_manager.resolve_code_action(path, selected, **kw_timeout)
        edit = selected.get("edit")
        command = self._command_payload(selected)
        result_lines: list[str] = []
        touched_paths: list[str] = []

        if isinstance(edit, dict) and edit:
            preview_result = self._preview_workspace_edit(edit)
            result = await self._apply_or_preview_workspace_edit(
                "apply_code_action",
                edit,
                should_apply,
                include_diagnostics=False,
            )
            result_lines.append(result)
            touched_paths.extend(preview_result.touched_paths)

        if command:
            if not should_apply:
                result_lines.append(f"Preview: command `{command['command']}` would be executed on apply.")
            else:
                applier = WorkspaceEditApplier(self.project_path, self.filesystem)
                server_results: list[WorkspaceEditResult] = []

                def apply_edit_handler(params: dict[str, Any]) -> dict[str, Any]:
                    workspace_edit = params.get("edit") if isinstance(params, dict) else None
                    try:
                        preview = applier.preview(workspace_edit)
                        blocked = self._first_blocked_path(preview.touched_paths)
                        if blocked:
                            return {"applied": False, "failureReason": blocked}
                        applied = applier.apply(workspace_edit)
                        server_results.append(applied)
                        return {"applied": True}
                    except WorkspaceEditError as exc:
                        return {"applied": False, "failureReason": str(exc)}

                self._lsp_manager.set_workspace_apply_edit_handler(apply_edit_handler)
                try:
                    command_result = await self._lsp_manager.execute_command(path, command, **kw_timeout)
                finally:
                    self._lsp_manager.set_workspace_apply_edit_handler(None)

                result_lines.append(f"Executed LSP command `{command['command']}`.")
                for server_result in server_results:
                    await self._send_previews(server_result, "apply_code_action")
                    touched_paths.extend(server_result.touched_paths)
                    result_lines.append(self._format_workspace_edit_result("apply_code_action", server_result))
                if isinstance(command_result, dict) and (
                    "changes" in command_result or "documentChanges" in command_result
                ):
                    preview_result = self._preview_workspace_edit(command_result)
                    result = await self._apply_or_preview_workspace_edit(
                        "apply_code_action:executeCommand",
                        command_result,
                        True,
                        include_diagnostics=False,
                    )
                    result_lines.append(result)
                    touched_paths.extend(preview_result.touched_paths)

        if not result_lines:
            return "Selected code action returned no edit or executable command."

        if should_apply:
            diagnostics = await self._diagnostics_for_paths(touched_paths)
            if diagnostics:
                result_lines.append(diagnostics)

        return "\n\n".join(line for line in result_lines if line)

    async def _apply_or_preview_workspace_edit(
        self,
        operation: str,
        edit: dict[str, Any],
        should_apply: bool,
        *,
        include_diagnostics: bool = True,
    ) -> str:
        applier = WorkspaceEditApplier(self.project_path, self.filesystem)
        preview = applier.preview(edit)
        blocked = self._first_blocked_path(preview.touched_paths)
        if should_apply and blocked:
            return blocked

        result = applier.apply(edit) if should_apply else preview
        await self._send_previews(result, operation)
        lines = [self._format_workspace_edit_result(operation, result)]
        if should_apply and include_diagnostics:
            diagnostics = await self._diagnostics_for_paths(result.touched_paths)
            if diagnostics:
                lines.append(diagnostics)
        return "\n\n".join(line for line in lines if line)

    def _preview_workspace_edit(self, edit: dict[str, Any]) -> WorkspaceEditResult:
        return WorkspaceEditApplier(self.project_path, self.filesystem).preview(edit)

    async def _send_previews(self, result: WorkspaceEditResult, operation: str) -> None:
        for change in result.text_changes:
            if change.old_text == change.new_text:
                continue
            await self.send_edit_preview(
                build_diff_preview(change.old_text, change.new_text, change.path),
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                tool_name="lsp_edit",
            )

    def _format_workspace_edit_result(self, operation: str, result: WorkspaceEditResult) -> str:
        label = "Applied" if result.applied else "Preview"
        lines = [f"{label} LSP edit `{operation}`."]
        for summary in result.summaries:
            lines.append(f"- {summary}")
        return "\n".join(lines)

    async def _diagnostics_for_paths(self, paths: list[str] | tuple[str, ...]) -> str:
        assert self._lsp_manager is not None
        if not self._lsp_manager._config.auto_diagnostics_on_edit:
            return ""
        lines: list[str] = []
        for path in dict.fromkeys(paths):
            if not path or not self.filesystem.exists(path) or self.filesystem.is_dir(path):
                continue
            server_name = self._lsp_manager.server_for_path(path)
            if server_name is None:
                continue
            diagnostics = await self._lsp_manager.get_fresh_diagnostics(path)
            if diagnostics:
                lines.append(format_diagnostics(diagnostics, path, source=server_name))
        return "\n\n".join(lines)

    def _first_blocked_path(self, paths: list[str] | tuple[str, ...]) -> Optional[str]:
        for path in paths:
            blocked = self._enforce_vibe_edit_policy(path)
            if blocked:
                return blocked
        return None

    def _text_edits_to_workspace_edit(self, path: str, edits: Any) -> dict[str, Any]:
        if not isinstance(edits, list):
            raise WorkspaceEditError("Server returned invalid text edits.")
        return {"changes": {self._file_uri(path): edits}}

    def _file_uri(self, path: str) -> str:
        path_obj = Path(path)
        absolute = path_obj.resolve() if path_obj.is_absolute() else (self.project_path / path_obj).resolve()
        return absolute.as_uri()

    def _line_end_character(self, path: str, line_1based: int) -> int:
        text = self.filesystem.read_text(path)
        lines = text.split("\n")
        line_index = line_1based - 1
        if line_index < 0 or line_index >= len(lines):
            raise ValueError(f"Line {line_1based} is out of range (file has {len(lines)} lines).")
        return sum(2 if ord(char) > 0xFFFF else 1 for char in lines[line_index])

    def _select_code_action(
        self,
        actions: list[Any],
        *,
        action_id: Optional[str],
        query: Optional[str],
    ) -> Optional[dict[str, Any]]:
        dict_actions = [action for action in actions if isinstance(action, dict)]
        if action_id:
            for index, action in enumerate(dict_actions):
                if LspTool._action_id(action, index) == action_id:
                    return action
            return None

        if query:
            query_text = query.strip()
            if query_text.isdigit():
                selected_index = int(query_text)
                if 0 <= selected_index < len(dict_actions):
                    return dict_actions[selected_index]
                if 1 <= selected_index <= len(dict_actions):
                    return dict_actions[selected_index - 1]
            folded = query_text.casefold()
            for action in dict_actions:
                command_payload = self._command_payload(action)
                haystack = " ".join(
                    str(value)
                    for value in (
                        action.get("title"),
                        action.get("kind"),
                        command_payload.get("command") if command_payload else None,
                    )
                    if value
                ).casefold()
                if folded in haystack:
                    return action
            return None

        if len(dict_actions) == 1:
            return dict_actions[0]
        return None

    def _command_payload(self, action: dict[str, Any]) -> Optional[dict[str, Any]]:
        command = action.get("command")
        if isinstance(command, dict):
            command_name = command.get("command")
            if isinstance(command_name, str) and command_name:
                return {
                    "title": command.get("title") or action.get("title") or command_name,
                    "command": command_name,
                    "arguments": command.get("arguments", []),
                }
            return None
        if isinstance(command, str) and command:
            return {
                "title": action.get("title") or command,
                "command": command,
                "arguments": action.get("arguments", []),
            }
        return None


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
