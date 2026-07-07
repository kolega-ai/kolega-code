"""Language server lifecycle manager.

``LspManager`` holds a pool of active ``LspClient`` subprocesses, routes files
to the correct language server, and handles LSP initialization handshakes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from .client import LspClient, LspClientError, LspDiagnostic, parse_publish_diagnostics
from .config import LspConfig
from .detector import DetectionReport, ResolvedLanguage, detect_languages
from .diagnostics import (
    MissingServer,
    dedupe_and_sort,
    format_detected_summary,
    format_missing_prompt,
)
from .registry import LspRegistry, load_project_lsp_config

logger = logging.getLogger(__name__)

# Maximum concurrent server starts to avoid thundering herd
_MAX_CONCURRENT_STARTS = 4

# Client capabilities advertised in the initialize handshake.
_CLIENT_CAPABILITIES: dict = {
    "textDocument": {
        "synchronization": {"didSave": True, "willSave": False, "dynamicRegistration": False},
        "diagnostic": {"dynamicRegistration": True},
        "publishDiagnostics": {
            "relatedInformation": True,
            "tagSupport": {"valueSet": [1, 2]},
        },
        "hover": {"contentFormat": ["markdown", "plaintext"]},
        "definition": {"linkSupport": False},
        "typeDefinition": {"linkSupport": False},
        "implementation": {"linkSupport": False},
        "references": {},
        "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
        "codeAction": {"dynamicRegistration": True, "resolveSupport": {"properties": ["edit", "command"]}},
        "rename": {
            "dynamicRegistration": True,
            "prepareSupport": False,
            "honorsChangeAnnotations": False,
        },
        "formatting": {"dynamicRegistration": True},
        "rangeFormatting": {"dynamicRegistration": True},
        "callHierarchy": {"dynamicRegistration": True},
    },
    "workspace": {
        "symbol": {},
        "configuration": True,
        "workspaceFolders": True,
        "workspaceEdit": {
            "documentChanges": True,
            "resourceOperations": ["create", "rename", "delete"],
            "failureHandling": "textOnlyTransactional",
            "normalizesLineEndings": True,
            "changeAnnotationSupport": {"groupsOnLabel": False},
        },
        "executeCommand": {"dynamicRegistration": True},
        "fileOperations": {
            "dynamicRegistration": True,
            "willRename": True,
            "didRename": True,
        },
    },
}


@dataclass
class _LspSession:
    key: str
    language_id: str
    server_name: str
    client: LspClient
    root_uri: str
    opened_docs: dict[str, str] = field(default_factory=dict)
    doc_versions: dict[str, int] = field(default_factory=dict)
    diagnostics: dict[str, list[LspDiagnostic]] = field(default_factory=dict)
    diag_versions: dict[str, Optional[int]] = field(default_factory=dict)


class LspManager:
    """Manages language server subprocesses for a project.

    Created once per ``ToolCollection``; shared across ``EditTool`` and ``LspTool``.

    Lifecycle:
        - ``initialize()`` is called once (after construction) to auto-detect languages
          and optionally prompt the user about missing servers.
        - ``get_diagnostics(path)`` queries the appropriate language server for a file.
        - ``shutdown()`` stops all server processes.
    """

    def __init__(
        self,
        project_path: str | Path,
        *,
        config: Optional[LspConfig] = None,
        trusted: bool = False,
    ) -> None:
        self._project_path = Path(project_path).resolve()
        self._config = config or LspConfig()
        # Whether the project's .kolega/lsp.json is trusted to define custom
        # language servers. When False, project-level config is never loaded.
        self._trusted = trusted
        self._registry = LspRegistry(config=self._config)

        # Per-language/server LspClient sessions. Kept as a client map because
        # existing UI/tests inspect it directly.
        self._sessions: dict[str, LspClient] = {}
        self._session_records: dict[str, _LspSession] = {}
        # Per-key locks serializing session creation (prevents the race where two
        # concurrent callers for the same language both spawn a subprocess).
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Caps concurrent server starts to avoid a thundering herd of subprocesses.
        self._start_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_STARTS)
        # Maps file URI to language_id for tracking open documents
        self._open_files: dict[str, str] = {}
        # Latest diagnostics per URI (from publishDiagnostics notifications)
        self._diagnostics: dict[str, list[LspDiagnostic]] = {}
        # Diagnostics events for non-blocking await, keyed by (session_key, uri)
        self._diag_events: dict[tuple[str, str], asyncio.Event] = {}
        # Detection report (populated by initialize)
        self.report: Optional[DetectionReport] = None
        # Initialization lock
        self._init_lock = asyncio.Lock()
        self._initialized = False
        # Track resolved server info per language
        self._resolved: dict[str, ResolvedLanguage] = {}
        self._missing: dict[str, ResolvedLanguage] = {}

        # -- state tracking ---------------------------------------
        # Per-URI incrementing document version counter
        self._doc_versions: dict[str, int] = {}
        # Per-URI count of last received diagnostics (for status display)
        self.last_diagnostic_count: dict[str, int] = {}
        # Per-URI version tracked from publishDiagnostics (for freshness checks)
        self._diag_versions: dict[str, Optional[int]] = {}
        # Dynamically registered capabilities (from client/registerCapability)
        self._registered_capabilities: dict[str, dict] = {}
        # Additional diagnostic server sessions keyed by (lang_id, server_name)
        self._extra_sessions: dict[tuple[str, str], LspClient] = {}
        # Scoped handler used only by trusted mutating tools while a server
        # command is running. Default workspace/applyEdit remains read-only.
        self._workspace_apply_edit_handler: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None

    # -- public API --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    async def initialize(self) -> list[str]:
        """Auto-detect languages and resolve servers. Returns status messages for the UI.

        Called once per session (idempotent — subsequent calls are no-ops).
        """
        if self._initialized:
            return []
        async with self._init_lock:
            if self._initialized:
                return []

            if not self._config.enabled:
                self._initialized = True
                return []

            # Merge in project-level config (.kolega/lsp.json) only when the
            # project is trusted. Untrusted projects never load committed
            # custom_servers, preventing arbitrary-code-execution via a
            # committed config. The merge preserves the user's master
            # kill-switch (enabled) and any user-level settings the project
            # file omits.
            if self._trusted:
                project_overrides = load_project_lsp_config(self._project_path)
                if project_overrides is not None:
                    self._config = _merge_lsp_config(self._config, project_overrides)
                    self._registry = LspRegistry(config=self._config)

            # Auto-detect
            try:
                self.report = await detect_languages(self._project_path, self._registry)
            except Exception:
                logger.exception("LSP auto-detection failed")
                self._initialized = True
                return []

            self._resolved = {r.language_id: r for r in self.report.resolved}
            self._missing = {r.language_id: r for r in self.report.missing}

            messages: list[str] = []

            # Summary of detected languages
            detected_lines = []
            for dr in self.report.detected:
                detected_lines.append((dr.language_id, dr.display_name, dr.detection_reason))
            if detected_lines:
                messages.append(format_detected_summary(detected_lines))

            # Prompt about missing servers
            if self._config.prompt_on_missing and self._missing:
                missing_list = []
                for rl in self._missing.values():
                    missing_list.append(
                        MissingServer(
                            language_id=rl.language_id,
                            display_name=rl.display_name,
                            detection_reason=rl.detection_reason,
                            server_name=rl.server_name,
                            server_bin=rl.server_name,
                            install_commands=rl.install_commands,
                            alternatives=rl.alternatives,
                        )
                    )
                messages.append(format_missing_prompt(missing_list))

            self._initialized = True
            return messages

    async def get_diagnostics(self, path: str) -> list[LspDiagnostic]:
        """Get diagnostics for *path* from the appropriate language server.

        If no server is available for the file's language, returns an empty list.
        Servers are started lazily on first request.
        """
        if not self._config.enabled:
            return []

        # Determine language
        lang_id = self._language_for_path(path)
        if not lang_id:
            return []

        # Resolve server info
        rl = self._resolved.get(lang_id)
        if rl is None:
            rl = self._missing.get(lang_id)
        if rl is None:
            return []

        # Resolve family (e.g. typescript → javascript)
        effective_id = lang_id
        spec = self._registry.get(lang_id)
        if spec and spec.family:
            effective_id = spec.family

        # Get or start session
        try:
            client = await self._get_session(effective_id, rl)
            if client is None:
                return []
        except Exception:
            logger.exception("Failed to start LSP session for %s", effective_id)
            return []

        uri = _path_to_uri(self._project_path, path)

        # Open / change the document so the server knows about it
        try:
            await self._ensure_document_open(client, uri, path, lang_id, session_key=effective_id)
        except LspClientError:
            logger.warning("LSP document sync failed for %s", path)
            # Continue — some servers publish diagnostics without explicit didOpen

        # Request pull diagnostics (LSP 3.17)
        primary = await self._pull_diagnostics(client, uri, source=rl.server_name)
        extra = await self._get_extra_diagnostics(path, lang_id, uri)
        if primary is not None:
            return dedupe_and_sort(primary + extra, self._config.max_diagnostics)

        # Fallback: wait briefly for publishDiagnostics
        await self._wait_for_push_diagnostics(effective_id, uri, timeout=3.0)
        return dedupe_and_sort(self._diagnostics_for(effective_id, uri) + extra, self._config.max_diagnostics)

    async def get_fresh_diagnostics(self, path: str) -> list[LspDiagnostic]:
        """Get diagnostics for *path* after an edit, accepting only fresh results.

        Captures the pre-edit diagnostic snapshot, syncs new content, sends didSave,
        then accepts only diagnostics whose version matches the current document
        version (or falls back to push diagnostics if pull is unsupported).
        """
        if not self._config.enabled:
            return []

        effective_id, client = await self._get_or_start_session(path)
        if client is None or effective_id is None:
            return []

        lang_id = self._language_for_path(path)
        if not lang_id:
            return []

        uri = _path_to_uri(self._project_path, path)

        # Snapshot pre-edit version for freshness comparison
        record = self._session_records.get(effective_id)
        pre_edit_version = record.diag_versions.get(uri) if record else self._diag_versions.get(uri)

        # Sync new content (sends didChange with incremented version)
        try:
            await self._ensure_document_open(client, uri, path, lang_id, session_key=effective_id)
        except LspClientError:
            logger.warning("LSP document sync failed for %s", path)

        # Send didSave so servers that trigger on save re-analyze
        try:
            await client.notify("textDocument/didSave", {"textDocument": {"uri": uri}})
        except LspClientError:
            pass

        # Try pull diagnostics first
        primary = await self._pull_diagnostics(client, uri, source=self.server_for_path(path) or None)
        extra = await self._get_extra_diagnostics(path, lang_id, uri)
        if primary is not None:
            return dedupe_and_sort(primary + extra, self._config.max_diagnostics)

        # Fallback: wait for fresh push diagnostics
        await self._wait_for_push_diagnostics(effective_id, uri, timeout=3.0)

        # Accept push diagnostics only if the version is fresh (different from
        # pre-edit, or version tracking is not supported by the server)
        record = self._session_records.get(effective_id)
        push_version = record.diag_versions.get(uri) if record else self._diag_versions.get(uri)
        push_diags = self._diagnostics_for(effective_id, uri)

        if push_version is not None and push_version == pre_edit_version:
            logger.debug("LSP diagnostics for %s are stale at version %s", path, push_version)
            return []

        return dedupe_and_sort(push_diags + extra, self._config.max_diagnostics)

    # -- code intelligence ---------------------------------------

    async def _query_position(
        self,
        path: str,
        line: int,
        character: int,
        method: str,
        capability_path: tuple[str, ...],
        *,
        timeout: float = 30,
    ) -> Any:
        """Send an LSP position request, checking capabilities first."""
        if not self._config.enabled:
            return None

        effective_id, client = await self._get_or_start_session(path)
        if client is None or effective_id is None:
            return None

        lang_id = self._language_for_path(path)
        if not lang_id:
            return None

        uri = _path_to_uri(self._project_path, path)

        if not self._has_capability(client, *capability_path):
            return None

        try:
            await self._ensure_document_open(client, uri, path, lang_id, session_key=effective_id)
        except LspClientError:
            pass

        try:
            result = await client.request(
                method,
                {
                    "textDocument": {"uri": uri},
                    "position": {"line": line, "character": character},
                },
                timeout=timeout,
            )
            return result
        except LspClientError:
            return None

    async def get_definition(self, path: str, line: int, character: int, *, timeout: float = 30) -> Any:
        """Query ``textDocument/definition``."""
        return await self._query_position(
            path, line, character, "textDocument/definition", ("definitionProvider",), timeout=timeout
        )

    async def get_type_definition(self, path: str, line: int, character: int, *, timeout: float = 30) -> Any:
        """Query ``textDocument/typeDefinition``."""
        return await self._query_position(
            path, line, character, "textDocument/typeDefinition", ("typeDefinitionProvider",), timeout=timeout
        )

    async def get_implementation(self, path: str, line: int, character: int, *, timeout: float = 30) -> Any:
        """Query ``textDocument/implementation``."""
        return await self._query_position(
            path, line, character, "textDocument/implementation", ("implementationProvider",), timeout=timeout
        )

    async def get_references(self, path: str, line: int, character: int, *, timeout: float = 30) -> Any:
        """Query ``textDocument/references``."""
        return await self._query_position(
            path, line, character, "textDocument/references", ("referencesProvider",), timeout=timeout
        )

    async def get_hover(self, path: str, line: int, character: int, *, timeout: float = 30) -> Any:
        """Query ``textDocument/hover``."""
        return await self._query_position(
            path, line, character, "textDocument/hover", ("hoverProvider",), timeout=timeout
        )

    async def get_document_symbols(self, path: str, *, timeout: float = 30) -> Any:
        """Query ``textDocument/documentSymbol``."""
        if not self._config.enabled:
            return None

        effective_id, client = await self._get_or_start_session(path)
        if client is None or effective_id is None:
            return None

        lang_id = self._language_for_path(path)
        if not lang_id:
            return None

        uri = _path_to_uri(self._project_path, path)

        if not self._has_capability(client, "documentSymbolProvider"):
            return None

        try:
            await self._ensure_document_open(client, uri, path, lang_id, session_key=effective_id)
        except LspClientError:
            pass

        try:
            return await client.request(
                "textDocument/documentSymbol",
                {"textDocument": {"uri": uri}},
                timeout=timeout,
            )
        except LspClientError:
            return None

    async def get_workspace_symbols(self, query: str, *, timeout: float = 30) -> Any:
        """Query ``workspace/symbol``."""
        if not self._config.enabled:
            return None

        # Use any active session for workspace symbols
        client: Optional[LspClient] = None
        for c in self._sessions.values():
            if c.running:
                client = c
                break

        if client is None:
            # Try to start a session for any resolved language
            for rl in self._resolved.values():
                lang_id = rl.language_id
                effective_id = lang_id
                spec = self._registry.get(lang_id)
                if spec and spec.family:
                    effective_id = spec.family
                client = await self._get_session(effective_id, rl)
                if client:
                    break

        if client is None or not self._has_capability(client, "workspaceSymbolProvider"):
            return None

        try:
            return await client.request(
                "workspace/symbol",
                {"query": query},
                timeout=timeout,
            )
        except LspClientError:
            return None

    async def get_code_actions(
        self,
        path: str,
        line: int,
        character: int,
        *,
        end_line: Optional[int] = None,
        end_character: Optional[int] = None,
        kind: Optional[str] = None,
        timeout: float = 30,
    ) -> Any:
        """Query ``textDocument/codeAction`` without applying returned edits."""
        if not self._config.enabled:
            return None

        effective_id, client = await self._get_or_start_session(path)
        if client is None or effective_id is None:
            return None

        lang_id = self._language_for_path(path)
        if not lang_id:
            return None

        if not self._has_capability(client, "codeActionProvider"):
            return None

        uri = _path_to_uri(self._project_path, path)
        try:
            await self._ensure_document_open(client, uri, path, lang_id, session_key=effective_id)
        except LspClientError:
            pass

        range_end = {
            "line": end_line if end_line is not None else line,
            "character": end_character if end_character is not None else character + 1,
        }
        context: dict[str, Any] = {
            "diagnostics": [self._diagnostic_to_wire(diag) for diag in self._diagnostics_for(effective_id, uri)]
        }
        if kind:
            context["only"] = [kind]

        try:
            return await client.request(
                "textDocument/codeAction",
                {
                    "textDocument": {"uri": uri},
                    "range": {"start": {"line": line, "character": character}, "end": range_end},
                    "context": context,
                },
                timeout=timeout,
            )
        except LspClientError:
            return None

    async def get_rename(
        self,
        path: str,
        line: int,
        character: int,
        new_name: str,
        *,
        timeout: float = 30,
    ) -> Any:
        """Query ``textDocument/rename`` and return the server-provided WorkspaceEdit."""
        if not self._config.enabled:
            return None

        effective_id, client = await self._get_or_start_session(path)
        if client is None or effective_id is None:
            return None

        lang_id = self._language_for_path(path)
        if not lang_id:
            return None

        if not self._supports_method(client, ("renameProvider",), "textDocument/rename"):
            return None

        uri = _path_to_uri(self._project_path, path)
        try:
            await self._ensure_document_open(client, uri, path, lang_id, session_key=effective_id)
        except LspClientError:
            pass

        try:
            return await client.request(
                "textDocument/rename",
                {
                    "textDocument": {"uri": uri},
                    "position": {"line": line, "character": character},
                    "newName": new_name,
                },
                timeout=timeout,
            )
        except LspClientError:
            return None

    async def get_document_formatting(self, path: str, *, timeout: float = 30) -> Any:
        """Query ``textDocument/formatting`` and return text edits."""
        if not self._config.enabled:
            return None

        effective_id, client = await self._get_or_start_session(path)
        if client is None or effective_id is None:
            return None

        lang_id = self._language_for_path(path)
        if not lang_id:
            return None

        if not self._supports_method(client, ("documentFormattingProvider",), "textDocument/formatting"):
            return None

        uri = _path_to_uri(self._project_path, path)
        try:
            await self._ensure_document_open(client, uri, path, lang_id, session_key=effective_id)
        except LspClientError:
            pass

        try:
            return await client.request(
                "textDocument/formatting",
                {"textDocument": {"uri": uri}, "options": self._formatting_options(path)},
                timeout=timeout,
            )
        except LspClientError:
            return None

    async def get_range_formatting(
        self,
        path: str,
        start_line: int,
        start_character: int,
        end_line: int,
        end_character: int,
        *,
        timeout: float = 30,
    ) -> Any:
        """Query ``textDocument/rangeFormatting`` and return text edits."""
        if not self._config.enabled:
            return None

        effective_id, client = await self._get_or_start_session(path)
        if client is None or effective_id is None:
            return None

        lang_id = self._language_for_path(path)
        if not lang_id:
            return None

        if not self._supports_method(client, ("documentRangeFormattingProvider",), "textDocument/rangeFormatting"):
            return None

        uri = _path_to_uri(self._project_path, path)
        try:
            await self._ensure_document_open(client, uri, path, lang_id, session_key=effective_id)
        except LspClientError:
            pass

        try:
            return await client.request(
                "textDocument/rangeFormatting",
                {
                    "textDocument": {"uri": uri},
                    "range": {
                        "start": {"line": start_line, "character": start_character},
                        "end": {"line": end_line, "character": end_character},
                    },
                    "options": self._formatting_options(path),
                },
                timeout=timeout,
            )
        except LspClientError:
            return None

    async def resolve_code_action(self, path: str, action: dict[str, Any], *, timeout: float = 30) -> dict[str, Any]:
        """Resolve a code action if the server supports ``codeAction/resolve``."""
        effective_id, client = await self._get_or_start_session(path)
        if client is None or effective_id is None:
            return action

        provider = client.server_capabilities.get("codeActionProvider") if client.server_capabilities else None
        supports_resolve = isinstance(provider, dict) and bool(provider.get("resolveProvider"))
        if not supports_resolve and "codeAction/resolve" not in self._registered_capabilities:
            return action

        try:
            resolved = await client.request("codeAction/resolve", action, timeout=timeout)
        except LspClientError:
            return action
        return resolved if isinstance(resolved, dict) else action

    async def execute_command(self, path: str, command: dict[str, Any], *, timeout: float = 30) -> Any:
        """Execute a server command through ``workspace/executeCommand``."""
        effective_id, client = await self._get_or_start_session(path)
        if client is None or effective_id is None:
            return None

        command_name = command.get("command")
        if not isinstance(command_name, str) or not command_name:
            return None

        execute_provider = (
            client.server_capabilities.get("executeCommandProvider") if client.server_capabilities else None
        )
        supported_commands = execute_provider.get("commands") if isinstance(execute_provider, dict) else None
        if isinstance(supported_commands, list) and command_name not in supported_commands:
            return None

        try:
            return await client.request(
                "workspace/executeCommand",
                {
                    "command": command_name,
                    "arguments": command.get("arguments", []),
                },
                timeout=timeout,
            )
        except LspClientError:
            return None

    async def will_rename_files(
        self,
        old_path: str,
        new_path: str,
        *,
        timeout: float = 30,
    ) -> list[dict[str, Any]]:
        """Notify relevant servers before a file rename and collect WorkspaceEdits."""
        edits: list[dict[str, Any]] = []
        payload = {
            "files": [
                {
                    "oldUri": _path_to_uri(self._project_path, old_path),
                    "newUri": _path_to_uri(self._project_path, new_path),
                }
            ]
        }
        for client in await self._clients_for_file_operation(old_path, new_path):
            if not self._supports_method(
                client, ("workspace", "fileOperations", "willRename"), "workspace/willRenameFiles"
            ):
                continue
            try:
                result = await client.request("workspace/willRenameFiles", payload, timeout=timeout)
            except LspClientError:
                continue
            if isinstance(result, dict) and result:
                edits.append(result)
        return edits

    async def did_rename_files(self, old_path: str, new_path: str) -> None:
        """Notify relevant servers after a file rename."""
        payload = {
            "files": [
                {
                    "oldUri": _path_to_uri(self._project_path, old_path),
                    "newUri": _path_to_uri(self._project_path, new_path),
                }
            ]
        }
        for client in await self._clients_for_file_operation(old_path, new_path):
            if not self._supports_method(
                client, ("workspace", "fileOperations", "didRename"), "workspace/didRenameFiles"
            ):
                continue
            try:
                await client.notify("workspace/didRenameFiles", payload)
            except LspClientError:
                pass

        old_uri = _path_to_uri(self._project_path, old_path)
        for record in self._session_records.values():
            if old_uri in record.opened_docs:
                record.opened_docs.pop(old_uri, None)
        self._open_files.pop(old_uri, None)

    def set_workspace_apply_edit_handler(self, handler: Optional[Callable[[dict[str, Any]], dict[str, Any]]]) -> None:
        """Set the temporary handler for server-initiated ``workspace/applyEdit``."""
        self._workspace_apply_edit_handler = handler

    async def get_call_hierarchy(self, path: str, line: int, character: int, *, timeout: float = 30) -> Any:
        """Query call hierarchy prepare + incoming/outgoing calls."""
        if not self._config.enabled:
            return None

        effective_id, client = await self._get_or_start_session(path)
        if client is None or effective_id is None:
            return None

        lang_id = self._language_for_path(path)
        if not lang_id:
            return None

        if not self._has_capability(client, "callHierarchyProvider"):
            return None

        uri = _path_to_uri(self._project_path, path)
        try:
            await self._ensure_document_open(client, uri, path, lang_id, session_key=effective_id)
        except LspClientError:
            pass

        try:
            items = await client.request(
                "textDocument/prepareCallHierarchy",
                {
                    "textDocument": {"uri": uri},
                    "position": {"line": line, "character": character},
                },
                timeout=timeout,
            )
        except LspClientError:
            return None

        if items is None:
            return {"items": [], "incoming": [], "outgoing": []}
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return None

        incoming: list[Any] = []
        outgoing: list[Any] = []
        for item in items[:10]:
            if not isinstance(item, dict):
                continue
            try:
                result = await client.request("callHierarchy/incomingCalls", {"item": item}, timeout=timeout)
                if isinstance(result, list):
                    incoming.extend(result)
            except LspClientError:
                pass
            try:
                result = await client.request("callHierarchy/outgoingCalls", {"item": item}, timeout=timeout)
                if isinstance(result, list):
                    outgoing.extend(result)
            except LspClientError:
                pass

        return {"items": items, "incoming": incoming, "outgoing": outgoing}

    def get_capabilities(self, path: str) -> dict:
        """Return the stored server capabilities for *path*'s language server."""
        _, client, _ = self._resolve_session(path)
        if client is None or client.server_capabilities is None:
            return {}
        return dict(client.server_capabilities)

    def status(self) -> dict:
        """Return a structured status dict for observability."""
        detected = []
        if self.report:
            for d in self.report.detected:
                detected.append(
                    {
                        "language_id": d.language_id,
                        "display_name": d.display_name,
                        "detection_reason": d.detection_reason,
                    }
                )

        missing = []
        if self.report:
            for m in self.report.missing:
                missing.append(
                    {
                        "language_id": m.language_id,
                        "display_name": m.display_name,
                        "server_name": m.server_name,
                    }
                )

        resolved = []
        if self.report:
            for r in self.report.resolved:
                resolved.append(
                    {
                        "language_id": r.language_id,
                        "display_name": r.display_name,
                        "server_name": r.server_name,
                    }
                )

        sessions = []
        for lang_id, client in self._sessions.items():
            record = self._session_records.get(lang_id)
            rl = self._resolved.get(lang_id) or self._missing.get(lang_id)
            sessions.append(
                {
                    "key": lang_id,
                    "language_id": record.language_id if record else lang_id,
                    "server_name": record.server_name if record else (rl.server_name if rl else "unknown"),
                    "status": client.status,
                    "pid": client.server_pid,
                    "connected": client.running and client.status == "initialized",
                    "last_error": client.last_error,
                    "root": record.root_uri if record else client.active_root,
                }
            )

        return {
            "enabled": self._config.enabled,
            "initialized": self._initialized,
            "detected": detected,
            "resolved": resolved,
            "missing": missing,
            "sessions": sessions,
            "diagnostic_counts": dict(self.last_diagnostic_count),
        }

    async def reload(self) -> list[str]:
        """Re-run detection and restart all sessions."""
        await self.shutdown()
        self._initialized = False
        self.report = None
        return await self.initialize()

    async def shutdown(self) -> None:
        """Send ``shutdown`` + ``exit`` to all language servers and terminate them."""
        for lang_id, client in list(self._sessions.items()):
            try:
                if client.running:
                    await client.request("shutdown")
                    await client.notify("exit")
            except Exception:
                pass
            try:
                await client.stop()
            except Exception:
                logger.exception("Error stopping LSP session for %s", lang_id)
        self._sessions.clear()
        self._session_records.clear()
        self._open_files.clear()
        self._diagnostics.clear()
        self._diag_events.clear()
        self._doc_versions.clear()
        self._diag_versions.clear()
        self._registered_capabilities.clear()
        self._extra_sessions.clear()

    def server_for_path(self, path: str) -> Optional[str]:
        """Return the server name used for *path*, or ``None`` if none available."""
        lang_id = self._language_for_path(path)
        if not lang_id:
            return None
        rl = self._resolved.get(lang_id) or self._missing.get(lang_id)
        return rl.server_name if rl else None

    # -- internals ----------------------------------------------------------

    async def _get_session(
        self, lang_id: str, rl: ResolvedLanguage, *, session_key: Optional[str] = None
    ) -> Optional[LspClient]:
        """Return (or create and initialize) the LspClient for *lang_id*.

        Args:
            lang_id: Language identifier (used for logging).
            rl: Resolved language with server binary info.
            session_key: Override key for ``self._sessions``. Defaults to *lang_id*.
                Use a compound key like ``"python:ruff-lsp"`` for extra diagnostic servers.
        """
        key = session_key or lang_id
        # Serialize session creation per key so two concurrent callers for the
        # same language cannot both spawn + initialize a subprocess (the second
        # would overwrite the first client, orphaning its process).
        lock = self._session_locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Re-check after acquiring: another caller may have created it.
            if key in self._sessions:
                client = self._sessions[key]
                if client.running:
                    return client
                # Restart crashed session
                await client.stop()
                self._session_records.pop(key, None)

            if not rl.server_bin:
                return None

            cmd = [rl.server_bin] + list(rl.server_args)
            client = LspClient(cmd, env=rl.env or None)

            # Cap concurrent subprocess starts across all languages to avoid a
            # thundering herd when many files are queried at once.
            async with self._start_semaphore:
                try:
                    await client.start()
                except Exception as exc:
                    logger.warning("Failed to start LS %s for %s: %s", rl.server_name, lang_id, exc)
                    client.last_error = str(exc)
                    return None

                # Register push-diagnostics handler
                client.on_notification(
                    "textDocument/publishDiagnostics",
                    lambda params, session_key=key: self._on_publish_diagnostics(session_key, params),
                )

                # Register server→client request handlers
                self._register_server_request_handlers(client, server_name=rl.server_name)

                root_uri = _path_to_uri(self._project_path, str(self._project_path))
                client.active_root = root_uri
                record = _LspSession(
                    key=key,
                    language_id=lang_id,
                    server_name=rl.server_name,
                    client=client,
                    root_uri=root_uri,
                )

                # LSP initialize handshake
                try:
                    init_result = await client.request(
                        "initialize",
                        {
                            "processId": client.server_pid,
                            "rootUri": root_uri,
                            "initializationOptions": rl.initialization_options or {},
                            "capabilities": _CLIENT_CAPABILITIES,
                            "workspaceFolders": [{"uri": root_uri, "name": "workspace"}],
                        },
                    )
                    await client.notify("initialized", {})
                    client.server_capabilities = (
                        init_result.get("capabilities") if isinstance(init_result, dict) else None
                    )
                    client.status = "initialized"
                except LspClientError as exc:
                    logger.warning("LSP initialize failed for %s: %s", lang_id, exc)
                    client.status = "error"
                    client.last_error = str(exc)
                    await client.stop()
                    return None

            self._sessions[key] = client
            self._session_records[key] = record
            if session_key is not None:
                self._extra_sessions[(lang_id, rl.server_name)] = client
            logger.debug("LSP session started: %s (%s)", lang_id, rl.server_name)
            return client

    def _record_for_client(self, client: LspClient) -> Optional[_LspSession]:
        for record in self._session_records.values():
            if record.client is client:
                return record
        return None

    @staticmethod
    def _parse_diagnostic_items(items: list[dict[str, Any]], *, source: Optional[str] = None) -> list[LspDiagnostic]:
        parsed: list[LspDiagnostic] = []
        for item in items:
            parsed.append(
                LspDiagnostic(
                    range=item.get("range", {}),
                    severity=item.get("severity"),
                    code=str(item.get("code")) if item.get("code") else None,
                    message=item.get("message", ""),
                    source=item.get("source") or source,
                )
            )
        return parsed

    @staticmethod
    def _diagnostic_to_wire(diag: LspDiagnostic) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "range": diag.range,
            "message": diag.message,
        }
        if diag.severity is not None:
            payload["severity"] = diag.severity
        if diag.code is not None:
            payload["code"] = diag.code
        if diag.source is not None:
            payload["source"] = diag.source
        return payload

    async def _pull_diagnostics(
        self, client: LspClient, uri: str, *, source: Optional[str] = None
    ) -> Optional[list[LspDiagnostic]]:
        try:
            result = await client.request(
                "textDocument/diagnostic",
                {"textDocument": {"uri": uri}},
            )
        except LspClientError:
            return None

        if not isinstance(result, dict) or "items" not in result:
            return None
        items = result.get("items")
        if not isinstance(items, list):
            return None
        return self._parse_diagnostic_items(items, source=source)

    async def _wait_for_push_diagnostics(self, session_key: str, uri: str, *, timeout: float) -> None:
        event_key = (session_key, uri)
        self._diag_events[event_key] = asyncio.Event()
        try:
            await asyncio.wait_for(self._diag_events[event_key].wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            self._diag_events.pop(event_key, None)

    def _diagnostics_for(self, session_key: str, uri: str) -> list[LspDiagnostic]:
        record = self._session_records.get(session_key)
        if record is None:
            return list(self._diagnostics.get(uri, []))
        return list(record.diagnostics.get(uri, []))

    def _supports_method(self, client: LspClient, capability_path: tuple[str, ...], method: str) -> bool:
        return self._has_capability(client, *capability_path) or method in self._registered_capabilities

    def _formatting_options(self, path: str) -> dict[str, Any]:
        tab_size = 4
        insert_spaces = True
        try:
            text = (self._project_path / path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        for line in text.splitlines():
            stripped = line.lstrip(" \t")
            if not stripped or stripped == line:
                continue
            indent = line[: len(line) - len(stripped)]
            if "\t" in indent:
                insert_spaces = False
                tab_size = 4
                break
            tab_size = min(max(len(indent), 2), 8)
            break
        return {
            "tabSize": tab_size,
            "insertSpaces": insert_spaces,
            "trimTrailingWhitespace": True,
            "insertFinalNewline": True,
            "trimFinalNewlines": True,
        }

    async def _clients_for_file_operation(self, *paths: str) -> list[LspClient]:
        clients: dict[int, LspClient] = {id(client): client for client in self._sessions.values() if client.running}

        for path in paths:
            lang_id = self._language_for_path(path)
            if not lang_id:
                continue

            rl = self._resolved.get(lang_id) or self._missing.get(lang_id)
            if rl is None:
                continue

            effective_id = lang_id
            spec = self._registry.get(lang_id)
            if spec and spec.family:
                effective_id = spec.family

            try:
                client = await self._get_session(effective_id, rl)
            except Exception:
                logger.exception("Failed to start LSP session for file operation on %s", path)
                continue
            if client is not None and client.running:
                clients[id(client)] = client

        return list(clients.values())

    def _aggregate_diagnostics(self, uri: str) -> list[LspDiagnostic]:
        merged: list[LspDiagnostic] = []
        for record in self._session_records.values():
            merged.extend(record.diagnostics.get(uri, []))
        return merged

    async def _get_extra_diagnostics(self, path: str, lang_id: str, uri: str) -> list[LspDiagnostic]:
        """Get diagnostics from extra ``diagnostic_servers`` configured for the language.

        Returns an empty list when no extra servers are configured or available.
        """
        if not self._config.diagnostic_servers:
            return []

        spec = self._registry.get(lang_id)
        if not spec:
            return []

        import shutil

        all_diagnostics: list[LspDiagnostic] = []

        for extra_name in self._config.diagnostic_servers:
            # Find the server spec by name in this language's available servers
            server_spec = None
            for srv in spec.language_servers:
                if srv.name == extra_name:
                    server_spec = srv
                    break
            if server_spec is None:
                continue

            # Check if the binary is on PATH
            resolved_bin = shutil.which(server_spec.bin)
            if not resolved_bin:
                continue

            # Build a ResolvedLanguage for the extra server
            extra_rl = ResolvedLanguage(
                language_id=lang_id,
                display_name=spec.display_name,
                detection_reason="extra diagnostic server",
                server_name=server_spec.name,
                server_bin=resolved_bin,
                server_args=list(server_spec.args),
                install_commands=[],
                alternatives=[],
                family=None,
                env=dict(server_spec.env),
                initialization_options=self._registry.initialization_options_for(server_spec.name),
            )

            session_key = f"{lang_id}:{extra_name}"
            try:
                client = await self._get_session(lang_id, extra_rl, session_key=session_key)
                if client is None:
                    continue
            except Exception:
                logger.exception("Failed to start extra diagnostic server %s", extra_name)
                continue

            # Ensure document is open in the extra server
            try:
                await self._ensure_document_open(client, uri, path, lang_id, session_key=session_key)
            except LspClientError:
                pass

            # Try pull diagnostics
            pulled = await self._pull_diagnostics(client, uri, source=extra_name)
            if pulled is not None:
                all_diagnostics.extend(pulled)
                continue

            await self._wait_for_push_diagnostics(session_key, uri, timeout=3.0)
            all_diagnostics.extend(self._diagnostics_for(session_key, uri))

        return all_diagnostics

    def _next_version(self, uri: str, *, session_key: Optional[str] = None) -> int:
        """Return the next incrementing document version for *uri*."""
        record = self._session_records.get(session_key or "")
        if record is not None:
            record.doc_versions[uri] = record.doc_versions.get(uri, 0) + 1
            self._doc_versions[uri] = max(self._doc_versions.get(uri, 0), record.doc_versions[uri])
            return record.doc_versions[uri]
        self._doc_versions[uri] = self._doc_versions.get(uri, 0) + 1
        return self._doc_versions[uri]

    async def _ensure_document_open(
        self,
        client: LspClient,
        uri: str,
        path: str,
        lang_id: str,
        *,
        session_key: Optional[str] = None,
    ) -> None:
        """Ensure the document is open in the language server (didOpen or didChange)."""
        try:
            full_path = self._project_path / path
            text = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return

        record = self._session_records.get(session_key or "") or self._record_for_client(client)
        opened_docs = record.opened_docs if record is not None else self._open_files

        if uri in opened_docs:
            # Already open — send didChange with incrementing version
            version = self._next_version(uri, session_key=record.key if record else session_key)
            await client.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": text}],
                },
            )
        else:
            # Send didOpen
            version = self._next_version(uri, session_key=record.key if record else session_key)
            await client.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": lang_id,
                        "version": version,
                        "text": text,
                    },
                },
            )
            opened_docs[uri] = lang_id
            self._open_files[uri] = lang_id

    def _language_for_path(self, path: str) -> Optional[str]:
        """Determine the language_id for a file path."""
        import os

        filename = os.path.basename(path)
        _, ext = os.path.splitext(filename)

        # Exact filename match (e.g. "Dockerfile")
        lang = self._registry.language_for_filename(filename)
        if lang:
            return lang

        # Extension match
        if ext:
            lang = self._registry.language_for_extension(ext.lower())
            if lang:
                return lang

        return None

    def _on_publish_diagnostics(self, session_key: str, params: dict) -> None:
        """Handle ``textDocument/publishDiagnostics`` notification."""
        parsed = parse_publish_diagnostics(params)
        record = self._session_records.get(session_key)
        diagnostics = list(parsed.diagnostics)
        if record is not None:
            for diag in diagnostics:
                if not diag.source:
                    diag.source = record.server_name
            record.diagnostics[parsed.uri] = diagnostics
        aggregate = self._aggregate_diagnostics(parsed.uri) if record is not None else diagnostics
        self._diagnostics[parsed.uri] = aggregate
        self.last_diagnostic_count[parsed.uri] = len(aggregate)
        # Track the document version the server is responding to (if provided)
        version = params.get("version")
        if record is not None:
            record.diag_versions[parsed.uri] = version
        self._diag_versions[parsed.uri] = version
        event = self._diag_events.get((session_key, parsed.uri))
        if event:
            event.set()

    # -- server→client request handlers ------------------------

    def _register_server_request_handlers(self, client: LspClient, *, server_name: str) -> None:
        """Register handlers for server→client requests so they get responses."""
        root_uri = _path_to_uri(self._project_path, str(self._project_path))

        def _workspace_configuration(params: dict) -> list:
            server_config = self._config.workspace_configuration.get(server_name, {})
            items = params.get("items", [])
            if not isinstance(items, list):
                return []
            results = []
            for item in items:
                section = item.get("section") if isinstance(item, dict) else None
                if section is None:
                    results.append(server_config)
                else:
                    results.append(server_config.get(section, {}))
            return results

        def _workspace_folders(params: dict) -> list:
            return [{"uri": root_uri, "name": "workspace"}]

        def _work_done_progress_create(params: dict):
            return None

        def _register_capability(params: dict):
            for reg in params.get("registrations", []):
                method = reg.get("method", "")
                self._registered_capabilities[method] = reg
            return None

        def _unregister_capability(params: dict):
            for unreg in params.get("unregisterations", []):
                method = unreg.get("method", "")
                self._registered_capabilities.pop(method, None)
            return None

        def _apply_edit(params: dict):
            if self._workspace_apply_edit_handler is not None:
                return self._workspace_apply_edit_handler(params)
            # Read-only default: do not apply edits.
            return {"applied": False}

        client.on_request("workspace/configuration", _workspace_configuration)
        client.on_request("workspace/workspaceFolders", _workspace_folders)
        client.on_request("window/workDoneProgress/create", _work_done_progress_create)
        client.on_request("client/registerCapability", _register_capability)
        client.on_request("client/unregisterCapability", _unregister_capability)
        client.on_request("workspace/applyEdit", _apply_edit)

    # -- session resolution helpers ------------------------------

    def _resolve_session(self, path: str) -> tuple[Optional[str], Optional[LspClient], Optional[ResolvedLanguage]]:
        """Resolve the (lang_id, client, resolved_language) for *path*.

        Returns ``(None, None, None)`` if no server is available.
        """
        lang_id = self._language_for_path(path)
        if not lang_id:
            return None, None, None

        rl = self._resolved.get(lang_id) or self._missing.get(lang_id)
        if rl is None:
            return lang_id, None, None

        # Resolve family (e.g. typescript → javascript)
        effective_id = lang_id
        spec = self._registry.get(lang_id)
        if spec and spec.family:
            effective_id = spec.family

        client = self._sessions.get(effective_id)
        if client and not client.running:
            client = None

        return effective_id, client, rl

    async def _get_or_start_session(self, path: str) -> tuple[Optional[str], Optional[LspClient]]:
        """Resolve and optionally start the session for *path*.

        Returns ``(effective_lang_id, client)`` or ``(lang_id, None)`` on failure.
        """
        lang_id = self._language_for_path(path)
        if not lang_id:
            return None, None

        rl = self._resolved.get(lang_id) or self._missing.get(lang_id)
        if rl is None:
            return lang_id, None

        effective_id = lang_id
        spec = self._registry.get(lang_id)
        if spec and spec.family:
            effective_id = spec.family

        try:
            client = await self._get_session(effective_id, rl)
        except Exception:
            logger.exception("Failed to start LSP session for %s", effective_id)
            client = None

        return effective_id, client

    def _resolve_position(self, path: str, line_1based: int, symbol: str) -> tuple[int, int]:
        """Resolve a 1-based line + symbol name to an LSP ``(line, character)`` position.

        Supports ``name#N`` syntax for the Nth occurrence of *symbol* on the line.
        Note: identifiers containing ``#`` are not supported — a trailing
        ``#<digits>`` is always interpreted as an occurrence selector.

        Raises:
            ValueError: if the symbol is not found on the specified line.
        """
        # Parse name#N suffix. Only treat a trailing "#<digits>" as an occurrence
        # selector when the part before "#" is non-empty (so a bare "#123" or a
        # symbol that is purely numeric is not mis-parsed).
        occurrence = 1
        actual_symbol = symbol
        if "#" in symbol:
            parts = symbol.rsplit("#", 1)
            if parts[0] and parts[1].isdigit():
                actual_symbol = parts[0]
                occurrence = int(parts[1])

        try:
            full_path = self._project_path / path
            text = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise ValueError(f"Cannot read {path}: {exc}") from exc

        lines = text.split("\n")
        line_idx = line_1based - 1
        if line_idx < 0 or line_idx >= len(lines):
            raise ValueError(f"Line {line_1based} is out of range (file has {len(lines)} lines).")

        line_text = lines[line_idx]
        # Find the Nth occurrence of the symbol on this line
        start = 0
        found_char = -1
        for _i in range(occurrence):
            found_char = line_text.find(actual_symbol, start)
            if found_char == -1:
                if occurrence > 1:
                    raise ValueError(
                        f"Symbol '{actual_symbol}' occurrence #{occurrence} not found on line {line_1based}."
                    )
                raise ValueError(f"Symbol '{actual_symbol}' not found on line {line_1based}.")
            start = found_char + len(actual_symbol)

        return (line_idx, self._utf16_character_offset(line_text, found_char))

    @staticmethod
    def _utf16_character_offset(text: str, python_offset: int) -> int:
        return sum(2 if ord(char) > 0xFFFF else 1 for char in text[:python_offset])

    @staticmethod
    def _has_capability(client: Optional[LspClient], *keys: str) -> bool:
        """Check whether the server advertises a capability at ``capabilities.*keys``.

        Per the LSP spec, a capability may be a boolean ``true`` **or** an options
        object (including an empty ``{}``). Only ``false`` and absence (``None``)
        count as "not supported".
        """
        if client is None or client.server_capabilities is None:
            return False
        node: Any = client.server_capabilities
        for key in keys:
            if not isinstance(node, dict):
                return False
            node = node.get(key)
        return node is not None and node is not False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _path_to_uri(project_path: Path, relative_path: str) -> str:
    """Convert a project-relative path to a ``file://`` URI."""
    if Path(relative_path).is_absolute():
        abs_path = Path(relative_path).resolve()
    else:
        abs_path = (project_path / relative_path).resolve()
    return abs_path.as_uri()


# Maps .kolega/lsp.json keys to LspConfig dataclass fields. Keys absent from this
# map (notably ``enabled``) are intentionally ignored by the merger.
_PROJECT_CONFIG_FIELD_MAP: dict[str, str] = {
    "auto_diagnostics_on_edit": "auto_diagnostics_on_edit",
    "max_diagnostics": "max_diagnostics",
    "auto_fallback": "auto_fallback",
    "prompt_on_missing": "prompt_on_missing",
    "disabled_languages": "disabled_languages",
    "preferences": "preferences",
    "servers": "custom_servers",
    "initialization_options": "initialization_options",
    "diagnostic_servers": "diagnostic_servers",
    "workspace_configuration": "workspace_configuration",
}


def _merge_lsp_config(base: LspConfig, overrides: dict[str, Any]) -> LspConfig:
    """Overlay validated project-config *overrides* onto *base*.

    Only keys present in *overrides* are applied; all other fields keep *base*'s
    value (so a project file can no longer silently discard user-level settings).

    The master kill-switch (``enabled``) is **always** taken from *base*: a
    committed ``.kolega/lsp.json`` cannot enable or disable LSP — only the user's
    setting (via CLI/TUI) controls that.
    """
    merged = replace(base)
    for json_key, value in overrides.items():
        if json_key == "enabled":
            # Never let a project file flip the kill-switch.
            continue
        field_name = _PROJECT_CONFIG_FIELD_MAP.get(json_key)
        if field_name is None:
            continue
        setattr(merged, field_name, value)
    return merged
