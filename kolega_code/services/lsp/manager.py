"""Language server lifecycle manager.

``LspManager`` holds a pool of active ``LspClient`` subprocesses, routes files
to the correct language server, and handles LSP initialization handshakes.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

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
    },
    "workspace": {
        "symbol": {},
        "configuration": True,
        "workspaceFolders": True,
    },
}


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
    ) -> None:
        self._project_path = Path(project_path).resolve()
        self._config = config or LspConfig()
        self._registry = LspRegistry(config=self._config)

        # Per-language (or family) LspClient sessions
        self._sessions: dict[str, LspClient] = {}
        # Maps file URI to language_id for tracking open documents
        self._open_files: dict[str, str] = {}
        # Latest diagnostics per URI (from publishDiagnostics notifications)
        self._diagnostics: dict[str, list[LspDiagnostic]] = {}
        # Diagnostics events for non-blocking await
        self._diag_events: dict[str, asyncio.Event] = {}
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

            # Merge in project-level config if present
            project_cfg = load_project_lsp_config(self._project_path)
            if project_cfg:
                self._config = project_cfg
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
            await self._ensure_document_open(client, uri, path, lang_id)
        except LspClientError:
            logger.warning("LSP document sync failed for %s", path)
            # Continue — some servers publish diagnostics without explicit didOpen

        # Request pull diagnostics (LSP 3.17)
        try:
            result = await client.request(
                "textDocument/diagnostic",
                {"textDocument": {"uri": uri}},
            )
            items = result.get("items", [])
            if items:
                parsed = [
                    LspDiagnostic(
                        range=item.get("range", {}),
                        severity=item.get("severity"),
                        code=str(item.get("code")) if item.get("code") else None,
                        message=item.get("message", ""),
                        source=item.get("source"),
                    )
                    for item in items
                ]
                extra = await self._get_extra_diagnostics(path, lang_id, uri)
                return dedupe_and_sort(parsed + extra, self._config.max_diagnostics)
        except LspClientError:
            # Pull diagnostics not supported — wait for push notification
            pass

        # Fallback: wait briefly for publishDiagnostics
        self._diag_events[uri] = asyncio.Event()
        try:
            await asyncio.wait_for(self._diag_events[uri].wait(), timeout=3.0)
        except asyncio.TimeoutError:
            pass
        finally:
            self._diag_events.pop(uri, None)

        extra = await self._get_extra_diagnostics(path, lang_id, uri)
        return dedupe_and_sort(self._diagnostics.get(uri, []) + extra, self._config.max_diagnostics)

    async def get_fresh_diagnostics(self, path: str) -> list[LspDiagnostic]:
        """Get diagnostics for *path* after an edit, accepting only fresh results.

        Captures the pre-edit diagnostic snapshot, syncs new content, sends didSave,
        then accepts only diagnostics whose version matches the current document
        version (or falls back to push diagnostics if pull is unsupported).
        """
        if not self._config.enabled:
            return []

        effective_id, client = await self._get_or_start_session(path)
        if client is None:
            return []

        lang_id = self._language_for_path(path)
        if not lang_id:
            return []

        uri = _path_to_uri(self._project_path, path)

        # Snapshot pre-edit version for freshness comparison
        pre_edit_version = self._diag_versions.get(uri)

        # Sync new content (sends didChange with incremented version)
        try:
            await self._ensure_document_open(client, uri, path, lang_id)
        except LspClientError:
            logger.warning("LSP document sync failed for %s", path)

        # Send didSave so servers that trigger on save re-analyze
        try:
            await client.notify("textDocument/didSave", {"textDocument": {"uri": uri}})
        except LspClientError:
            pass

        # Try pull diagnostics first
        try:
            result = await client.request(
                "textDocument/diagnostic",
                {"textDocument": {"uri": uri}},
            )
            items = result.get("items", [])
            if items:
                parsed = [
                    LspDiagnostic(
                        range=item.get("range", {}),
                        severity=item.get("severity"),
                        code=str(item.get("code")) if item.get("code") else None,
                        message=item.get("message", ""),
                        source=item.get("source"),
                    )
                    for item in items
                ]
                return dedupe_and_sort(parsed, self._config.max_diagnostics)
        except LspClientError:
            pass

        # Fallback: wait for fresh push diagnostics
        self._diag_events[uri] = asyncio.Event()
        try:
            await asyncio.wait_for(self._diag_events[uri].wait(), timeout=3.0)
        except asyncio.TimeoutError:
            pass
        finally:
            self._diag_events.pop(uri, None)

        # Accept push diagnostics only if the version is fresh (different from
        # pre-edit, or version tracking is not supported by the server)
        push_version = self._diag_versions.get(uri)
        push_diags = self._diagnostics.get(uri, [])

        if push_version is not None and push_version == pre_edit_version:
            # Server hasn't published fresh diagnostics yet — return what we have
            # but note they may be stale
            return dedupe_and_sort(push_diags, self._config.max_diagnostics)

        return dedupe_and_sort(push_diags, self._config.max_diagnostics)

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
        if client is None:
            return None

        lang_id = self._language_for_path(path)
        if not lang_id:
            return None

        uri = _path_to_uri(self._project_path, path)

        if not self._has_capability(client, *capability_path):
            return None

        try:
            await self._ensure_document_open(client, uri, path, lang_id)
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
        if client is None:
            return None

        lang_id = self._language_for_path(path)
        if not lang_id:
            return None

        uri = _path_to_uri(self._project_path, path)

        if not self._has_capability(client, "documentSymbolProvider"):
            return None

        try:
            await self._ensure_document_open(client, uri, path, lang_id)
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

        sessions = []
        for lang_id, client in self._sessions.items():
            rl = self._resolved.get(lang_id) or self._missing.get(lang_id)
            sessions.append(
                {
                    "language_id": lang_id,
                    "server_name": rl.server_name if rl else "unknown",
                    "status": client.status,
                    "pid": client.server_pid,
                    "connected": client.running and client.status == "initialized",
                    "last_error": client.last_error,
                    "root": client.active_root,
                }
            )

        return {
            "enabled": self._config.enabled,
            "initialized": self._initialized,
            "detected": detected,
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
        if key in self._sessions:
            client = self._sessions[key]
            if client.running:
                return client
            # Restart crashed session
            await client.stop()

        if not rl.server_bin:
            return None

        cmd = [rl.server_bin] + list(rl.server_args)
        client = LspClient(cmd, env=rl.env or None)

        try:
            await client.start()
        except Exception as exc:
            logger.warning("Failed to start LS %s for %s: %s", rl.server_name, lang_id, exc)
            client.last_error = str(exc)
            return None

        # Register push-diagnostics handler
        client.on_notification("textDocument/publishDiagnostics", self._on_publish_diagnostics)

        # Register server→client request handlers
        self._register_server_request_handlers(client)

        root_uri = _path_to_uri(self._project_path, str(self._project_path))
        client.active_root = root_uri

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
            client.server_capabilities = init_result.get("capabilities") if isinstance(init_result, dict) else None
            client.status = "initialized"
        except LspClientError as exc:
            logger.warning("LSP initialize failed for %s: %s", lang_id, exc)
            client.status = "error"
            client.last_error = str(exc)
            await client.stop()
            return None

        self._sessions[key] = client
        logger.debug("LSP session started: %s (%s)", lang_id, rl.server_name)
        return client

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
                await self._ensure_document_open(client, uri, path, lang_id)
            except LspClientError:
                pass

            # Try pull diagnostics
            try:
                result = await client.request(
                    "textDocument/diagnostic",
                    {"textDocument": {"uri": uri}},
                )
                items = result.get("items", [])
                for item in items:
                    all_diagnostics.append(
                        LspDiagnostic(
                            range=item.get("range", {}),
                            severity=item.get("severity"),
                            code=str(item.get("code")) if item.get("code") else None,
                            message=item.get("message", ""),
                            source=item.get("source") or extra_name,
                        )
                    )
            except LspClientError:
                pass

        return all_diagnostics

    def _next_version(self, uri: str) -> int:
        """Return the next incrementing document version for *uri*."""
        self._doc_versions[uri] = self._doc_versions.get(uri, 0) + 1
        return self._doc_versions[uri]

    async def _ensure_document_open(self, client: LspClient, uri: str, path: str, lang_id: str) -> None:
        """Ensure the document is open in the language server (didOpen or didChange)."""
        try:
            full_path = self._project_path / path
            text = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return

        if uri in self._open_files:
            # Already open — send didChange with incrementing version
            version = self._next_version(uri)
            await client.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": text}],
                },
            )
        else:
            # Send didOpen
            version = self._next_version(uri)
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

    def _on_publish_diagnostics(self, params: dict) -> None:
        """Handle ``textDocument/publishDiagnostics`` notification."""
        parsed = parse_publish_diagnostics(params)
        self._diagnostics[parsed.uri] = list(parsed.diagnostics)
        self.last_diagnostic_count[parsed.uri] = len(parsed.diagnostics)
        # Track the document version the server is responding to (if provided)
        version = params.get("version")
        self._diag_versions[parsed.uri] = version
        event = self._diag_events.get(parsed.uri)
        if event:
            event.set()

    # -- server→client request handlers ------------------------

    def _register_server_request_handlers(self, client: LspClient) -> None:
        """Register handlers for server→client requests so they get responses."""
        root_uri = _path_to_uri(self._project_path, str(self._project_path))

        def _workspace_configuration(params: dict) -> list:
            # Return empty list — no config sections configured yet.
            # Servers tolerate this and fall back to defaults.
            return []

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
            # Read-only wave: do not apply edits, acknowledge safely.
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

        Raises:
            ValueError: if the symbol is not found on the specified line.
        """
        # Parse name#N suffix
        occurrence = 1
        actual_symbol = symbol
        if "#" in symbol:
            parts = symbol.rsplit("#", 1)
            if parts[1].isdigit():
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

        return (line_idx, found_char)

    @staticmethod
    def _has_capability(client: Optional[LspClient], *keys: str) -> bool:
        """Check whether the server advertises a capability at ``capabilities.*keys``."""
        if client is None or client.server_capabilities is None:
            return False
        node: Any = client.server_capabilities
        for key in keys:
            if not isinstance(node, dict):
                return False
            node = node.get(key)
        return bool(node)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _path_to_uri(project_path: Path, relative_path: str) -> str:
    """Convert a project-relative path to a ``file://`` URI."""
    from pathlib import PurePosixPath

    if Path(relative_path).is_absolute():
        abs_path = Path(relative_path)
    else:
        abs_path = (project_path / relative_path).resolve()
    # Normalize to posix for URI
    posix = PurePosixPath(abs_path)
    return f"file://{posix}"
