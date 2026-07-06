"""Language server lifecycle manager.

``LspManager`` holds a pool of active ``LspClient`` subprocesses, routes files
to the correct language server, and handles LSP initialization handshakes.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from .client import LspClient, LspClientError, LspDiagnostic, parse_publish_diagnostics
from .config import LspConfig
from .detector import DetectionReport, ResolvedLanguage, detect_languages
from .diagnostics import MissingServer, format_missing_prompt, format_detected_summary
from .registry import LspRegistry, load_project_lsp_config

logger = logging.getLogger(__name__)

# Maximum concurrent server starts to avoid thundering herd
_MAX_CONCURRENT_STARTS = 4


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
                return parsed[: self._config.max_diagnostics]
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

        return self._diagnostics.get(uri, [])[: self._config.max_diagnostics]

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

    def server_for_path(self, path: str) -> Optional[str]:
        """Return the server name used for *path*, or ``None`` if none available."""
        lang_id = self._language_for_path(path)
        if not lang_id:
            return None
        rl = self._resolved.get(lang_id) or self._missing.get(lang_id)
        return rl.server_name if rl else None

    # -- internals ----------------------------------------------------------

    async def _get_session(self, lang_id: str, rl: ResolvedLanguage) -> Optional[LspClient]:
        """Return (or create and initialize) the LspClient for *lang_id*."""
        if lang_id in self._sessions:
            client = self._sessions[lang_id]
            if client.running:
                return client
            # Restart crashed session
            await client.stop()

        if not rl.server_bin:
            return None

        cmd = [rl.server_bin] + list(rl.server_args)
        client = LspClient(cmd)

        try:
            await client.start()
        except Exception as exc:
            logger.warning("Failed to start LS %s for %s: %s", rl.server_name, lang_id, exc)
            return None

        # Register push-diagnostics handler
        client.on_notification("textDocument/publishDiagnostics", self._on_publish_diagnostics)

        # LSP initialize handshake
        try:
            _ = await client.request(
                "initialize",
                {
                    "processId": None,
                    "rootUri": _path_to_uri(self._project_path, str(self._project_path)),
                    "capabilities": {
                        "textDocument": {
                            "diagnostic": {"dynamicRegistration": True},
                            "publishDiagnostics": {},
                        },
                    },
                    "workspaceFolders": [
                        {"uri": _path_to_uri(self._project_path, str(self._project_path)), "name": "workspace"}
                    ],
                },
            )
            await client.notify("initialized", {})
        except LspClientError as exc:
            logger.warning("LSP initialize failed for %s: %s", lang_id, exc)
            await client.stop()
            return None

        self._sessions[lang_id] = client
        logger.debug("LSP session started: %s (%s)", lang_id, rl.server_name)
        return client

    async def _ensure_document_open(self, client: LspClient, uri: str, path: str, lang_id: str) -> None:
        """Ensure the document is open in the language server (didOpen or didChange)."""
        try:
            full_path = self._project_path / path
            text = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return

        if uri in self._open_files:
            # Already open — send didChange
            await client.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": 2},
                    "contentChanges": [{"text": text}],
                },
            )
        else:
            # Send didOpen
            await client.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": lang_id,
                        "version": 1,
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
        event = self._diag_events.get(parsed.uri)
        if event:
            event.set()


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
