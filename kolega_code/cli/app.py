"""Textual application for Kolega Code."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Select, Static, TabPane, TabbedContent

from kolega_code.agent import AgentConfig, AgentEvent, CoderAgent
from kolega_code.agent.llm.models import Message, TextBlock, ToolCall, ToolResult
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.services.browser import PlaywrightBrowserManager

from .config import CliConfigError, CliConfigOverrides, build_agent_config, config_summary, key_status
from .connection import CliConnectionManager
from .provider_registry import UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER, get_ui_model, ui_model_options, ui_provider_options
from .session_store import SessionRecord, SessionStore
from .settings import CliSettings, SettingsStore

TOOL_RESULT_PREVIEW_CHARS = 500
CLI_AGENT_MODE = AgentMode.CLI.value
COMPOSER_PLACEHOLDER = "Ask Kolega Code..."


@dataclass
class ConversationEntry:
    kind: str
    content: str
    complete: bool = True
    uuid: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None


class KolegaCodeApp(App):
    """Interactive terminal UI for Kolega Code."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        height: 1fr;
    }

    #conversation_panel {
        width: 2fr;
        height: 100%;
    }

    #side_panel {
        width: 1fr;
        min-width: 34;
        height: 100%;
    }

    #conversation, #logs, #terminal, #status {
        height: 1fr;
        border: round $surface;
    }

    #settings_form {
        height: 1fr;
        padding: 1;
    }

    #settings_status {
        margin-top: 1;
    }

    #composer {
        dock: bottom;
        height: 3;
    }

    .meta {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_generation", "Cancel", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        project_path: Path,
        mode: str,
        store: SessionStore,
        session: SessionRecord,
        config: Optional[AgentConfig] = None,
        settings_store: Optional[SettingsStore] = None,
        overrides: Optional[CliConfigOverrides] = None,
        browser_visible: bool = False,
    ) -> None:
        super().__init__()
        self.project_path = project_path
        self.config = config
        self.mode = CLI_AGENT_MODE
        self.store = store
        self.session = session
        self.session.mode = CLI_AGENT_MODE
        self.settings_store = settings_store or SettingsStore(store.root)
        self.overrides = overrides or CliConfigOverrides()
        self.settings: CliSettings = CliSettings()
        self.browser_visible = browser_visible
        self.connection_manager = CliConnectionManager()
        self.agent: Optional[CoderAgent] = None
        self.agent_worker = None
        self.conversation_entries: list[ConversationEntry] = []
        self._stream_entries: dict[str, ConversationEntry] = {}
        self._tool_entries: dict[str, ConversationEntry] = {}
        self._active_progress_entry: Optional[ConversationEntry] = None
        self._turn_active = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="conversation_panel"):
                yield Static(
                    f"{self.project_path} | session {self.session.session_id} | {self.mode}",
                    classes="meta",
                )
                yield RichLog(id="conversation", wrap=True, markup=True, highlight=True)
                yield Input(placeholder=COMPOSER_PLACEHOLDER, id="composer")
            with Vertical(id="side_panel"):
                with TabbedContent(id="events"):
                    with TabPane("Logs"):
                        yield RichLog(id="logs", wrap=True, markup=True)
                    with TabPane("Terminal"):
                        yield RichLog(id="terminal", wrap=True, markup=False)
                    with TabPane("Status"):
                        yield RichLog(id="status", wrap=True, markup=True)
                    with TabPane("Settings", id="settings_pane"):
                        with Vertical(id="settings_form"):
                            yield Label("Provider")
                            yield Select(
                                ui_provider_options(),
                                id="provider_select",
                                allow_blank=False,
                                value=UI_DEFAULT_PROVIDER,
                            )
                            yield Label("Model")
                            yield Select(
                                ui_model_options(UI_DEFAULT_PROVIDER),
                                id="model_select",
                                allow_blank=False,
                                value=UI_DEFAULT_MODEL,
                            )
                            yield Label("API key")
                            yield Input(password=True, id="api_key_input")
                            yield Button("Save Settings", variant="primary", id="save_settings")
                            yield Static("", id="settings_status")
        yield Footer()

    async def on_mount(self) -> None:
        self.settings = self.settings_store.load()
        self._populate_settings_controls()
        self._ensure_startup_entry()
        self.run_worker(self._consume_events(), name="kolega-events", group="events")
        if self.config is not None:
            await self._build_agent(self.config)
            self._set_chat_enabled(True)
            self.query_one("#composer", Input).focus()
        else:
            await self._ensure_agent_from_settings()

    @property
    def _conversation(self) -> RichLog:
        return self.query_one("#conversation", RichLog)

    @property
    def _logs(self) -> RichLog:
        return self.query_one("#logs", RichLog)

    @property
    def _terminal(self) -> RichLog:
        return self.query_one("#terminal", RichLog)

    @property
    def _status(self) -> RichLog:
        return self.query_one("#status", RichLog)

    @property
    def _settings_status(self) -> Static:
        return self.query_one("#settings_status", Static)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.agent is None:
            if text:
                self._settings_status.update("Save a provider, model, and API key before chatting.")
            return
        event.input.value = ""
        self._add_conversation_entry(ConversationEntry(kind="user", content=text))
        self.agent_worker = self.run_worker(self._process_message(text), name="kolega-turn", group="turns", exclusive=True)

    async def _process_message(self, message: str) -> None:
        if self.agent is None:
            return
        self._begin_turn_progress()
        self._status.write("[green]Generating...[/green]")
        try:
            async for chunk in self.agent.process_message_stream(message):
                if chunk.get("type") == "response":
                    if chunk.get("content"):
                        self._update_progress("Reading model response...", complete=False)
                    self._apply_stream_chunk(chunk, kind="assistant")
                    continue

                content = chunk.get("content")
                if chunk.get("type") == "thinking":
                    self._update_progress("Thinking...", complete=False)
                    self._apply_stream_chunk(chunk, kind="thinking")
                    if content:
                        self._status.write(f"[dim]{content}[/dim]")
            await self._drain_pending_events()
            self._save_session_history()
            self._finish_turn_progress("Finished")
            self._status.write("[green]Finished[/green]")
        except asyncio.CancelledError:
            await self._drain_pending_events()
            self._save_session_history()
            self._finish_turn_progress("Stopped by user")
            self._status.write("[yellow]Stopped by user.[/yellow]")
        except Exception as exc:
            await self._drain_pending_events()
            self._save_session_history()
            self._finish_turn_progress(f"Stopped due to error: {exc}")
            self._status.write(f"[red]Stopped due to error: {escape(str(exc))}[/red]")
            raise
        finally:
            self._active_progress_entry = None
            self._turn_active = False
            self.agent_worker = None
            self._restore_composer_placeholder()
            self._set_chat_enabled(self.agent is not None)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save_settings":
            await self._save_settings_from_ui()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "provider_select":
            return
        provider = str(event.value)
        model_select = self.query_one("#model_select", Select)
        model_options = ui_model_options(provider)
        model_select.set_options(model_options)
        if model_options:
            model_select.value = model_options[0][1]
        api_key_input = self.query_one("#api_key_input", Input)
        api_key_input.placeholder = self._api_key_placeholder(provider)

    async def _consume_events(self) -> None:
        while True:
            event = await self.connection_manager.next_event()
            self._render_event(event)

    async def _drain_pending_events(self) -> None:
        while True:
            try:
                event = self.connection_manager.events.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._render_event(event)

    def _render_event(self, event: AgentEvent) -> None:
        text = event.content.get("text") or event.content.get("message") or ""
        if event.event_type == "log_message":
            level = event.content.get("level", "info")
            self._logs.write(f"[{level}] {text}")
        elif event.event_type == "terminal_output":
            self._terminal.write(event.content.get("output", ""))
        elif event.event_type == "terminal_command":
            command = str(event.content.get("command") or "")
            self._terminal.write(f"$ {command}")
            if command:
                self._update_activity_progress("Running terminal command...")
        elif event.event_type == "chat_message":
            message_text = event.content.get("text", "")
            message_type = event.content.get("message_type", "message")
            if message_type in {"tool_call", "tool_result", "tool_error"}:
                self._add_tool_message(message_type, event.content)
            elif message_text:
                self._add_conversation_entry(ConversationEntry(kind="message", content=message_text))
        elif event.event_type == "tool_streaming_update":
            self._apply_tool_streaming_update(event.content)
        elif event.event_type in {"llm_status_update", "status_update", "llm_context_update"}:
            self._status.write(str(event.content))
            status_text = text or str(event.content)
            if status_text:
                self._update_activity_progress(status_text)
        else:
            self._status.write(f"{event.event_type}: {event.content}")

    def action_cancel_generation(self) -> None:
        if self.agent_worker is not None:
            self._update_progress("Stop requested...", complete=False)
            self.agent_worker.cancel()
            self._status.write("[yellow]Cancellation requested.[/yellow]")

    async def action_quit(self) -> None:
        if self.agent is not None:
            self.session.history = self.agent.dump_message_history()
            self.store.save(self.session)
            await self.agent.cleanup()
        self.exit()

    def _populate_settings_controls(self) -> None:
        if not self.settings.active_provider:
            self.settings.active_provider = UI_DEFAULT_PROVIDER
        provider = self.settings.active_provider
        model_options = ui_model_options(provider)
        valid_models = {value for _, value in model_options}
        if not self.settings.active_model or self.settings.active_model not in valid_models:
            self.settings.active_model = model_options[0][1] if model_options else UI_DEFAULT_MODEL
        model = self.settings.active_model
        provider_select = self.query_one("#provider_select", Select)
        model_select = self.query_one("#model_select", Select)
        api_key_input = self.query_one("#api_key_input", Input)

        provider_select.value = provider
        model_select.set_options(model_options)
        model_select.value = model
        api_key_input.placeholder = self._api_key_placeholder(provider)
        self._update_settings_status()

    async def _save_settings_from_ui(self) -> None:
        provider = str(self.query_one("#provider_select", Select).value)
        model = str(self.query_one("#model_select", Select).value)
        api_key_input = self.query_one("#api_key_input", Input)
        api_key = api_key_input.value.strip()

        self.settings.active_provider = provider
        self.settings.active_model = model
        if api_key:
            self.settings.set_api_key(provider, api_key)
        self.settings_store.save(self.settings)
        api_key_input.value = ""
        api_key_input.placeholder = self._api_key_placeholder(provider)

        await self._ensure_agent_from_settings(rebuild=True)

    async def _ensure_agent_from_settings(self, rebuild: bool = False) -> None:
        try:
            config = build_agent_config(self.project_path, self.overrides, settings=self.settings)
        except CliConfigError as exc:
            self._set_chat_enabled(False)
            self._settings_status.update(f"Configuration incomplete: {exc}")
            self.query_one("#events", TabbedContent).active = "settings_pane"
            return

        self.config = config
        self.session.config = config_summary(config)
        self.store.save(self.session)
        await self._build_agent(config, rebuild=rebuild)
        self._set_chat_enabled(True)
        self._update_settings_status()
        self.query_one("#composer", Input).focus()

    async def _build_agent(self, config: AgentConfig, rebuild: bool = False) -> None:
        history = self.session.history
        if self.agent is not None:
            history = self.agent.dump_message_history()
            self.session.history = history
            self.store.save(self.session)
            if rebuild:
                await self.agent.cleanup()

        browser_manager = PlaywrightBrowserManager()
        browser_manager.headless = not self.browser_visible
        self.agent = CoderAgent(
            project_path=self.project_path,
            workspace_id=self.session.workspace_id,
            thread_id=self.session.thread_id,
            connection_manager=self.connection_manager,
            config=config,
            browser_manager=browser_manager,
            agent_mode=AgentMode(self.mode),
        )
        if history:
            self.agent.restore_message_history(history)
            self._restore_conversation_history(history)

    def _set_chat_enabled(self, enabled: bool) -> None:
        composer = self.query_one("#composer", Input)
        composer.disabled = not enabled

    def _set_composer_status(self, status: str) -> None:
        self.query_one("#composer", Input).placeholder = status

    def _restore_composer_placeholder(self) -> None:
        self.query_one("#composer", Input).placeholder = COMPOSER_PLACEHOLDER

    def _update_settings_status(self) -> None:
        provider = self.settings.active_provider or UI_DEFAULT_PROVIDER
        model = self.settings.active_model or UI_DEFAULT_MODEL
        status = key_status(provider, self.project_path, self.settings)
        self._settings_status.update(f"Active model: {provider}/{model}\nAPI key: {status}")

    def _api_key_placeholder(self, provider: str) -> str:
        if self.settings.has_api_key(provider):
            return "Stored API key will be kept if blank"
        model = get_ui_model(provider, (ui_model_options(provider) or [("", "")])[0][1])
        return f"{model.provider_label} API key" if model else "API key"

    def _add_conversation_entry(self, entry: ConversationEntry) -> None:
        self.conversation_entries.append(entry)
        if entry.uuid:
            self._stream_entries[entry.uuid] = entry
        if entry.tool_call_id:
            self._tool_entries[entry.tool_call_id] = entry
        self._render_conversation()

    def _ensure_startup_entry(self, *, render: bool = True) -> None:
        existing = next((entry for entry in self.conversation_entries if entry.kind == "startup"), None)
        if existing is None:
            self.conversation_entries.insert(0, ConversationEntry(kind="startup", content=self._startup_content()))
        else:
            existing.content = self._startup_content()
            if self.conversation_entries[0] is not existing:
                self.conversation_entries.remove(existing)
                self.conversation_entries.insert(0, existing)
        if render:
            self._render_conversation()

    def _startup_content(self) -> str:
        session_id = str(self.session.session_id)[:8]
        return "\n".join(
            [
                "Kolega Code",
                f"Project: {self.project_path}",
                f"Session: {session_id} - mode {self.mode}",
                "Type a request below. Press Ctrl+C to stop a turn.",
            ]
        )

    def _restore_conversation_history(self, history: list[dict]) -> None:
        self.conversation_entries = []
        self._stream_entries = {}
        self._tool_entries = {}
        self._active_progress_entry = None
        self._ensure_startup_entry(render=False)
        for item in history:
            try:
                message = Message.from_dict(item)
            except Exception:
                continue
            self.conversation_entries.extend(self._conversation_entries_from_message(message))
        self._render_conversation()

    def _conversation_entries_from_message(self, message: Message) -> list[ConversationEntry]:
        entries: list[ConversationEntry] = []

        if isinstance(message.content, str):
            content = message.content.strip()
            if content:
                entries.append(ConversationEntry(kind=self._entry_kind_for_role(message.role), content=content))
            return entries

        pending_text: list[str] = []

        def flush_text() -> None:
            text = "\n".join(part for part in pending_text if part).strip()
            pending_text.clear()
            if text:
                entries.append(ConversationEntry(kind=self._entry_kind_for_role(message.role), content=text))

        for block in message.content:
            if isinstance(block, TextBlock):
                pending_text.append(block.text)
            elif isinstance(block, ToolCall):
                flush_text()
                entries.append(
                    ConversationEntry(
                        kind="tool_call",
                        content=f"Calling {block.name}",
                        complete=True,
                        tool_name=block.name,
                        tool_call_id=getattr(block, "execution_id", None),
                    )
                )
            elif isinstance(block, ToolResult):
                flush_text()
                text = self._tool_content_to_text(block.content)
                entries.append(
                    ConversationEntry(
                        kind="tool_error" if block.is_error else "tool_result",
                        content=self._truncate_tool_text(text) if block.is_error else self._tool_result_preview(text),
                        tool_name=block.name,
                        tool_call_id=getattr(block, "execution_id", None),
                    )
                )

        flush_text()
        return entries

    def _entry_kind_for_role(self, role: str) -> str:
        if role == "assistant":
            return "assistant"
        if role == "user":
            return "user"
        return "system"

    def _tool_content_to_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n\n".join(
                item.to_markdown() if hasattr(item, "to_markdown") else str(item) for item in content
            )
        return str(content)

    def _apply_stream_chunk(self, chunk: dict, *, kind: str) -> None:
        chunk_uuid = str(chunk.get("uuid") or "")
        content = str(chunk.get("content") or "")
        complete = bool(chunk.get("complete"))

        entry = self._stream_entries.get(chunk_uuid) if chunk_uuid else None
        if entry is None:
            if not content:
                return
            entry = ConversationEntry(kind=kind, content="", complete=complete, uuid=chunk_uuid or None)
            self.conversation_entries.append(entry)
            if chunk_uuid:
                self._stream_entries[chunk_uuid] = entry

        entry.content += content
        entry.complete = complete
        self._render_conversation()

    def _begin_turn_progress(self) -> None:
        self._tool_entries = {}
        self._active_progress_entry = None
        self._turn_active = True
        self._set_chat_enabled(False)
        self._update_progress("Agent is working...", complete=False)

    def _update_progress(self, content: str, complete: bool) -> None:
        if complete:
            if content != "Finished":
                self._add_conversation_entry(ConversationEntry(kind="progress", content=content, complete=True))
            self._restore_composer_placeholder()
            return
        self._set_composer_status(content)

    def _update_activity_progress(self, content: str) -> None:
        if self._turn_active:
            self._update_progress(content, complete=False)

    def _finish_turn_progress(self, content: str) -> None:
        self._update_progress(content, complete=True)

    def _save_session_history(self) -> None:
        if self.agent is None:
            return
        self.session.history = self.agent.dump_message_history()
        self.store.save(self.session)

    def _add_tool_message(self, message_type: str, content: dict) -> None:
        tool_name = str(content.get("tool_description") or content.get("tool_name") or "tool")
        tool_call_id = str(content.get("tool_call_id") or "")
        text = str(content.get("text") or "")
        entry = self._find_tool_entry(tool_call_id, tool_name)

        if message_type == "tool_call":
            entry_content = text or f"Calling {tool_name}"
            complete = False
            self._update_activity_progress(f"Running {tool_name}...")
        elif message_type == "tool_error":
            entry_content = self._truncate_tool_text(text)
            complete = True
            self._update_activity_progress(f"Tool {tool_name} failed.")
        else:
            entry_content = self._tool_result_preview(text)
            complete = True
            self._update_activity_progress(f"Tool {tool_name} completed.")

        if entry is None:
            self._add_conversation_entry(
                ConversationEntry(
                    kind=message_type,
                    content=entry_content,
                    complete=complete,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id or None,
                )
            )
            return

        entry.kind = message_type
        entry.content = entry_content
        entry.complete = complete
        entry.tool_name = tool_name
        entry.tool_call_id = tool_call_id or entry.tool_call_id
        if entry.tool_call_id:
            self._tool_entries[entry.tool_call_id] = entry
        self._render_conversation()

    def _apply_tool_streaming_update(self, content: dict) -> None:
        tool_name = str(content.get("tool_name") or content.get("tool_description") or "tool")
        tool_call_id = str(content.get("tool_call_id") or "")
        text = str(content.get("text") or "")
        is_complete = bool(content.get("is_complete"))
        entry = self._find_tool_entry(tool_call_id, tool_name)
        entry_content = self._tool_result_preview(text) if is_complete else self._truncate_tool_text(text)

        self._update_activity_progress(f"Tool {tool_name} completed." if is_complete else f"Running {tool_name}...")

        if entry is None:
            self._add_conversation_entry(
                ConversationEntry(
                    kind="tool_result" if is_complete else "tool_call",
                    content=entry_content or f"Running {tool_name}",
                    complete=is_complete,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id or None,
                )
            )
            return

        entry.kind = "tool_result" if is_complete else "tool_call"
        entry.content = entry_content or entry.content
        entry.complete = is_complete
        entry.tool_name = tool_name
        entry.tool_call_id = tool_call_id or entry.tool_call_id
        if entry.tool_call_id:
            self._tool_entries[entry.tool_call_id] = entry
        self._render_conversation()

    def _find_tool_entry(self, tool_call_id: str, tool_name: str) -> Optional[ConversationEntry]:
        if tool_call_id and tool_call_id in self._tool_entries:
            return self._tool_entries[tool_call_id]
        for entry in reversed(self.conversation_entries):
            if entry.kind not in {"tool_call", "tool_result", "tool_error"}:
                continue
            if entry.complete:
                continue
            if entry.tool_name == tool_name:
                return entry
        return None

    def _tool_result_preview(self, text: str) -> str:
        if not text:
            return "completed"
        if len(text) <= TOOL_RESULT_PREVIEW_CHARS:
            return f"completed\n{text}"
        return f"completed\n{text[:TOOL_RESULT_PREVIEW_CHARS]}..."

    def _truncate_tool_text(self, text: str) -> str:
        if len(text) <= TOOL_RESULT_PREVIEW_CHARS:
            return text
        return f"{text[:TOOL_RESULT_PREVIEW_CHARS]}..."

    def _render_conversation(self) -> None:
        conversation = self._conversation
        conversation.clear()
        for entry in self.conversation_entries:
            conversation.write(self._format_conversation_entry(entry))

    def _format_conversation_entry(self, entry: ConversationEntry) -> str:
        escaped_content = escape(entry.content)
        if entry.kind == "startup":
            return self._format_startup_entry(entry)
        if entry.kind == "user":
            return f"[bold cyan]You[/bold cyan]\n{escaped_content}"
        if entry.kind == "assistant":
            suffix = "" if entry.complete else "\n[dim]...[/dim]"
            return f"[bold magenta]Agent[/bold magenta]\n{escaped_content}{suffix}"
        if entry.kind == "thinking":
            suffix = "" if entry.complete else "\n[dim]...[/dim]"
            return f"[dim italic]Thinking[/dim italic]\n[italic]{escaped_content}[/italic]{suffix}"
        if entry.kind == "progress":
            suffix = "" if entry.complete else "\n[dim]...[/dim]"
            label_style = "bold red" if "error" in entry.content.lower() else "bold yellow"
            return f"[{label_style}]Status[/]\n{escaped_content}{suffix}"
        if entry.kind == "tool_call":
            return self._format_tool_entry(entry, label="[black on yellow] TOOL [/black on yellow]", state="running")
        if entry.kind == "tool_result":
            return self._format_tool_entry(entry, label="[black on green] TOOL [/black on green]", state="completed")
        if entry.kind == "tool_error":
            return self._format_tool_entry(entry, label="[white on red] TOOL ERROR [/white on red]", state="failed")
        if entry.kind == "system":
            return f"[dim]{escaped_content}[/dim]"
        return escaped_content

    def _format_startup_entry(self, entry: ConversationEntry) -> str:
        lines = entry.content.splitlines()
        title = escape(lines[0]) if lines else "Kolega Code"
        body = "\n".join(escape(line) for line in lines[1:])
        return f"[bold white on blue] {title} [/bold white on blue]\n[dim]{body}[/dim]"

    def _format_tool_entry(self, entry: ConversationEntry, *, label: str, state: str) -> str:
        tool_name = escape(entry.tool_name or "tool")
        body = self._format_inset_content(entry.content)
        return f"{label} [bold]{tool_name}[/bold] [dim]{state}[/dim]\n{body}"

    def _format_inset_content(self, content: str) -> str:
        lines = content.splitlines() or [""]
        return "\n".join(f"[dim]  │[/dim] {escape(line)}" if line else "[dim]  │[/dim]" for line in lines)
