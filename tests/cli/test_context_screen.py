"""Tests for the interactive ``/context`` visualizer modal and slash command."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from rich.text import Text
from textual.containers import Horizontal
from textual.widgets import Static

from kolega_code.agent.conversation import Conversation
from kolega_code.agent.prompt_provider import PromptExtension
from kolega_code.cli import theme
from kolega_code.cli.slash_commands import agent_command_entries
from kolega_code.cli.tui.context_screen import ContextInspectorScreen
from kolega_code.cli.tui.widgets import ChatComposer
from kolega_code.llm.models import ContentBlock, Message, TextBlock, ToolCall, ToolDefinition, ToolResult

from ._app_test_utils import wait_for_layout as _wait_for_layout
from .test_tui_settings_screens import _configured_app


def _plain(widget: Static) -> str:
    rendered = widget.render()
    return rendered.plain if isinstance(rendered, Text) else str(rendered)


def _populate_rich_agent(app) -> None:
    agent = app.agent
    assert agent is not None
    skills_md = "- `deep-research`: Deep multi-source research."
    agent.primary_model_config = SimpleNamespace(provider="anthropic", model="claude-sonnet", thinking_effort="medium")
    agent.agent_name = "coder_agent"
    agent.model_context_length = 200_000
    agent.model_completion_tokens = 64_000
    agent.model_max_input_tokens = 136_000
    agent.strict_context_budget = False
    agent.history_compression_threshold = 0.95
    agent.system_prompt = Message(
        role="system",
        content=[TextBlock(text=f"Core system prompt.\n\n### Agent Skills\n\n{skills_md}")],
    )
    agent.prompt_extensions = [PromptExtension(id="cli-agent-skills", title="Agent Skills", markdown=skills_md)]
    agent.tool_collection = SimpleNamespace(
        lsp_manager=None,
        get_tool_list=lambda: [
            ToolDefinition(name="read", description="Read files.", parameters=[], input_schema={"type": "object"}),
            ToolDefinition(
                name="mcp__github__list_prs",
                description="List GitHub PRs.",
                parameters=[],
                input_schema={"type": "object"},
            ),
        ],
        mcp_tools=["mcp__github__list_prs"],
        extension_tool_sources={"mcp__github__list_prs": "mcp"},
        extension_metadata={"mcp__github__list_prs": {"mcp_server_id": "github", "mcp_tool_id": "list_prs"}},
    )
    conv = Conversation(
        [
            Message(
                role="user",
                content=[
                    TextBlock(
                        text='<system-reminder source="guidance" path="AGENTS.md">\nRepo rules.\n</system-reminder>\n'
                        '<system-reminder source="memory">\n# Memory\n</system-reminder>'
                    )
                ],
            ),
            Message(role="user", content=[TextBlock(text="Inspect context usage please.")]),
            Message(
                role="assistant",
                content=[
                    TextBlock(text="Checking now."),
                    ToolCall(id="c1", name="read", input={"file_path": "README.md"}),
                ],
                usage_metadata={"provider": "anthropic"},
            ),
            Message(
                role="user",
                content=list[ContentBlock](
                    [ToolResult(tool_use_id="c1", name="read", content="README contents\n" * 60, is_error=False)]
                ),
            ),
        ]
    )
    agent.conversation = conv
    agent.history = conv.history
    agent.supports_vision = True
    agent._context_residual_tokens = 0
    agent._last_raw_context_count = None


async def _submit_command(app, pilot, command: str) -> None:
    composer = app.query_one("#composer", ChatComposer)
    composer.load_text(command)
    await app._handle_tui_slash_command(command, composer)
    await pilot.pause()


def test_slash_command_description_override():
    entries = {entry.name: entry for entry in agent_command_entries()}
    assert "context" in entries
    assert "breakdown" in entries["context"].description.lower()


@pytest.mark.asyncio
async def test_context_command_opens_modal_and_esc_closes_without_transcript_entry(
    isolated_cli_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    app, _ = _configured_app(tmp_path, monkeypatch)
    async with app.run_test(size=(110, 36)) as pilot:
        await _wait_for_layout(pilot, lambda: app.agent is not None)
        _populate_rich_agent(app)
        entries_before = len(app.conversation_entries)

        await _submit_command(app, pilot, "/context")
        await _wait_for_layout(
            pilot,
            lambda: (
                isinstance(app.screen, ContextInspectorScreen)
                and bool(app.screen.query("#context_summary"))
                and "Token usage by category" in _plain(app.screen.query_one("#context_summary", Static))
            ),
        )

        screen = app.screen
        assert isinstance(screen, ContextInspectorScreen)
        # No chat turn was injected into the transcript.
        assert len(app.conversation_entries) == entries_before

        grid_text = _plain(screen.query_one("#context_grid", Static))
        summary_text = _plain(screen.query_one("#context_summary", Static))
        details_text = _plain(screen.query_one("#context_details", Static))

        assert len(grid_text.splitlines()) == 10
        assert "anthropic/claude-sonnet (medium)" in summary_text
        assert "System prompt:" in summary_text
        assert "Built-in tools:" in summary_text
        assert "MCP tools:" in summary_text
        assert "Guidance & memory:" in summary_text
        assert "Tool calls & results:" in summary_text
        assert "Free space:" in summary_text
        assert "Auto-compact buffer:" in summary_text
        assert "AGENTS.md" in details_text
        assert "github" in details_text
        assert "read" in details_text

        await pilot.press("escape")
        await _wait_for_layout(pilot, lambda: not isinstance(app.screen, ContextInspectorScreen))
        assert app._context_screen is None


@pytest.mark.asyncio
async def test_context_modal_copy_and_refresh_and_compress_shortcuts(
    isolated_cli_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    app, _ = _configured_app(tmp_path, monkeypatch)
    copied: list[str] = []
    compress_started: list[bool] = []
    monkeypatch.setattr(app, "copy_to_clipboard", lambda text: copied.append(text))
    monkeypatch.setattr(app, "_start_compress_command", lambda: compress_started.append(True))

    async with app.run_test(size=(110, 36)) as pilot:
        await _wait_for_layout(pilot, lambda: app.agent is not None)
        _populate_rich_agent(app)
        assert app.agent is not None
        app.agent.count_current_context = AsyncMock(return_value=SimpleNamespace(input_tokens=12_345))

        await _submit_command(app, pilot, "/context")
        await _wait_for_layout(
            pilot,
            lambda: (
                isinstance(app.screen, ContextInspectorScreen)
                and "Token usage by category" in _plain(app.screen.query_one("#context_summary", Static))
            ),
        )

        # Copy report
        await pilot.press("y")
        await _wait_for_layout(pilot, lambda: len(copied) == 1)
        assert "Token usage by category:" in copied[0]
        assert "anthropic/claude-sonnet (medium)" in copied[0]

        # Explicit refresh queries provider count when idle
        await pilot.press("r")
        await _wait_for_layout(
            pilot,
            lambda: "12.3k/136k tokens" in _plain(app.screen.query_one("#context_summary", Static)),
        )

        # Compress dismisses modal and delegates to app
        await pilot.press("c")
        await _wait_for_layout(pilot, lambda: not isinstance(app.screen, ContextInspectorScreen))
        assert compress_started == [True]


@pytest.mark.asyncio
async def test_context_modal_disconnected_and_live_turn_guards(
    isolated_cli_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    app, _ = _configured_app(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 32)) as pilot:
        await _wait_for_layout(pilot, lambda: app.agent is not None)
        _populate_rich_agent(app)
        assert app.agent is not None

        # Mid-turn live snapshot: opens without calling count_current_context and blocks compress.
        app.agent.count_current_context = AsyncMock()
        app._turn_active = True
        await _submit_command(app, pilot, "/context")
        await _wait_for_layout(
            pilot,
            lambda: (
                isinstance(app.screen, ContextInspectorScreen)
                and "live turn" in _plain(app.screen.query_one("#context_summary", Static))
            ),
        )
        app.agent.count_current_context.assert_not_awaited()
        await pilot.press("c")
        await pilot.pause()
        # Modal stays open because compress is blocked while busy.
        assert isinstance(app.screen, ContextInspectorScreen)
        await pilot.press("escape")
        await _wait_for_layout(pilot, lambda: not isinstance(app.screen, ContextInspectorScreen))

        # Disconnected state shows friendly empty notice.
        app._turn_active = False
        app.agent = None
        await _submit_command(app, pilot, "/context")
        await _wait_for_layout(
            pilot,
            lambda: (
                isinstance(app.screen, ContextInspectorScreen)
                and "No active agent" in _plain(app.screen.query_one("#context_status", Static))
            ),
        )
        await pilot.press("escape")
        await _wait_for_layout(pilot, lambda: not isinstance(app.screen, ContextInspectorScreen))


@pytest.mark.asyncio
async def test_context_text_inline_and_usage_error(isolated_cli_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    app, _ = _configured_app(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 32)) as pilot:
        await _wait_for_layout(pilot, lambda: app.agent is not None)
        _populate_rich_agent(app)

        await _submit_command(app, pilot, "/context text")
        await _wait_for_layout(
            pilot,
            lambda: any("Token usage by category:" in entry.content for entry in app.conversation_entries),
        )
        assert not isinstance(app.screen, ContextInspectorScreen)

        await _submit_command(app, pilot, "/context bogus")
        await _wait_for_layout(
            pilot,
            lambda: any("Usage: /context [text]" in entry.content for entry in app.conversation_entries),
        )


@pytest.mark.asyncio
async def test_context_modal_responsive_narrow_layout_and_ascii_fallback(
    isolated_cli_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(theme, "supports_unicode", lambda encoding=None: False)
    app, _ = _configured_app(tmp_path, monkeypatch)
    async with app.run_test(size=(72, 30)) as pilot:
        await _wait_for_layout(pilot, lambda: app.agent is not None)
        _populate_rich_agent(app)

        await _submit_command(app, pilot, "/context")
        await _wait_for_layout(
            pilot,
            lambda: (
                isinstance(app.screen, ContextInspectorScreen)
                and "Token usage by category" in _plain(app.screen.query_one("#context_summary", Static))
            ),
        )

        screen = app.screen
        assert isinstance(screen, ContextInspectorScreen)
        assert screen.query_one("#context_hero", Horizontal).has_class("stacked")

        for widget_id in (
            "#context_title",
            "#context_header_hint",
            "#context_grid",
            "#context_summary",
            "#context_details",
            "#context_footer",
        ):
            rendered = _plain(screen.query_one(widget_id, Static))
            assert rendered.isascii(), f"Non-ASCII character in {widget_id}: {rendered!r}"
