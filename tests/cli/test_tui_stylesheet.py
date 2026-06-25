# ruff: noqa: F401,F811,E402
from pathlib import Path

import pytest

from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.tui import agent_runtime as agent_runtime_module
from kolega_code.config import ModelProvider


EXPECTED_CSS_PATH = "tui/styles.tcss"


from ._app_test_utils import build_test_config


def test_tui_uses_external_textual_stylesheet(tmp_path: Path) -> None:
    from kolega_code.cli.app import KolegaCodeApp

    stylesheet = Path(__file__).parents[2] / "kolega_code" / "cli" / EXPECTED_CSS_PATH

    assert KolegaCodeApp.CSS_PATH == EXPECTED_CSS_PATH
    assert KolegaCodeApp.CSS == ""
    assert not KolegaCodeApp.__dict__.get("CSS")
    assert stylesheet.is_file()
    assert "Screen {" in stylesheet.read_text()

    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", {})
    app = KolegaCodeApp(project_path=project, config=None, mode="code", store=store, session=session)

    assert app.css_path == [stylesheet.resolve()]


@pytest.mark.asyncio
async def test_tui_external_stylesheet_loads_in_textual_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            pass

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.kwargs["agent_mode"] == AgentMode.CLI
        assert app.query_one("#conversation") is not None
        assert app.query_one("#composer") is not None


def test_tui_stylesheet_themes_scrollbars_and_avoids_disabled_opacity() -> None:
    stylesheet = Path(__file__).parents[2] / "kolega_code" / "cli" / EXPECTED_CSS_PATH
    css = stylesheet.read_text()

    for property_name in (
        "scrollbar-background:",
        "scrollbar-background-hover:",
        "scrollbar-background-active:",
        "scrollbar-color:",
        "scrollbar-color-hover:",
        "scrollbar-color-active:",
        "scrollbar-corner-color:",
    ):
        assert property_name in css

    composer_block = css.split("#composer {", 1)[1].split("}", 1)[0]
    assert "border: none" in composer_block
    assert "border-top: solid $surface-lighten-2" in composer_block
    assert "background: $surface" in composer_block
    assert "color: $text" in composer_block
    assert "border: round" not in composer_block
    assert "margin-top" not in composer_block
    assert "background: $panel" not in composer_block

    queued_messages_block = css.split("#queued_messages", 1)[1].split("}", 1)[0]
    assert "background: $surface" in queued_messages_block
    assert "text-style: italic" not in queued_messages_block
    assert "color: $warning" in queued_messages_block
    assert "border-top: solid $surface-lighten-2" in queued_messages_block
    assert "border-top: solid $warning" not in queued_messages_block
    assert "margin-top" not in queued_messages_block

    disabled_block = css.split("#composer:disabled", 1)[1].split("}", 1)[0]

    def property_value(block: str, property_name: str) -> str:
        return block.split(f"{property_name}:", 1)[1].split(";", 1)[0].strip()

    for property_name in ("background", "color", "border", "border-top"):
        assert property_value(disabled_block, property_name) == property_value(composer_block, property_name)
    assert property_value(disabled_block, "background") == "$surface"
    assert property_value(disabled_block, "color") == "$text"
    assert property_value(disabled_block, "opacity") == "1"
    assert property_value(disabled_block, "background-tint") == "transparent"
    assert "color: $text-muted" not in disabled_block

    # Leave the normal blinking cursor on Textual's default styling (main-branch
    # behavior: bright/white in dark terminals). Only pin the disabled cursor so
    # disabled-state composer colors remain stable.
    assert "#composer .text-area--cursor,\n#composer:disabled .text-area--cursor" not in css
    disabled_cursor_selector = "#composer:disabled .text-area--cursor"
    assert disabled_cursor_selector in css
    cursor_block = css.split(disabled_cursor_selector, 1)[1].split("}", 1)[0]
    assert property_value(cursor_block, "background") == "$surface-lighten-2"
    assert property_value(cursor_block, "color") == "$text-muted"

    for selector in (
        "#composer .text-area--selection,\n#composer:disabled .text-area--selection",
        "#composer .text-area--cursor-line,\n#composer:disabled .text-area--cursor-line",
        "#composer .text-area--gutter,\n#composer:disabled .text-area--gutter",
        "#composer .text-area--cursor-gutter,\n#composer:disabled .text-area--cursor-gutter",
        "#composer .text-area--placeholder,\n#composer:disabled .text-area--placeholder",
    ):
        assert selector in css

    prompt_panel_block = css.split("#question_prompt, #approval_prompt", 1)[1].split("}", 1)[0]
    assert "border: none" in prompt_panel_block
    assert "padding: 0" in prompt_panel_block
    assert "margin: 0" in prompt_panel_block
    assert "border: round" not in prompt_panel_block

    prompt_header_scroll_block = css.split(
        "#question_prompt > .prompt-header-scroll, #approval_prompt > .prompt-header-scroll", 1
    )[1].split("}", 1)[0]
    assert "margin-bottom: 1" in prompt_header_scroll_block

    prompt_actions_block = css.split("#question_prompt > ActionList, #approval_prompt > ActionList", 1)[1].split(
        "}", 1
    )[0]
    assert "border: none" in prompt_actions_block
    assert "padding: 0" in prompt_actions_block
    assert "background-tint: transparent" in prompt_actions_block

    prompt_header_block = css.split(".prompt-header {", 1)[1].split("}", 1)[0]
    assert "padding: 0 1 1 1" in prompt_header_block

    prompt_option_rows_block = css.split("#question_prompt > ActionList > .option-list--option,", 1)[1].split("}", 1)[0]
    assert "#approval_prompt > ActionList > .option-list--option" in prompt_option_rows_block
    assert "padding: 0 1" in prompt_option_rows_block

    plan_actions_block = css.split("#plan_actions,", 1)[1].split("}", 1)[0]
    assert "border: none" in plan_actions_block
    assert "padding: 0" in plan_actions_block
    assert "margin: 0" in plan_actions_block
    assert "background-tint: transparent" in plan_actions_block

    plan_option_rows_block = css.split("#plan_actions > .option-list--option", 1)[1].split("}", 1)[0]
    assert "padding: 0 1" in plan_option_rows_block

    model_actions_block = css.split("#model_actions, #effort_actions, #theme_actions", 1)[1].split("}", 1)[0]
    assert "border: round $surface" in model_actions_block
    assert "#plan_actions" not in model_actions_block


def test_tui_stylesheet_explicitly_themes_footer_select_overlay_and_output_scrollbars() -> None:
    stylesheet = Path(__file__).parents[2] / "kolega_code" / "cli" / EXPECTED_CSS_PATH
    css = stylesheet.read_text()

    conversation_block = css.split("#conversation {", 1)[1].split("}", 1)[0]
    assert "height: 1fr" in conversation_block
    assert "border: none" in conversation_block
    assert "background: $surface" in conversation_block
    assert "color: $text" in conversation_block
    assert "overflow-x: hidden" in conversation_block

    sidebar_output_block = css.split("#logs, #terminal", 1)[1].split("}", 1)[0]
    assert "height: 1fr" in sidebar_output_block
    assert "border: round $surface" in sidebar_output_block
    assert "background: $surface" in sidebar_output_block
    assert "overflow-x: hidden" in sidebar_output_block

    footer_block = css.split("Footer {", 1)[1].split("}", 1)[0]
    assert "background: $surface" in footer_block
    assert "color: $text-muted" in footer_block

    select_overlay_block = css.split("Select > SelectOverlay", 1)[1].split("}", 1)[0]
    assert "background: $surface" in select_overlay_block
    assert "color: $text" in select_overlay_block
