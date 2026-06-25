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

    disabled_block = css.split("#composer:disabled", 1)[1].split("}", 1)[0]
    assert "opacity" not in disabled_block
    assert "background: $surface" in disabled_block
    assert "color: $text-muted" in disabled_block


def test_tui_stylesheet_explicitly_themes_footer_select_overlay_and_output_scrollbars() -> None:
    stylesheet = Path(__file__).parents[2] / "kolega_code" / "cli" / EXPECTED_CSS_PATH
    css = stylesheet.read_text()

    output_block = css.split("#conversation, #logs, #terminal", 1)[1].split("}", 1)[0]
    assert "overflow-x: hidden" in output_block
    assert "background: $surface" in output_block

    footer_block = css.split("Footer {", 1)[1].split("}", 1)[0]
    assert "background: $surface" in footer_block
    assert "color: $text-muted" in footer_block

    select_overlay_block = css.split("Select > SelectOverlay", 1)[1].split("}", 1)[0]
    assert "background: $surface" in select_overlay_block
    assert "color: $text" in select_overlay_block
