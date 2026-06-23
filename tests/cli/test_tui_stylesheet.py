from pathlib import Path

import pytest

from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.tui import agent_runtime as agent_runtime_module
from kolega_code.config import ModelProvider


EXPECTED_CSS_PATH = "tui/styles.tcss"


def build_test_config(project: Path):
    return build_agent_config(
        project,
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_PROVIDER": ModelProvider.ANTHROPIC.value,
        },
    )


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
