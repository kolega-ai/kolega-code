"""Settings-tab LSP status reflects the runtime agent, not a stale startup snapshot.

Regression coverage for the bug where ``#lsp_status`` permanently read
"LSP is not active" because it was computed once at startup (before the agent
existed) and never refreshed after ``_build_agent`` constructed the agent and
its ``lsp_manager``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ._app_test_utils import FakeCoderAgent, build_test_config, install_fake_agents


def _static_text(widget) -> str:
    """Extract the plain text currently shown by a Textual ``Static`` widget."""
    # Textual 8.x stores the last update() argument in a name-mangled attribute.
    content = getattr(widget, "_Static__content", None)
    if isinstance(content, str):
        return content
    visual = getattr(widget, "_Static__visual", None)
    if visual is not None:
        return str(visual)
    return str(getattr(widget, "renderable", "") or "")


def _make_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, coder_cls=FakeCoderAgent):
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.config import config_summary
    from kolega_code.cli.session_store import SessionStore

    install_fake_agents(monkeypatch, coder_cls=coder_cls)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


class _LspEnabledFakeAgent(FakeCoderAgent):
    """Fake agent whose tool collection carries an enabled lsp_manager with a report."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Typed ``Any`` so the assignment is valid against the fake collection's
        # ``lsp_manager`` attribute (which defaults to ``None``).
        manager: Any = SimpleNamespace(
            enabled=True,
            report=SimpleNamespace(
                detected=[SimpleNamespace(display_name="Python", language_id="python")],
                resolved=[SimpleNamespace(language_id="python", server_name="pylsp")],
                missing=[],
            ),
            _sessions={},
            _resolved={},
            _missing={},
            last_diagnostic_count={},
        )
        self.tool_collection.lsp_manager = manager


@pytest.mark.asyncio
async def test_lsp_status_shows_detected_languages_after_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After the agent is built, the status reflects the detected language — not 'not active'."""
    from textual.widgets import Static

    app = _make_app(tmp_path, monkeypatch, coder_cls=_LspEnabledFakeAgent)
    async with app.run_test():
        status = app.query_one("#lsp_status", Static)
        text = _static_text(status)
        assert "not active" not in text
        assert "Python" in text


@pytest.mark.asyncio
async def test_lsp_status_shows_not_active_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When no lsp_manager is present, the status correctly reads 'not active'."""
    from textual.widgets import Static

    # Default FakeCoderAgent -> tool_collection.lsp_manager is None.
    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        status = app.query_one("#lsp_status", Static)
        text = _static_text(status)
        assert "not active" in text


@pytest.mark.asyncio
async def test_build_agent_refreshes_lsp_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_build_agent`` must refresh the LSP status after the agent exists.

    The pre-build ``_populate_settings_controls`` call sees ``agent is None``;
    the fix adds a refresh at the end of ``_build_agent`` where the agent (and
    its lsp_manager) is available.
    """
    from kolega_code.cli.app import KolegaCodeApp

    agent_at_call: list[object] = []

    def spy(self) -> None:
        agent_at_call.append(self.agent)

    monkeypatch.setattr(KolegaCodeApp, "_update_lsp_settings_status", spy)

    app = _make_app(tmp_path, monkeypatch, coder_cls=_LspEnabledFakeAgent)
    async with app.run_test():
        # At least one refresh happened with a real agent in place.
        assert any(agent is not None for agent in agent_at_call)
