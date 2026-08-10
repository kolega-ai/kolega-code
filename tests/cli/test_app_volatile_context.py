"""The CLI registers the plan handle and shared task list as volatile-context sections.

Agent-level injection behavior is covered by ``tests/agent/test_volatile_sections.py``.
These tests verify the CLI wiring: ``_build_agent`` registers exactly the plan and
task-list providers on the top-level agent, and the providers render the session state
(artifact path / task-list markdown) — or empty text when unset.
"""

from typing import Any, cast

import pytest

from tests.cli._app_test_utils import build_test_config, install_fake_agents

from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore


def _make_app(tmp_path, monkeypatch):
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


@pytest.mark.asyncio
async def test_build_agent_registers_plan_and_task_list_providers(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    app = _make_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.agent is not None
        fake_agent = cast(Any, app.agent)
        providers = fake_agent.volatile_section_providers
        assert len(providers) == 2
        # Bound-method equality: same function + same instance.
        assert providers[0] == app._plan_volatile_section
        assert providers[1] == app._task_list_volatile_section


@pytest.mark.asyncio
async def test_plan_section_carries_artifact_path_and_persists_artifact(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    app = _make_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await pilot.pause()
        app._latest_plan = "# Build the feature\n\n- [ ] step one"

        section = app._plan_volatile_section()

        assert section.key == "plan"
        assert "current-plan.md" in section.text
        assert app.session.session_id in section.text
        artifact = app.store.root / "plans" / app.session.session_id / "current-plan.md"
        assert artifact.exists()
        assert "Build the feature" in artifact.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_task_list_section_mirrors_session(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    app = _make_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await pilot.pause()
        app.session.task_list_markdown = "- [x] one\n- [ ] two"

        section = app._task_list_volatile_section()

        assert section.key == "task_list"
        assert section.text == "- [x] one\n- [ ] two"


@pytest.mark.asyncio
async def test_sections_empty_when_unset(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    app = _make_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await pilot.pause()

        assert app._plan_volatile_section().text == ""
        assert app._task_list_volatile_section().text == ""
