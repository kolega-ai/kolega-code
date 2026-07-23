# ruff: noqa: F401,F811,E402
"""TUI wiring for the per-session scratchpad prompt extension."""

from pathlib import Path

import pytest

from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore
from kolega_code.scratchpad import SCRATCHPAD_PROMPT_EXTENSION_ID, scratchpad_dir_for

from ._app_test_utils import (
    FakeCoderAgent,
    build_test_config,
    extension_by_name,
    install_fake_agents,
)


def _build_app(tmp_path: Path):
    from kolega_code.cli.app import KolegaCodeApp

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    return app, project, store, session


@pytest.mark.asyncio
async def test_textual_app_agent_gets_scratchpad_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    install_fake_agents(monkeypatch, planning_cls=FakeCoderAgent)
    app, project, store, session = _build_app(tmp_path)

    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        extension = extension_by_name(app.agent.kwargs["prompt_extensions"], SCRATCHPAD_PROMPT_EXTENSION_ID)
        expected = scratchpad_dir_for(project, session.session_id)

        assert extension.title == "Session Scratchpad"
        assert str(expected) in extension.markdown
        assert extension.propagate_to_sub_agents is True
        assert app.agent.scratchpad_dir == expected
        assert expected.is_dir()


@pytest.mark.asyncio
async def test_textual_app_scratchpad_path_is_stable_across_modes_and_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    install_fake_agents(monkeypatch, planning_cls=FakeCoderAgent)
    app, project, store, session = _build_app(tmp_path)

    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        build_path = app.agent.scratchpad_dir
        assert build_path == scratchpad_dir_for(project, session.session_id)

        # Plan mode rebuilds the agent with the same session id, so the
        # scratchpad path must not change.
        await app.action_toggle_interaction_mode()
        assert app.interaction_mode == "plan"
        assert isinstance(app.agent, FakeCoderAgent)
        plan_extension = extension_by_name(app.agent.kwargs["prompt_extensions"], SCRATCHPAD_PROMPT_EXTENSION_ID)
        assert app.agent.scratchpad_dir == build_path
        assert str(build_path) in plan_extension.markdown
        assert plan_extension.propagate_to_sub_agents is True

        # Switching back to build mode keeps the same path again.
        await app.action_toggle_interaction_mode()
        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.scratchpad_dir == build_path
        extension_by_name(app.agent.kwargs["prompt_extensions"], SCRATCHPAD_PROMPT_EXTENSION_ID)


@pytest.mark.asyncio
async def test_textual_app_resumed_session_keeps_scratchpad_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    install_fake_agents(monkeypatch, planning_cls=FakeCoderAgent)
    app, project, store, session = _build_app(tmp_path)

    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        original_path = app.agent.scratchpad_dir

    # A second app instance bound to the same session record (resume) resolves
    # the identical scratchpad directory.
    from kolega_code.cli.app import KolegaCodeApp

    resumed = KolegaCodeApp(
        project_path=project,
        config=app.config,
        mode="code",
        store=store,
        session=store.load(session.session_id),
    )
    async with resumed.run_test():
        assert isinstance(resumed.agent, FakeCoderAgent)
        assert resumed.agent.scratchpad_dir == original_path
        extension_by_name(resumed.agent.kwargs["prompt_extensions"], SCRATCHPAD_PROMPT_EXTENSION_ID)
