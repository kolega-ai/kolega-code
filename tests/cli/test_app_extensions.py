"""Interactive-mode extension lifecycle: every top-level agent generation gets a
fresh bundle from the launch-selected factory, injected and bound before use,
with the prior generation's bundle cleaned up exactly once on rebuild and the
last one on app exit."""

# ruff: noqa: F401,F811,E402
import asyncio
import sys
import types

import pytest

from kolega_code.agent.prompt_provider import AgentMode, PromptExtension
from kolega_code.agent.tools import ToolExtension
from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore
from kolega_code.extensions import KolegaExtensionBundle, resolve_extension_selection

from ._app_test_utils import (
    FakeCoderAgent,
    build_test_config,
    install_fake_agents,
)


class FakePlanningAgent(FakeCoderAgent):
    pass


class _Recorder:
    def __init__(self):
        self.factory_calls: list = []
        self.binds: list = []
        self.cleanups: int = 0
        self.fail_bind: bool = False


def _sink(record):
    pass


def _install_extension(monkeypatch, recorder):
    def _bind(agent):
        if recorder.fail_bind:
            raise RuntimeError("bind boom")
        recorder.binds.append(agent)

    def create_extension(host, config_path):
        recorder.factory_calls.append((host, config_path))
        return KolegaExtensionBundle(
            prompt_extensions=[PromptExtension(id="app-test-extension-prompt", title="App test", markdown="body")],
            tool_extensions=[
                ToolExtension(
                    name="app-test-extension",
                    tools={"extension_probe": lambda: "ok"},
                    tool_descriptions={"extension_probe": "A harmless probe."},
                    tool_schemas={"extension_probe": {"type": "object", "properties": {}, "required": []}},
                )
            ],
            llm_trace_sink=_sink,
            bind_agent=_bind,
            cleanup=lambda: setattr(recorder, "cleanups", recorder.cleanups + 1),
        )

    module = types.ModuleType("fake_app_ext_mod")
    setattr(module, "create_extension", create_extension)
    monkeypatch.setitem(sys.modules, "fake_app_ext_mod", module)
    return resolve_extension_selection("fake_app_ext_mod:create_extension", None)


def _build_app(tmp_path, monkeypatch, selection):
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, planning_cls=FakePlanningAgent)
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(
        project_path=project,
        config=config,
        mode="code",
        store=store,
        session=session,
        extension_selection=selection,
    )
    return app, config


@pytest.mark.asyncio
async def test_extension_bundle_injected_and_bound_per_generation(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    recorder = _Recorder()
    selection = _install_extension(monkeypatch, recorder)
    app, config = _build_app(tmp_path, monkeypatch, selection)

    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        first_agent = app.agent

        # Generation 1: factory called once with the CLI host, injected, bound.
        assert len(recorder.factory_calls) == 1
        host, config_path = recorder.factory_calls[0]
        assert host.agent_mode is AgentMode.CLI
        assert host.config is config
        assert config_path is None
        assert any(
            getattr(ext, "id", "") == "app-test-extension-prompt" for ext in first_agent.kwargs["prompt_extensions"]
        )
        assert any(getattr(ext, "name", "") == "app-test-extension" for ext in first_agent.kwargs["tool_extensions"])
        assert first_agent.kwargs["llm_trace_sink"] is _sink
        assert recorder.binds == [first_agent]
        assert recorder.cleanups == 0

        # Generation 2: a rebuild cleans up the prior bundle and repeats the
        # full inject/bind cycle with a fresh bundle.
        await app._build_agent(config, rebuild=True)
        second_agent = app.agent
        assert second_agent is not first_agent
        assert len(recorder.factory_calls) == 2
        assert recorder.cleanups == 1
        assert recorder.binds == [first_agent, second_agent]
        assert second_agent.kwargs["llm_trace_sink"] is _sink

        # App-exit path releases the final bundle exactly once, idempotently.
        await app._cleanup_extension_bundle()
        assert recorder.cleanups == 2
        await app._cleanup_extension_bundle()
        assert recorder.cleanups == 2


@pytest.mark.asyncio
async def test_cleanup_agent_generation_is_idempotent_and_detaches_agent(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    recorder = _Recorder()
    selection = _install_extension(monkeypatch, recorder)
    app, _config = _build_app(tmp_path, monkeypatch, selection)

    async with app.run_test():
        agent = app.agent
        assert isinstance(agent, FakeCoderAgent)
        await app._cleanup_agent_generation()
        assert app.agent is None
        assert app._extension_bundle is None
        assert agent.cleanup_calls == 1
        assert recorder.cleanups == 1
        # Repeated calls find nothing to clean.
        await app._cleanup_agent_generation()
        assert agent.cleanup_calls == 1
        assert recorder.cleanups == 1


@pytest.mark.asyncio
async def test_concurrent_generation_cleanup_preserves_agent_before_bundle_order(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    recorder = _Recorder()
    selection = _install_extension(monkeypatch, recorder)
    app, _config = _build_app(tmp_path, monkeypatch, selection)

    async with app.run_test():
        agent = app.agent
        bundle = app._extension_bundle
        assert isinstance(agent, FakeCoderAgent)
        assert bundle is not None

        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        events: list[str] = []

        async def cleanup_agent() -> None:
            events.append("agent-start")
            cleanup_started.set()
            await release_cleanup.wait()
            events.append("agent-end")

        async def cleanup_bundle() -> None:
            events.append("bundle")

        agent.cleanup = cleanup_agent
        bundle.cleanup = cleanup_bundle

        first_cleanup = asyncio.create_task(app._cleanup_agent_generation())
        await cleanup_started.wait()

        second_cleanup = asyncio.create_task(app._cleanup_agent_generation())
        await second_cleanup

        assert app.agent is None
        assert app._extension_bundle is None
        assert events == ["agent-start"]

        release_cleanup.set()
        await first_cleanup

        assert events == ["agent-start", "agent-end", "bundle"]


@pytest.mark.asyncio
async def test_cancelled_generation_cleanup_still_releases_claimed_bundle(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    recorder = _Recorder()
    selection = _install_extension(monkeypatch, recorder)
    app, _config = _build_app(tmp_path, monkeypatch, selection)

    async with app.run_test():
        agent = app.agent
        bundle = app._extension_bundle
        assert isinstance(agent, FakeCoderAgent)
        assert bundle is not None

        cleanup_started = asyncio.Event()
        events: list[str] = []

        async def cleanup_agent() -> None:
            events.append("agent-start")
            cleanup_started.set()
            await asyncio.Event().wait()

        async def cleanup_bundle() -> None:
            events.append("bundle")

        agent.cleanup = cleanup_agent
        bundle.cleanup = cleanup_bundle

        cleanup_task = asyncio.create_task(app._cleanup_agent_generation())
        await cleanup_started.wait()
        cleanup_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await cleanup_task

        assert app.agent is None
        assert app._extension_bundle is None
        assert events == ["agent-start", "bundle"]


@pytest.mark.asyncio
async def test_failed_rebuild_reclaims_partial_generation_exactly_once(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    from kolega_code.extensions import KolegaExtensionLoadError

    recorder = _Recorder()
    selection = _install_extension(monkeypatch, recorder)
    app, config = _build_app(tmp_path, monkeypatch, selection)

    async with app.run_test():
        first_agent = app.agent
        assert isinstance(first_agent, FakeCoderAgent)

        # Make the second generation fail at bind time, after its bundle and
        # agent already exist.
        recorder.fail_bind = True

        with pytest.raises(KolegaExtensionLoadError, match="bind_agent failed"):
            await app._build_agent(config, rebuild=True)

        # Old generation cleaned on rebuild entry, failed new generation
        # reclaimed by the transaction guard: two bundle cleanups total, the
        # partially built agent cleaned, and no live generation installed.
        assert recorder.cleanups == 2
        assert first_agent.cleanup_calls == 1
        assert app.agent is None
        assert app._extension_bundle is None


@pytest.mark.asyncio
async def test_action_quit_cleans_generation_even_when_save_fails(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    from unittest.mock import AsyncMock, MagicMock

    recorder = _Recorder()
    selection = _install_extension(monkeypatch, recorder)
    app, _config = _build_app(tmp_path, monkeypatch, selection)

    async with app.run_test():
        agent = app.agent
        assert isinstance(agent, FakeCoderAgent)
        app._save_session_history_async = AsyncMock(side_effect=RuntimeError("session save failed"))
        app.exit = MagicMock()

        with pytest.raises(RuntimeError, match="session save failed"):
            await app.action_quit()

        assert agent.cleanup_calls == 1
        assert recorder.cleanups == 1
        assert app.agent is None
        assert app._extension_bundle is None
        # The session was not saved, so the post-quit resume hint must not fire.
        assert app._quit_cleanly is False
        app.exit.assert_called_once_with()


@pytest.mark.asyncio
async def test_app_without_extension_selection_is_unchanged(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    app, _ = _build_app(tmp_path, monkeypatch, None)

    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.kwargs["llm_trace_sink"] is None
        assert not any(
            getattr(ext, "id", "") == "app-test-extension-prompt" for ext in app.agent.kwargs["prompt_extensions"]
        )
        assert app._extension_bundle is None
