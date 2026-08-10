"""Interactive-mode extension lifecycle: every top-level agent generation gets a
fresh bundle from the launch-selected factory, injected and bound before use,
with the prior generation's bundle cleaned up exactly once on rebuild and the
last one on app exit."""

# ruff: noqa: F401,F811,E402
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


def _sink(record):
    pass


def _install_extension(monkeypatch, recorder):
    def create_extension(host, config_path):
        recorder.factory_calls.append((host, config_path))
        return KolegaExtensionBundle(
            prompt_extensions=[PromptExtension(id="app-test-extension-prompt", title="App test", markdown="body")],
            tool_extensions=[ToolExtension(name="app-test-extension", tools={"extension_probe": lambda: "ok"})],
            llm_trace_sink=_sink,
            bind_agent=recorder.binds.append,
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
