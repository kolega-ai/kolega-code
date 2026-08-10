"""CLI extension loading: import-string resolution, bundle validation, flag
parsing on all three launch parsers, and the full ask-mode lifecycle (inject →
bind → run → cleanup) against a real CoderAgent."""

import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.prompt_provider import AgentMode, PromptExtension
from kolega_code.agent.tools import ToolExtension
from kolega_code.extensions import (
    ExtensionSelection,
    KolegaExtensionBundle,
    KolegaExtensionHost,
    KolegaExtensionLoadError,
    bind_extension_agent,
    cleanup_extension_bundle,
    create_extension_bundle,
    resolve_extension_selection,
)


def _install_module(monkeypatch, name="fake_ext_mod", **attrs):
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _noop_factory(host, config_path):
    return KolegaExtensionBundle()


# --- resolve_extension_selection --------------------------------------------------


@pytest.mark.parametrize("spec", ["nomodule", ":factory", "module:", ""])
def test_resolve_rejects_malformed_specs(spec):
    with pytest.raises(KolegaExtensionLoadError, match="MODULE:FACTORY"):
        resolve_extension_selection(spec, None)


def test_resolve_reports_import_failure():
    with pytest.raises(KolegaExtensionLoadError, match="cannot import module"):
        resolve_extension_selection("kolega_no_such_module:factory", None)


def test_resolve_reports_missing_factory(monkeypatch):
    _install_module(monkeypatch)
    with pytest.raises(KolegaExtensionLoadError, match="has no attribute"):
        resolve_extension_selection("fake_ext_mod:create_extension", None)


def test_resolve_reports_non_callable_factory(monkeypatch):
    _install_module(monkeypatch, create_extension="not callable")
    with pytest.raises(KolegaExtensionLoadError, match="is not callable"):
        resolve_extension_selection("fake_ext_mod:create_extension", None)


def test_resolve_returns_factory_and_absolute_config_path(monkeypatch, tmp_path):
    _install_module(monkeypatch, create_extension=_noop_factory)
    monkeypatch.chdir(tmp_path)
    selection = resolve_extension_selection("fake_ext_mod:create_extension", "rel.json")
    assert selection.factory is _noop_factory
    assert selection.config_path is not None
    assert selection.config_path == tmp_path / "rel.json"
    assert selection.config_path.is_absolute()


def test_resolve_without_config_path(monkeypatch):
    _install_module(monkeypatch, create_extension=_noop_factory)
    selection = resolve_extension_selection("fake_ext_mod:create_extension", None)
    assert selection.config_path is None


# --- create_extension_bundle ------------------------------------------------------


def _host(tmp_path):
    return KolegaExtensionHost(
        project_path=tmp_path,
        workspace_id="ws",
        thread_id="th",
        config=cast(Any, None),  # opaque to validation; the CLI passes the real AgentConfig
        agent_mode=AgentMode.ASK,
    )


def _selection(factory):
    return ExtensionSelection(spec="fake_ext_mod:create_extension", factory=factory, config_path=None)


def test_factory_receives_host_and_config_path(tmp_path):
    received = {}

    def factory(host, config_path):
        received["host"] = host
        received["config_path"] = config_path
        return KolegaExtensionBundle()

    selection = ExtensionSelection(
        spec="fake_ext_mod:create_extension", factory=factory, config_path=tmp_path / "cfg.json"
    )
    create_extension_bundle(selection, _host(tmp_path))
    assert received["host"].workspace_id == "ws"
    assert received["host"].agent_mode is AgentMode.ASK
    assert received["config_path"] == tmp_path / "cfg.json"


def test_factory_exception_is_wrapped(tmp_path):
    def factory(host, config_path):
        raise RuntimeError("boom")

    with pytest.raises(KolegaExtensionLoadError, match="raised RuntimeError: boom"):
        create_extension_bundle(_selection(factory), _host(tmp_path))


def test_wrong_bundle_type_is_rejected(tmp_path):
    with pytest.raises(KolegaExtensionLoadError, match="expected KolegaExtensionBundle"):
        create_extension_bundle(_selection(lambda host, path: {"nope": True}), _host(tmp_path))


def test_wrong_prompt_element_type_is_rejected(tmp_path):
    bundle = KolegaExtensionBundle(prompt_extensions=cast(Any, ["not a prompt extension"]))
    with pytest.raises(KolegaExtensionLoadError, match="non-PromptExtension"):
        create_extension_bundle(_selection(lambda host, path: bundle), _host(tmp_path))


def test_wrong_tool_element_type_is_rejected(tmp_path):
    bundle = KolegaExtensionBundle(tool_extensions=cast(Any, ["not a tool extension"]))
    with pytest.raises(KolegaExtensionLoadError, match="non-ToolExtension"):
        create_extension_bundle(_selection(lambda host, path: bundle), _host(tmp_path))


def test_duplicate_prompt_ids_are_rejected(tmp_path):
    prompt = PromptExtension(id="dup", title="One", markdown="text")
    bundle = KolegaExtensionBundle(prompt_extensions=[prompt, prompt])
    with pytest.raises(KolegaExtensionLoadError, match="duplicate prompt extension id"):
        create_extension_bundle(_selection(lambda host, path: bundle), _host(tmp_path))


def test_bundle_lists_are_copied(tmp_path):
    prompt = PromptExtension(id="p", title="P", markdown="text")
    original = KolegaExtensionBundle(prompt_extensions=[prompt])
    validated = create_extension_bundle(_selection(lambda host, path: original), _host(tmp_path))
    original.prompt_extensions.append(PromptExtension(id="late", title="L", markdown="x"))
    assert [ext.id for ext in validated.prompt_extensions] == ["p"]


# --- bind / cleanup helpers -------------------------------------------------------


@pytest.mark.asyncio
async def test_bind_agent_supports_sync_and_async():
    calls = []
    agent_object = cast(Any, "agent-object")
    sync_bundle = KolegaExtensionBundle(bind_agent=lambda agent: calls.append(("sync", agent)))
    await bind_extension_agent(sync_bundle, agent_object)

    async def async_bind(agent):
        calls.append(("async", agent))

    await bind_extension_agent(KolegaExtensionBundle(bind_agent=async_bind), agent_object)
    assert calls == [("sync", "agent-object"), ("async", "agent-object")]


@pytest.mark.asyncio
async def test_bind_agent_failure_is_wrapped():
    def bad_bind(agent):
        raise RuntimeError("bind boom")

    with pytest.raises(KolegaExtensionLoadError, match="bind_agent failed"):
        await bind_extension_agent(KolegaExtensionBundle(bind_agent=bad_bind), cast(Any, "agent-object"))


@pytest.mark.asyncio
async def test_cleanup_supports_sync_and_async():
    calls = []
    await cleanup_extension_bundle(KolegaExtensionBundle(cleanup=lambda: calls.append("sync")))

    async def async_cleanup():
        calls.append("async")

    await cleanup_extension_bundle(KolegaExtensionBundle(cleanup=async_cleanup))
    await cleanup_extension_bundle(KolegaExtensionBundle())  # no hook: no-op
    assert calls == ["sync", "async"]


# --- CLI flag parsing -------------------------------------------------------------


def test_extension_flags_parse_on_all_three_launch_forms():
    from kolega_code.cli.main import parse_args

    implicit = parse_args(["--extension", "mod:factory", "--extension-config", "/tmp/x.json"])
    assert implicit.extension == "mod:factory"
    assert implicit.extension_config == Path("/tmp/x.json")

    tui = parse_args(["tui", "--extension", "mod:factory"])
    assert tui.extension == "mod:factory"
    assert tui.extension_config is None

    ask = parse_args(["ask", "hello", "--extension", "mod:factory", "--extension-config", "cfg.json"])
    assert ask.extension == "mod:factory"
    assert ask.extension_config == Path("cfg.json")


@pytest.mark.parametrize(
    "argv",
    [
        ["--extension-config", "/tmp/x.json"],
        ["tui", "--extension-config", "/tmp/x.json"],
        ["ask", "hello", "--extension-config", "/tmp/x.json"],
    ],
)
def test_extension_config_without_extension_is_rejected(argv, capsys):
    from kolega_code.cli.main import parse_args

    with pytest.raises(SystemExit) as excinfo:
        parse_args(argv)
    assert excinfo.value.code == 2
    assert "--extension-config requires --extension" in capsys.readouterr().err


def test_other_subcommands_do_not_accept_extension_flags():
    from kolega_code.cli.main import parse_args

    with pytest.raises(SystemExit):
        parse_args(["sessions", "list", "--extension", "mod:factory"])


# --- ask-mode lifecycle (real CoderAgent, no network) -----------------------------


class RecordingCoderAgent(CoderAgent):
    """The real agent, minus the network turn, recording lifecycle order.

    The model-facing tool inventory is snapshotted during the turn: cleanup
    clears extension callbacks, so a post-run ``get_tool_list()`` would no
    longer show extension tools.
    """

    instances: list["RecordingCoderAgent"] = []
    events: list = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.turn_tool_names: set[str] = set()
        RecordingCoderAgent.instances.append(self)

    async def process_message_stream(self, message, attachments=None):
        RecordingCoderAgent.events.append("stream")
        assert self.tool_collection is not None
        self.turn_tool_names = {tool.name for tool in self.tool_collection.get_tool_list()}
        yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}


def _setup_ask(tmp_path, monkeypatch):
    from kolega_code.cli import main as main_module

    RecordingCoderAgent.instances = []
    RecordingCoderAgent.events = []
    monkeypatch.setattr(main_module, "CoderAgent", RecordingCoderAgent)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-opus-5")
    return main_module, project


def _sink(record):
    pass


async def _probe_tool() -> str:
    """A harmless probe."""
    return "ok"


def _lifecycle_factory(monkeypatch, *, tool_name="extension_probe", record_host=None):
    """Install fake_ext_mod:create_extension contributing one prompt + one tool."""

    def create_extension(host, config_path):
        if record_host is not None:
            record_host.append((host, config_path))
        return KolegaExtensionBundle(
            prompt_extensions=[
                PromptExtension(
                    id="test-extension-prompt",
                    title="Test extension",
                    markdown="Extension prompt body.",
                    modes=[AgentMode.CLI, AgentMode.ASK],
                )
            ],
            tool_extensions=[
                ToolExtension(
                    name="test-extension",
                    tools={tool_name: _probe_tool},
                    # Explicit description + bare JSON input schema, per the
                    # ToolExtension contract.
                    tool_descriptions={tool_name: "A harmless probe."},
                    tool_schemas={tool_name: {"type": "object", "properties": {}}},
                    propagate_to_sub_agents=False,
                )
            ],
            llm_trace_sink=_sink,
            bind_agent=lambda agent: RecordingCoderAgent.events.append(("bind", agent)),
            cleanup=lambda: RecordingCoderAgent.events.append("cleanup"),
        )

    _install_module(monkeypatch, create_extension=create_extension)
    return create_extension


def test_ask_injects_binds_and_cleans_up(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup_ask(tmp_path, monkeypatch)
    hosts = []
    _lifecycle_factory(monkeypatch, record_host=hosts)

    exit_code = main_module.main(
        ["ask", "do the thing", "--project", str(project), "--extension", "fake_ext_mod:create_extension"]
    )

    assert exit_code == 0
    agent = RecordingCoderAgent.instances[0]

    # Factory saw the resolved ASK host, no config path.
    ((host, config_path),) = hosts
    assert host.agent_mode is AgentMode.ASK
    assert host.project_path == agent.project_path
    assert host.workspace_id == agent.workspace_id
    assert host.thread_id == agent.thread_id
    assert host.config is agent.config
    assert config_path is None

    # Prompt and tool are installed through the normal inventories.
    assert any(ext.id == "test-extension-prompt" for ext in agent.prompt_extensions)
    assert "extension_probe" in agent.turn_tool_names

    # Trace sink landed on the agent (and therefore its context telemetry).
    assert agent.llm_trace_sink is _sink

    # bind ran exactly once, on the constructed agent, before the model turn;
    # cleanup ran exactly once, after it.
    bind_events = [event for event in RecordingCoderAgent.events if isinstance(event, tuple) and event[0] == "bind"]
    assert len(bind_events) == 1
    assert bind_events[0][1] is agent
    assert RecordingCoderAgent.events.index(bind_events[0]) < RecordingCoderAgent.events.index("stream")
    assert RecordingCoderAgent.events.count("cleanup") == 1
    assert RecordingCoderAgent.events.index("cleanup") > RecordingCoderAgent.events.index("stream")


def test_ask_without_extension_flag_is_unchanged(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup_ask(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project)])

    assert exit_code == 0
    agent = RecordingCoderAgent.instances[0]
    assert agent.llm_trace_sink is None
    assert not any(getattr(ext, "id", "") == "test-extension-prompt" for ext in agent.prompt_extensions)


def test_ask_unimportable_extension_fails_before_agent_construction(tmp_path, monkeypatch, isolated_cli_env, capsys):
    main_module, project = _setup_ask(tmp_path, monkeypatch)

    exit_code = main_module.main(
        ["ask", "do the thing", "--project", str(project), "--extension", "kolega_no_such_module:factory"]
    )

    assert exit_code == 2
    assert "cannot import module" in capsys.readouterr().err
    assert RecordingCoderAgent.instances == []


def test_ask_tool_conflict_refuses_to_start(tmp_path, monkeypatch, isolated_cli_env, capsys):
    """An extension tool clashing with a built-in provisioned tool must abort the run."""
    main_module, project = _setup_ask(tmp_path, monkeypatch)
    _lifecycle_factory(monkeypatch, tool_name="read")

    exit_code = main_module.main(
        ["ask", "do the thing", "--project", str(project), "--extension", "fake_ext_mod:create_extension"]
    )

    assert exit_code == 2
    assert "conflicts" in capsys.readouterr().err
    assert RecordingCoderAgent.events.count("stream") == 0


def test_ask_factory_exception_fails_with_concise_error(tmp_path, monkeypatch, isolated_cli_env, capsys):
    main_module, project = _setup_ask(tmp_path, monkeypatch)

    def create_extension(host, config_path):
        raise RuntimeError("manifest invalid")

    _install_module(monkeypatch, create_extension=create_extension)

    exit_code = main_module.main(
        ["ask", "do the thing", "--project", str(project), "--extension", "fake_ext_mod:create_extension"]
    )

    assert exit_code == 2
    assert "manifest invalid" in capsys.readouterr().err
    assert RecordingCoderAgent.instances == []
