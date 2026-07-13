from pathlib import Path

import pytest

from kolega_code.cli import main as main_module
from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore

from ._app_test_utils import FakeCoderAgent, build_test_config, install_fake_agents


def _write_agent(project: Path, content: str) -> None:
    path = project / ".kolega" / "agents" / "reviewer.md"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")


def test_parse_agents_subcommands() -> None:
    list_args = main_module.parse_args(["agents", "list", "--project", ".", "--state-dir", "/tmp/state"])
    validate_args = main_module.parse_args(["agents", "validate", "--project", "."])

    assert list_args.command == "agents"
    assert list_args.agents_command == "list"
    assert list_args.state_dir == Path("/tmp/state")
    assert validate_args.agents_command == "validate"


def test_agents_list_reports_effective_definitions(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write_agent(
        project,
        "---\nname: reviewer\ndescription: Reviews code\ntools: [read_entire_file]\n---\nReview carefully.\n",
    )

    result = main_module.main(["agents", "list", "--project", str(project), "--state-dir", str(tmp_path / "state")])

    output = capsys.readouterr().out
    assert result == 0
    assert "`reviewer` (project, build): Reviews code" in output
    assert "read_entire_file" in output


def test_agents_validate_returns_one_for_invalid_definition(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write_agent(project, "---\nname: reviewer\ndescription: Reviews code\ntyop: true\n---\nReview carefully.\n")

    result = main_module.main(["agents", "validate", "--project", str(project), "--state-dir", str(tmp_path / "state")])

    output = capsys.readouterr().out
    assert result == 1
    assert "unknown frontmatter field(s): tyop" in output


@pytest.mark.asyncio
async def test_tui_filters_custom_agents_when_switching_build_and_plan_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")
    from kolega_code.cli.app import KolegaCodeApp

    class FakePlanningAgent(FakeCoderAgent):
        pass

    install_fake_agents(monkeypatch, planning_cls=FakePlanningAgent)
    project = tmp_path / "project"
    project.mkdir()
    agent_dir = project / ".kolega" / "agents"
    agent_dir.mkdir(parents=True)
    (agent_dir / "builder.md").write_text(
        "---\nname: builder\ndescription: Build specialist\n---\nBuild.\n",
        encoding="utf-8",
    )
    (agent_dir / "planner.md").write_text(
        "---\nname: planner\ndescription: Plan specialist\nmode: plan\n---\nPlan.\n",
        encoding="utf-8",
    )
    (agent_dir / "shared.md").write_text(
        "---\nname: shared\ndescription: Shared specialist\nmode: all\n---\nWork.\n",
        encoding="utf-8",
    )
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.kwargs["custom_agent_catalog"].names() == ["builder", "shared"]
        await pilot.press("shift+tab")
        assert isinstance(app.agent, FakePlanningAgent)
        assert app.agent.kwargs["custom_agent_catalog"].names() == ["planner", "shared"]
