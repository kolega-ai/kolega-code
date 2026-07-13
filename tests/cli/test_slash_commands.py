from pathlib import Path

from kolega_code.cli.skills import SkillCatalog, SkillRecord
from kolega_code.cli.slash_commands import (
    SKILLS_LIST_COMMAND,
    THREAD_RESET_COMMANDS,
    TUI_COMMAND_NAMES,
    CommandScope,
    agent_command_names,
    all_command_entries,
    search_commands,
)


def _catalog(*names: str) -> SkillCatalog:
    skills = {
        name: SkillRecord(
            name=name,
            description=f"Description of {name}",
            skill_dir=Path("/tmp") / name,
            skill_file=Path("/tmp") / name / "SKILL.md",
            scope="project",
        )
        for name in names
    }
    return SkillCatalog(skills=skills)


def test_agent_command_names_match_command_processor():
    from kolega_code.agent.utils.commands import CommandProcessor

    assert agent_command_names() == {spec.name for spec in CommandProcessor.SPECS}
    assert THREAD_RESET_COMMANDS <= agent_command_names()
    assert SKILLS_LIST_COMMAND in TUI_COMMAND_NAMES
    assert "/agents" in TUI_COMMAND_NAMES
    assert "/init" in TUI_COMMAND_NAMES
    assert "/sidebar" in TUI_COMMAND_NAMES
    assert "/prompts" in TUI_COMMAND_NAMES
    assert "/queue-clear" in TUI_COMMAND_NAMES
    assert "/queue" not in TUI_COMMAND_NAMES
    assert "/exit" in TUI_COMMAND_NAMES


def test_all_command_entries_unique_with_descriptions():
    entries = all_command_entries(_catalog("demo-skill"))
    names = [entry.name for entry in entries]
    assert len(names) == len(set(names))
    assert all(entry.description for entry in entries)
    assert all(entry.token == f"/{entry.name}" for entry in entries)


def test_all_command_entries_include_each_scope():
    entries = all_command_entries(_catalog("demo-skill"))
    by_name = {entry.name: entry for entry in entries}
    assert by_name["help"].scope is CommandScope.AGENT
    assert by_name["agents"].scope is CommandScope.TUI
    assert by_name["init"].scope is CommandScope.TUI
    assert by_name["plan"].scope is CommandScope.TUI
    assert by_name["effort"].scope is CommandScope.TUI
    assert by_name["sidebar"].scope is CommandScope.TUI
    assert by_name["prompts"].scope is CommandScope.TUI
    assert by_name["queue-clear"].scope is CommandScope.TUI
    assert "queue" not in by_name
    assert by_name["exit"].scope is CommandScope.TUI
    assert by_name["demo-skill"].scope is CommandScope.SKILL


def test_builtin_command_shadows_skill_with_same_name():
    entries = all_command_entries(_catalog("agents", "init"))
    for name in ("agents", "init"):
        matches = [entry for entry in entries if entry.name == name]
        assert len(matches) == 1
        assert matches[0].scope is CommandScope.TUI


def test_search_commands_prefix_matches_first():
    results = search_commands("c", _catalog())
    names = [entry.name for entry in results]
    prefix = [name for name in names if name.startswith("c")]
    assert prefix == names[: len(prefix)]
    assert "clear" in prefix and "compress" in prefix and "context" in prefix and "copy" in prefix


def test_search_commands_substring_after_prefix():
    results = search_commands("ui", _catalog())
    names = [entry.name for entry in results]
    assert "quit" in names


def test_search_commands_empty_query_lists_all_up_to_limit():
    catalog = _catalog("a-skill", "b-skill")
    assert len(search_commands("", catalog, limit=5)) == 5
    everything = search_commands("", catalog, limit=100)
    assert {entry.name for entry in everything} >= {"help", "plan", "a-skill", "b-skill"}


def test_search_commands_includes_skills_dynamically():
    results = search_commands("demo", _catalog("demo-skill"))
    assert [entry.name for entry in results] == ["demo-skill"]
    assert results[0].description == "Description of demo-skill"


def test_search_commands_no_match():
    assert search_commands("zzz", _catalog()) == []
