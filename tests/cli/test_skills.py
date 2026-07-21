from pathlib import Path

import pytest

from kolega_code.cli.skills import (
    MAX_SKILL_METADATA_CHAR_BUDGET,
    SkillCatalogBudget,
    activated_skill_names,
    build_skill_prompt_extension,
    build_skill_tool_extension,
    discover_skills,
)
from kolega_code.llm.models import Message, TextBlock, ToolResult


def write_skill(
    root: Path, name: str, description: str = "Use this skill for testing.", body: str = "Do the work."
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def test_discover_skills_loads_user_and_project_spec_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    project.mkdir()

    user_skills = user_home / ".agents" / "skills"
    project_skills = project / ".agents" / "skills"
    write_skill(user_skills, "user-skill", "Use this user skill.")
    write_skill(project_skills, "project-skill", "Use this project skill.")

    catalog = discover_skills(project, user_home=user_home, include_builtin=False)

    assert list(catalog.skills) == ["project-skill", "user-skill"]
    assert catalog.skills["project-skill"].scope == "project"
    assert catalog.skills["user-skill"].scope == "user"


def test_discover_skills_does_not_scan_singular_project_agent_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    project.mkdir()
    write_skill(project / ".agent" / "skills", "ignored-skill", "Use this ignored skill.")

    catalog = discover_skills(project, user_home=user_home, include_builtin=False)

    assert "ignored-skill" not in catalog.skills


def test_project_skill_overrides_user_skill_with_same_name(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    project.mkdir()
    write_skill(user_home / ".agents" / "skills", "shared-skill", "Use the user skill.", body="user body")
    write_skill(project / ".agents" / "skills", "shared-skill", "Use the project skill.", body="project body")

    catalog = discover_skills(project, user_home=user_home, include_builtin=False)

    assert catalog.skills["shared-skill"].scope == "project"
    assert "project body" in catalog.activation_content("shared-skill")
    assert any("overrides user skill" in diagnostic.message for diagnostic in catalog.diagnostics)


def test_discover_skills_supports_folded_yaml_description(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    skills_root = project / ".agents" / "skills"
    skill_dir = skills_root / "folded-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: folded-skill
description: >
  Use this folded skill
  when testing YAML descriptions.
---

Follow folded instructions.
""",
        encoding="utf-8",
    )

    catalog = discover_skills(project, user_home=user_home, include_builtin=False)

    assert catalog.skills["folded-skill"].description == "Use this folded skill when testing YAML descriptions."


def test_prompt_catalog_uses_only_names_and_descriptions(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    project.mkdir()
    write_skill(project / ".agents" / "skills", "plain-skill", "Use this plain skill.")

    catalog = discover_skills(project, user_home=user_home, include_builtin=False)
    prompt = catalog.prompt_catalog(budget=SkillCatalogBudget(max_chars=1_000))

    assert "- `plain-skill`: Use this plain skill." in prompt
    assert "/plain-skill" not in prompt
    assert "(project)" not in prompt


def test_prompt_catalog_truncates_descriptions_before_omitting_skills(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    project.mkdir()
    skills_root = project / ".agents" / "skills"
    long_description = "Use this skill " + ("for a very long specialized workflow " * 8)
    for name in ("alpha-skill", "beta-skill", "gamma-skill"):
        write_skill(skills_root, name, long_description)

    catalog = discover_skills(project, user_home=user_home, include_builtin=False)
    render = catalog.render_prompt_catalog(SkillCatalogBudget(max_chars=95))

    assert render.report.included_count == 3
    assert render.report.omitted_count == 0
    assert render.report.truncated_description_count == 3
    assert "`alpha-skill`" in render.markdown
    assert "`beta-skill`" in render.markdown
    assert "`gamma-skill`" in render.markdown
    assert "Skill descriptions were shortened" in render.markdown
    assert long_description not in render.markdown


def test_prompt_catalog_omits_overflow_when_names_exceed_budget(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    project.mkdir()
    skills_root = project / ".agents" / "skills"
    for index in range(10):
        write_skill(skills_root, f"skill-{index:02d}", f"Use skill {index}.")

    catalog = discover_skills(project, user_home=user_home, include_builtin=False)
    render = catalog.render_prompt_catalog(SkillCatalogBudget(max_chars=30))

    assert render.report.included_count == 2
    assert render.report.omitted_count == 8
    assert "`skill-00`" in render.markdown
    assert "`skill-01`" in render.markdown
    assert "`skill-02`" not in render.markdown
    assert "8 additional skills were omitted" in render.markdown


def test_build_skill_prompt_extension_uses_context_window_budget(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    project.mkdir()
    write_skill(
        project / ".agents" / "skills",
        "budgeted-skill",
        "Use this skill " + ("when the model needs a long detailed workflow " * 12),
    )

    catalog = discover_skills(project, user_home=user_home, include_builtin=False)
    extension = build_skill_prompt_extension(catalog, context_window_tokens=1_000)

    assert extension is not None
    assert "`budgeted-skill`" in extension.markdown
    assert "Skill descriptions were shortened" in extension.markdown
    assert "(project)" not in extension.markdown


def test_context_window_budget_has_hard_cap() -> None:
    budget = SkillCatalogBudget.for_context_window(1_000_000)

    assert budget.max_chars == MAX_SKILL_METADATA_CHAR_BUDGET


def test_large_context_skill_prompt_extension_stays_bounded(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    project.mkdir()
    skills_root = project / ".agents" / "skills"
    long_description = "Use this skill " + ("when the model needs a long detailed stress workflow " * 10)
    for index in range(200):
        write_skill(skills_root, f"stress-skill-{index:03d}", long_description)

    catalog = discover_skills(project, user_home=user_home, include_builtin=False)
    extension = build_skill_prompt_extension(catalog, context_window_tokens=1_000_000)

    assert extension is not None
    assert len(extension.markdown) <= MAX_SKILL_METADATA_CHAR_BUDGET + 500
    assert "`stress-skill-000`" in extension.markdown
    assert "`stress-skill-199`" in extension.markdown
    assert "Skill descriptions were shortened" in extension.markdown


@pytest.mark.asyncio
async def test_list_skills_tool_is_bounded_and_queryable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    project.mkdir()
    skills_root = project / ".agents" / "skills"
    for index in range(60):
        write_skill(skills_root, f"alpha-{index:02d}", f"Use alpha skill {index}.")

    catalog = discover_skills(project, user_home=user_home, include_builtin=False)
    extension = build_skill_tool_extension(catalog, lambda: [])
    assert extension is not None
    list_skills = extension.tools["list_skills"]

    default_output = await list_skills()
    assert "`alpha-00`" in default_output
    assert "`alpha-49`" in default_output
    assert "`alpha-59`" not in default_output
    assert "10 matching skills were not shown" in default_output
    assert "/alpha-00" not in default_output
    assert "(project)" not in default_output

    queried_output = await list_skills(query="alpha-59")
    assert "`alpha-59`" in queried_output
    assert "Use alpha skill 59." in queried_output


def test_discover_skills_skips_missing_description_and_malformed_yaml(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    skills_root = project / ".agents" / "skills"
    missing = skills_root / "missing-description"
    malformed = skills_root / "malformed-skill"
    missing.mkdir(parents=True)
    malformed.mkdir(parents=True)
    (missing / "SKILL.md").write_text("---\nname: missing-description\n---\nbody\n", encoding="utf-8")
    (malformed / "SKILL.md").write_text(
        "---\nname: malformed-skill\ndescription: Use when: yaml breaks\n---\nbody\n",
        encoding="utf-8",
    )

    catalog = discover_skills(project, user_home=user_home, include_builtin=False)

    assert catalog.skills == {}
    assert len(catalog.diagnostics) == 2
    assert all(diagnostic.severity == "error" for diagnostic in catalog.diagnostics)


def test_activation_content_wraps_body_and_lists_resources(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    skill_dir = write_skill(project / ".agents" / "skills", "resource-skill", "Use resources.", body="# Steps")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "guide.md").write_text("reference content", encoding="utf-8")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.py").write_text("print('ok')", encoding="utf-8")

    catalog = discover_skills(project, user_home=user_home, include_builtin=False)
    content = catalog.activation_content("resource-skill")

    assert '<skill_content name="resource-skill">' in content
    assert "# Steps" in content
    assert "Skill directory:" in content
    assert "<file>references/guide.md</file>" in content
    assert "<file>scripts/run.py</file>" in content
    assert "name: resource-skill" not in content


def test_read_resource_rejects_path_traversal_and_caps_content(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    skill_dir = write_skill(project / ".agents" / "skills", "read-skill")
    (skill_dir / "big.txt").write_text("a" * 20, encoding="utf-8")

    catalog = discover_skills(project, user_home=user_home, include_builtin=False)

    assert catalog.read_resource("read-skill", "big.txt", max_chars=5).startswith("aaaaa\n\n[truncated")
    with pytest.raises(ValueError, match="inside the skill directory"):
        catalog.read_resource("read-skill", "../SKILL.md")


def test_activated_skill_names_scans_text_and_tool_results() -> None:
    history = [
        Message("user", [TextBlock('<skill_content name="one">body</skill_content>')]),
        Message("user", '<skill_content name="two">body</skill_content>'),
        Message(
            "user",
            [
                ToolResult(
                    tool_use_id="toolu_123",
                    content=[TextBlock('<skill_content name="three">body</skill_content>')],
                    name="activate_skill",
                    is_error=False,
                )
            ],
        ),
    ]

    assert activated_skill_names(history) == {"one", "two", "three"}
