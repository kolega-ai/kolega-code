from pathlib import Path

import pytest

from kolega_code.cli.skills import activated_skill_names, discover_skills
from kolega_code.agent.llm.models import Message, TextBlock, ToolResult


def write_skill(root: Path, name: str, description: str = "Use this skill for testing.", body: str = "Do the work.") -> Path:
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

    catalog = discover_skills(project, user_home=user_home)

    assert list(catalog.skills) == ["project-skill", "user-skill"]
    assert catalog.skills["project-skill"].scope == "project"
    assert catalog.skills["user-skill"].scope == "user"


def test_discover_skills_does_not_scan_singular_project_agent_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    project.mkdir()
    write_skill(project / ".agent" / "skills", "ignored-skill", "Use this ignored skill.")

    catalog = discover_skills(project, user_home=user_home)

    assert "ignored-skill" not in catalog.skills


def test_project_skill_overrides_user_skill_with_same_name(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    project.mkdir()
    write_skill(user_home / ".agents" / "skills", "shared-skill", "Use the user skill.", body="user body")
    write_skill(project / ".agents" / "skills", "shared-skill", "Use the project skill.", body="project body")

    catalog = discover_skills(project, user_home=user_home)

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

    catalog = discover_skills(project, user_home=user_home)

    assert catalog.skills["folded-skill"].description == "Use this folded skill when testing YAML descriptions."


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

    catalog = discover_skills(project, user_home=user_home)

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

    catalog = discover_skills(project, user_home=user_home)
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

    catalog = discover_skills(project, user_home=user_home)

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
