import json
from pathlib import Path

from kolega_code.cli.skills import BUNDLED_SKILLS_DIR, discover_skills


def write_skill(root: Path, name: str, body: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use {name} for testing precedence.\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_bundled_skills_are_discovered_by_default(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_home = tmp_path / "home"
    project.mkdir()
    bundled_skill_dirs = {
        path.name for path in BUNDLED_SKILLS_DIR.iterdir() if path.is_dir() and (path / "SKILL.md").is_file()
    }

    catalog = discover_skills(project, user_home=user_home)

    assert bundled_skill_dirs
    assert catalog.skills
    assert {record.scope for record in catalog.skills.values()} == {"bundled"}


def test_user_and_project_skills_override_bundled_skills(tmp_path: Path) -> None:
    bundled_root = tmp_path / "bundled"
    user_home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    write_skill(bundled_root, "shared-skill", "bundled body")
    write_skill(bundled_root, "bundled-only", "bundled-only body")
    write_skill(user_home / ".agents" / "skills", "shared-skill", "user body")
    write_skill(user_home / ".agents" / "skills", "user-only", "user-only body")

    user_catalog = discover_skills(project, user_home=user_home, bundled_root=bundled_root)

    assert user_catalog.skills["shared-skill"].scope == "user"
    assert "user body" in user_catalog.activation_content("shared-skill")
    assert set(user_catalog.skills) == {"bundled-only", "shared-skill", "user-only"}
    assert any("User skill `shared-skill` overrides bundled skill" in item.message for item in user_catalog.diagnostics)

    write_skill(project / ".agents" / "skills", "shared-skill", "project body")
    write_skill(project / ".agents" / "skills", "project-only", "project-only body")
    project_catalog = discover_skills(project, user_home=user_home, bundled_root=bundled_root)

    assert project_catalog.skills["shared-skill"].scope == "project"
    assert "project body" in project_catalog.activation_content("shared-skill")
    assert set(project_catalog.skills) == {"bundled-only", "project-only", "shared-skill", "user-only"}
    assert any(
        "Project skill `shared-skill` overrides user skill" in item.message for item in project_catalog.diagnostics
    )


def test_bundled_skill_directories_match_manifest() -> None:
    manifest_path = BUNDLED_SKILLS_DIR.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundled_skill_dirs = {path.name for path in BUNDLED_SKILLS_DIR.iterdir() if path.is_dir()}

    assert bundled_skill_dirs == set(manifest["skills"])
