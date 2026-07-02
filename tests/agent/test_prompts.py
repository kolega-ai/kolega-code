from pathlib import Path

from kolega_code.agent import prompts
from kolega_code.agent.orchestration.guide import GIGACODE_AUTHORING_GUIDE


def test_static_prompt_templates_load() -> None:
    assert "software architect" in prompts.THINK_HARD_PROMPT
    assert "continuity briefing" in prompts.COMPRESSION_SUMMARY_SYSTEM_PROMPT
    assert "evaluating shell commands for safety" in prompts.SHELL_SAFETY_SYSTEM_PROMPT
    assert "analyzing shell command output" in prompts.SHELL_COMPRESSION_SYSTEM_PROMPT
    assert "get_task_list" in prompts.SHARED_TASK_LIST_PROMPT
    assert "ask_user_choice" in prompts.PLANNING_QUESTION_PROMPT
    assert "read-only" in prompts.CURRENT_PLAN_ARTIFACT_PROMPT_TEMPLATE


def test_compression_summary_user_prompt_renders_history() -> None:
    rendered = prompts.build_compression_summary_user_prompt("User: hello")

    assert "User: hello" in rendered
    assert "{{ history }}" not in rendered
    assert "{HISTORY}" in prompts.COMPRESSION_SUMMARY_USER_PROMPT_TEMPLATE


def test_implement_plan_prompt_renders_plan() -> None:
    rendered = prompts.build_implement_plan_prompt("- [ ] update docs")

    assert "Implement the approved plan below." in rendered
    assert "- [ ] update docs" in rendered
    assert "{plan}" in prompts.IMPLEMENT_PLAN_PROMPT_TEMPLATE


def test_implement_plan_prompt_renders_plan_artifact_path_when_provided() -> None:
    rendered = prompts.build_implement_plan_prompt(
        "- [ ] update docs",
        plan_artifact_path="/tmp/kolega/plans/session/current-plan.md",
    )

    assert "/tmp/kolega/plans/session/current-plan.md" in rendered
    assert "conversation is compacted" in rendered
    assert "read_entire_file" in rendered
    assert "read-only" in rendered
    assert "- [ ] update docs" in rendered


def test_current_plan_artifact_prompt_renders_path() -> None:
    rendered = prompts.build_current_plan_artifact_prompt("/tmp/kolega/plans/session/current-plan.md")

    assert "/tmp/kolega/plans/session/current-plan.md" in rendered
    assert "read_entire_file" in rendered
    assert "read_file_section" in rendered
    assert "read-only" in rendered


def test_implement_plan_prompt_omits_gigacode_nudge_by_default() -> None:
    rendered = prompts.build_implement_plan_prompt("- [ ] update docs")

    assert "run_workflow" not in rendered
    assert "gigacode is enabled" not in rendered


def test_implement_plan_prompt_includes_gigacode_nudge_when_enabled() -> None:
    rendered = prompts.build_implement_plan_prompt("- [ ] update docs", gigacode_enabled=True)

    assert "gigacode is enabled" in rendered
    assert "run_workflow" in rendered
    assert "independent" in rendered
    # The plan still renders alongside the nudge.
    assert "- [ ] update docs" in rendered


def test_gigacode_guide_points_agents_to_artifact_transcripts() -> None:
    assert "resultPath" in GIGACODE_AUTHORING_GUIDE
    assert "transcriptPath" in GIGACODE_AUTHORING_GUIDE
    assert "Never re-run a completed workflow solely" in GIGACODE_AUTHORING_GUIDE
    assert "READ `resultPath` or `transcriptPath`" in GIGACODE_AUTHORING_GUIDE


def test_init_agents_prompt_renders_arguments() -> None:
    rendered = prompts.build_init_agents_prompt("focus on Python packaging")

    assert "Create or update `AGENTS.md` for this repository." in rendered
    assert "`focus on Python packaging`" in rendered
    assert "$ARGUMENTS" not in rendered
    assert "{{ arguments }}" not in rendered


def test_skill_catalog_prompt_renders_catalog() -> None:
    rendered = prompts.build_skill_catalog_prompt("- `demo-skill`: Use demo workflow")

    assert "Agent Skills discovered" in rendered
    assert "- `demo-skill`: Use demo workflow" in rendered
    assert "{catalog}" in prompts.SKILL_CATALOG_PROMPT_TEMPLATE


def test_planning_agent_prompt_renders_environment() -> None:
    rendered = prompts.build_planning_agent_system_prompt(
        system_name="Kolega Code",
        project_path="/repo",
        is_git_repo=True,
        platform="darwin",
        date_today="2026-06-17",
        model_name="test-model",
    )

    assert "Kolega Code's planning agent" in rendered
    assert "Working directory: /repo" in rendered
    assert "Is directory a git repo: True" in rendered
    assert "Model: test-model" in rendered
    assert "submit it through `write_plan`; do not only print the plan" in rendered
    assert "complete replacement plan that incorporates those decisions" in rendered


def test_prompt_template_tree_uses_canonical_agents_md_naming() -> None:
    template_root = Path(prompts.__file__).parent / "prompt_templates"

    assert (template_root / "system" / "includes" / "agents_md_instructions.md").is_file()
    assert not (template_root / "common" / "kolega_md_instructions.md").exists()
    assert not (template_root / "agents").exists()
    assert not (template_root / "cli").exists()
    assert not (template_root / "tasks").exists()
