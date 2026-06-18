"""Reusable prompt text loaded from bundled prompt templates."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import jinja2

_BASE_DIR = Path(__file__).parent / "prompt_templates"
_LOADER = jinja2.FileSystemLoader(str(_BASE_DIR))
_ENV = jinja2.Environment(
    loader=_LOADER,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


@lru_cache(maxsize=None)
def prompt_template_source(template_name: str) -> str:
    """Return the raw source for a bundled prompt template."""
    source, _, _ = _LOADER.get_source(_ENV, template_name)
    return source.strip()


@lru_cache(maxsize=None)
def _prompt_template(template_name: str) -> jinja2.Template:
    return _ENV.get_template(template_name)


def render_prompt_template(template_name: str, **variables: Any) -> str:
    """Render a bundled prompt template with Jinja variables."""
    return _prompt_template(template_name).render(**variables).strip()


THINK_HARD_PROMPT = render_prompt_template("auxiliary/tools/think_hard.system.md")
COMPRESSION_SUMMARY_SYSTEM_PROMPT = render_prompt_template("auxiliary/compression/summary.system.md")
SHELL_SAFETY_SYSTEM_PROMPT = render_prompt_template("auxiliary/terminal/safety.system.md")
SHELL_COMPRESSION_SYSTEM_PROMPT = render_prompt_template("auxiliary/terminal/output_summary.system.md")
SHARED_TASK_LIST_PROMPT = render_prompt_template("extensions/cli/shared_task_list.md")
PLANNING_QUESTION_PROMPT = render_prompt_template("extensions/cli/planning_questions.md")

# Compatibility templates for callers/tests that still use ``str.format`` or
# ``replace`` style placeholders.
COMPRESSION_SUMMARY_USER_PROMPT_TEMPLATE = prompt_template_source("auxiliary/compression/summary.user.md.j2").replace(
    "{{ history }}", "{HISTORY}"
)
IMPLEMENT_PLAN_PROMPT_TEMPLATE = prompt_template_source("user_tasks/cli/implement_plan.md.j2").replace(
    "{{ plan }}", "{plan}"
)
SKILL_CATALOG_PROMPT_TEMPLATE = prompt_template_source("extensions/skills/catalog.md.j2").replace(
    "{{ catalog }}", "{catalog}"
)
PLANNING_AGENT_SYSTEM_PROMPT_TEMPLATE = prompt_template_source("system/agents/planning.md.j2")


def build_compression_summary_user_prompt(history: str) -> str:
    return render_prompt_template("auxiliary/compression/summary.user.md.j2", history=history)


def build_implement_plan_prompt(plan: str, gigacode_enabled: bool = False) -> str:
    return render_prompt_template(
        "user_tasks/cli/implement_plan.md.j2", plan=plan, gigacode_enabled=gigacode_enabled
    )


def build_init_agents_prompt(arguments: str) -> str:
    return render_prompt_template("user_tasks/cli/init_agents.md.j2", arguments=arguments.strip())


def build_skill_catalog_prompt(catalog: str) -> str:
    return render_prompt_template("extensions/skills/catalog.md.j2", catalog=catalog)


def build_planning_agent_system_prompt(
    *,
    system_name: str,
    project_path: str,
    is_git_repo: bool,
    platform: str,
    date_today: str,
    model_name: str,
) -> str:
    return render_prompt_template(
        "system/agents/planning.md.j2",
        system_name=system_name,
        project_path=project_path,
        is_git_repo=is_git_repo,
        platform=platform,
        date_today=date_today,
        model_name=model_name,
    )
