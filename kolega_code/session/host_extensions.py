"""Host-callback extensions shared by interactive frontends."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from kolega_code.agent import PromptExtension, ToolExtension
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.prompts import (
    BUILD_QUESTION_PROMPT,
    PLANNING_QUESTION_PROMPT,
    SHARED_TASK_LIST_PROMPT,
    SHARED_TASK_LIST_READONLY_PROMPT,
)
from kolega_code.agent.tool_definitions import tool_description_asset
from kolega_code.tools import ASK_USER_CHOICE_INPUT_SCHEMA, ToolError
from kolega_code.tools.ask_user import normalize_choice_questions


def shared_task_list_prompt_extension() -> PromptExtension:
    return PromptExtension(
        id="cli-shared-task-list",
        title="Shared Task List",
        markdown=SHARED_TASK_LIST_PROMPT,
        modes=[AgentMode.CLI],
        # Shared mutable state belongs to the top-level agent, never sub-agents.
        propagate_to_sub_agents=False,
    )


def shared_task_list_tool_extension(
    *,
    get_task_list_markdown: Callable[[], str],
    set_task_list_markdown: Callable[[str], None],
    save_session: Callable[[], Awaitable[None]],
    refresh_task_list: Callable[[], None],
    empty_message: str = "No task list has been set.",
) -> ToolExtension:
    async def get_task_list() -> str:
        return get_task_list_markdown() or empty_message

    async def update_task_list(task_list_markdown: str) -> str:
        set_task_list_markdown(task_list_markdown.strip())
        await save_session()
        refresh_task_list()
        return "Task list updated."

    return ToolExtension(
        name="cli-shared-task-list",
        tools={
            "get_task_list": get_task_list,
            "update_task_list": update_task_list,
        },
        tool_descriptions={
            "get_task_list": tool_description_asset("get_task_list"),
            "update_task_list": tool_description_asset("update_task_list"),
        },
        tool_schemas={
            "get_task_list": {"type": "object", "properties": {}, "required": []},
            "update_task_list": {
                "type": "object",
                "properties": {
                    "task_list_markdown": {
                        "type": "string",
                        "description": "The full current shared task list as Markdown.",
                    }
                },
                "required": ["task_list_markdown"],
            },
        },
        tool_groups={
            "planning_tools": ["get_task_list", "update_task_list"],
            "cli_task_list_tools": ["get_task_list", "update_task_list"],
        },
        propagate_to_sub_agents=False,
    )


def shared_task_list_readonly_prompt_extension() -> PromptExtension:
    return PromptExtension(
        id="cli-shared-task-list-readonly",
        title="Shared Task List (read-only)",
        markdown=SHARED_TASK_LIST_READONLY_PROMPT,
        modes=[AgentMode.CLI],
        propagate_to_sub_agents=False,
    )


def shared_task_list_readonly_tool_extension(
    *,
    get_task_list_markdown: Callable[[], str],
    empty_message: str = "No task list has been set.",
) -> ToolExtension:
    async def get_task_list() -> str:
        return get_task_list_markdown() or empty_message

    return ToolExtension(
        name="cli-shared-task-list-readonly",
        tools={"get_task_list": get_task_list},
        tool_descriptions={"get_task_list": tool_description_asset("get_task_list_readonly")},
        tool_schemas={"get_task_list": {"type": "object", "properties": {}, "required": []}},
        # planning_tools makes the getter available to the read-only planning agent.
        tool_groups={
            "planning_tools": ["get_task_list"],
            "cli_task_list_tools": ["get_task_list"],
        },
        propagate_to_sub_agents=False,
    )


def planning_question_prompt_extension() -> PromptExtension:
    return PromptExtension(
        id="cli-planning-questions",
        title="Planning Questions",
        markdown=PLANNING_QUESTION_PROMPT,
        modes=[AgentMode.CLI],
        propagate_to_sub_agents=False,
    )


def build_question_prompt_extension() -> PromptExtension:
    return PromptExtension(
        id="cli-build-questions",
        title="Asking the User",
        markdown=BUILD_QUESTION_PROMPT,
        modes=[AgentMode.CLI],
        propagate_to_sub_agents=False,
    )


def planning_question_tool_extension(
    *,
    ask_user_choice: Callable[[str, list[str], list[str]], Awaitable[str]],
    has_pending_question: Callable[[], bool],
    normalize_questions: Callable[[object], list[tuple[str, str, list[str], list[str]]]] = normalize_choice_questions,
) -> ToolExtension:
    async def ask_questions(questions: list[dict]) -> str:
        normalized = normalize_questions(questions)
        if has_pending_question():
            raise ToolError("A planning question is already waiting for an answer.")

        answers: dict[str, str] = {}
        for clean_question, header, labels, descriptions in normalized:
            answer = await ask_user_choice(clean_question, labels, descriptions)
            answers[header or clean_question] = answer
        return json.dumps(answers)

    return ToolExtension(
        name="cli-planning-questions",
        tools={"ask_user_choice": ask_questions},
        tool_descriptions={"ask_user_choice": tool_description_asset("ask_user_choice")},
        tool_schemas={"ask_user_choice": ASK_USER_CHOICE_INPUT_SCHEMA},
        tool_groups={
            "planning_tools": ["ask_user_choice"],
            "coder_agent_tools": ["ask_user_choice"],
        },
        propagate_to_sub_agents=False,
    )
