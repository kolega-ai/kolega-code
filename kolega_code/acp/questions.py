"""``ask_user_choice`` over ACP elicitation.

The TUI answers planning questions on its control channel; the ACP server
answers them through the client's ``elicitation/create`` (form mode), so the
editor renders a native question form and the accepted value returns as the
tool result. Clients without elicitation support get a tool error instead, so
the model falls back to asking in the conversation.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from kolega_code.agent import ToolExtension
from kolega_code.agent.tool_definitions import tool_description_asset
from kolega_code.tools import ASK_USER_CHOICE_INPUT_SCHEMA, ToolError
from kolega_code.tools.ask_user import normalize_choice_questions

QUESTION_TOOL_NAME = "ask_user_choice"

#: Callable the server binds at session build: (question, labels, descriptions) -> chosen label.
ElicitAnswer = Callable[[str, list[str], list[str]], Awaitable[str]]


def build_question_extension(question_state: dict[str, Any]) -> ToolExtension:
    """The ACP ``ask_user_choice`` tool, backed by a per-session state dict.

    ``question_state`` carries the server-bound ``elicit`` callable (set when
    the session registers with the transport) and the single-question-in-flight
    guard, mirroring the TUI's one pending question at a time.
    """

    async def ask_user_choice(questions: list[dict]) -> str:
        normalized = normalize_choice_questions(questions)
        if question_state.get("pending"):
            raise ToolError("A question is already waiting for an answer.")
        elicit = question_state.get("elicit")
        if elicit is None:
            raise ToolError(
                "This editor client does not support interactive questions; ask the user directly in chat instead."
            )
        question_state["pending"] = True
        try:
            answers: dict[str, str] = {}
            for clean_question, header, labels, descriptions in normalized:
                answer = await elicit(clean_question, labels, descriptions)
                answers[header or clean_question] = answer
            return json.dumps(answers)
        finally:
            question_state["pending"] = False

    return ToolExtension(
        name="acp-user-questions",
        tools={QUESTION_TOOL_NAME: ask_user_choice},
        tool_descriptions={QUESTION_TOOL_NAME: tool_description_asset(QUESTION_TOOL_NAME)},
        tool_schemas={QUESTION_TOOL_NAME: ASK_USER_CHOICE_INPUT_SCHEMA},
        # Both top-level agents expose it: PlanningAgent allows planning_tools,
        # CoderAgent allows coder_agent_tools. Sub-agents get neither group
        # from a non-propagating extension, so only the agent talking to the
        # user asks.
        tool_groups={
            "planning_tools": [QUESTION_TOOL_NAME],
            "coder_agent_tools": [QUESTION_TOOL_NAME],
        },
        propagate_to_sub_agents=False,
    )
