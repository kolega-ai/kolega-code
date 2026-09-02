"""``ask_user_choice`` over the gateway control relay.

The TUI answers planning questions on its control channel; the gateway answers
them through the relay's inline buttons, so the accepted value returns as the
tool result and the prompt lands in the session recording. Mirrors the ACP
question extension — the ``elicit`` callable is bound by the session host to
the session's own control channel.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from kolega_code.agent import ToolExtension
from kolega_code.agent.tool_definitions import tool_description_asset
from kolega_code.tools import ASK_USER_CHOICE_INPUT_SCHEMA, ToolError
from kolega_code.tools.ask_user import normalize_choice_questions

QUESTION_TOOL_NAME = "ask_user_choice"

#: (question, labels, descriptions) -> chosen label.
ElicitAnswer = Callable[[str, list[str], list[str]], Awaitable[str]]


def build_question_extension(question_state: dict[str, Any]) -> ToolExtension:
    """The gateway's ``ask_user_choice`` tool, backed by a per-session state dict.

    ``question_state`` carries the session-bound ``elicit`` callable (set when
    the session host builds the agent) and the single-question-in-flight guard,
    mirroring the TUI's one pending question at a time.
    """

    async def ask_user_choice(questions: list[dict]) -> str:
        normalized = normalize_choice_questions(questions)
        if question_state.get("pending"):
            raise ToolError("A question is already waiting for an answer.")
        elicit = question_state.get("elicit")
        if elicit is None:
            raise ToolError("Interactive questions are unavailable here; ask the user directly in chat instead.")
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
        name="gateway-user-questions",
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
