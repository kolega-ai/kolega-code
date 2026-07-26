"""Interactive prompt flows for planning questions and permissions."""

from __future__ import annotations

import json
from typing import Optional

from rich.markup import escape
from textual.widgets.option_list import Option

from kolega_code.agent import PromptExtension, ToolExtension
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.prompts import (
    BUILD_QUESTION_PROMPT,
    PLANNING_QUESTION_PROMPT,
    SHARED_TASK_LIST_PROMPT,
    SHARED_TASK_LIST_READONLY_PROMPT,
)
from kolega_code.permissions import (
    PermissionDecision,
    PermissionRequest,
    PermissionRuleOption,
    PermissionStoreError,
    ProjectPermissionStore,
    allow_rule_options,
)
from kolega_code.tools import ASK_USER_CHOICE_INPUT_SCHEMA, ASK_USER_CHOICE_SHAPE_HINT, ToolError

from .. import messages, theme
from . import app_base as tui_app_base
from .constants import APPROVAL_OPTION_ID_PREFIX, QUESTION_OPTION_ID_PREFIX, QUESTION_TOOL_NAME
from .state import ConversationEntry, PendingApproval, PendingQuestion, TurnState
from .widgets import PromptPanel


class PromptFlowMixin(tui_app_base.KolegaAppBase):
    def _shared_task_list_prompt_extension(self) -> PromptExtension:
        return PromptExtension(
            id="cli-shared-task-list",
            title="Shared Task List",
            markdown=SHARED_TASK_LIST_PROMPT,
            modes=[AgentMode.CLI],
            # The task list is owned by the single top-level build agent; sub-agents
            # must not get it or they would race on the shared list.
            propagate_to_sub_agents=False,
        )

    def _shared_task_list_tool_extension(self) -> ToolExtension:
        async def get_task_list() -> str:
            """
            Return the shared CLI task list.

            Use this before planning or implementation work when you need the current task state.

            Returns:
                The current shared task list, or a note that no task list has been set.
            """
            return self.session.task_list_markdown or messages.TASK_LIST_EMPTY_MESSAGE

        async def update_task_list(task_list_markdown: str) -> str:
            """
            Replace the shared CLI task list.

            Format the list as Markdown checkboxes, for example `- [ ] inspect CLI state handling`.
            Use this after completing individual task-list items so progress is visible incrementally; do not wait
            until every TODO is complete before updating the list.

            Args:
                task_list_markdown: The full current shared task list as Markdown.

            Returns:
                A confirmation that the shared task list was updated.
            """
            self.session.task_list_markdown = task_list_markdown.strip()
            await self._save_session_async()
            self._refresh_planning_sidebar()
            return "Task list updated."

        return ToolExtension(
            name="cli-shared-task-list",
            tools={
                "get_task_list": get_task_list,
                "update_task_list": update_task_list,
            },
            tool_groups={
                "planning_tools": ["get_task_list", "update_task_list"],
                "cli_task_list_tools": ["get_task_list", "update_task_list"],
            },
            # Single-owner, shared mutable state: never hand it to sub-agents.
            propagate_to_sub_agents=False,
        )

    def _shared_task_list_readonly_prompt_extension(self) -> PromptExtension:
        return PromptExtension(
            id="cli-shared-task-list-readonly",
            title="Shared Task List (read-only)",
            markdown=SHARED_TASK_LIST_READONLY_PROMPT,
            modes=[AgentMode.CLI],
            # Same single-owner rule as the build-mode extension.
            propagate_to_sub_agents=False,
        )

    def _shared_task_list_readonly_tool_extension(self) -> ToolExtension:
        """Plan-mode task list access: read the build agent's list, never write it.

        Plan mode must not mutate session state, and ``session.task_list_markdown``
        is cleared only on thread reset — a plan-mode write would silently clobber
        a build session's in-flight progress. Reading it is what makes re-planning
        mid-implementation useful, so only the getter is exposed here. The docstring
        below is the model-facing tool description, so it is deliberately distinct
        from the build-mode getter's.
        """

        async def get_task_list() -> str:
            """
            Return the shared CLI task list, which the build agent owns.

            Use this when re-planning work that is already underway, to see what an in-progress build
            session has already completed. You cannot modify the list in plan mode; capture the remaining
            work in the plan you submit with `write_plan` instead.

            Returns:
                The current shared task list, or a note that no task list has been set.
            """
            return self.session.task_list_markdown or messages.TASK_LIST_EMPTY_MESSAGE

        return ToolExtension(
            name="cli-shared-task-list-readonly",
            tools={"get_task_list": get_task_list},
            # PlanningAgent only exposes extension tools declared in its
            # custom_tool_groups, of which planning_tools is one.
            tool_groups={
                "planning_tools": ["get_task_list"],
                "cli_task_list_tools": ["get_task_list"],
            },
            propagate_to_sub_agents=False,
        )

    def _planning_question_prompt_extension(self) -> PromptExtension:
        return PromptExtension(
            id="cli-planning-questions",
            title="Planning Questions",
            markdown=PLANNING_QUESTION_PROMPT,
            modes=[AgentMode.CLI],
            # ask_user_choice is a top-level, interactive planning tool; sub-agents
            # should not prompt the user.
            propagate_to_sub_agents=False,
        )

    def _build_question_prompt_extension(self) -> PromptExtension:
        """Build-mode framing for the same tool.

        Plan mode is a conversation, so asking is cheap there. Implementation is not:
        the guidance has to push toward deciding and stating the assumption, and
        reserve a prompt for choices that are expensive to get wrong.
        """
        return PromptExtension(
            id="cli-build-questions",
            title="Asking the User",
            markdown=BUILD_QUESTION_PROMPT,
            modes=[AgentMode.CLI],
            propagate_to_sub_agents=False,
        )

    def _planning_question_tool_extension(self) -> ToolExtension:
        async def ask_user_choice(questions: list[dict]) -> str:
            """
            Ask the user one or more multiple-choice questions and wait for their answers.

            Use this only for decisions that materially affect the outcome and cannot be settled from the
            repository. Each question has a short `header`, the `question` text, a `multiSelect` flag, and an
            `options` array of `{label, description}` choices. The user selects one option per question or
            types a custom free-text answer. Questions are asked one at a time, in order.

            Returns:
                A JSON object mapping each question's header (or its text) to the chosen option label
                or the user's custom answer.
            """
            normalized = self._normalize_choice_questions(questions)
            if self._pending_question is not None:
                raise ToolError("A planning question is already waiting for an answer.")

            answers: dict[str, str] = {}
            for clean_question, header, labels, descriptions in normalized:
                answer = await self._ask_user_choice(clean_question, labels, descriptions)
                answers[header or clean_question] = answer
            return json.dumps(answers)

        return ToolExtension(
            name="cli-planning-questions",
            tools={QUESTION_TOOL_NAME: ask_user_choice},
            tool_schemas={QUESTION_TOOL_NAME: ASK_USER_CHOICE_INPUT_SCHEMA},
            # Both top-level agents expose it: PlanningAgent allows planning_tools,
            # CoderAgent allows coder_agent_tools. Sub-agents get neither group from
            # a non-propagating extension, so only the agent talking to the user asks.
            tool_groups={
                "planning_tools": [QUESTION_TOOL_NAME],
                "coder_agent_tools": [QUESTION_TOOL_NAME],
            },
            propagate_to_sub_agents=False,
        )

    def _normalize_choice_questions(self, questions: object) -> list[tuple[str, str, list[str], list[str]]]:
        """Validate the structured questions input and return normalized questions.

        Strict: rejects malformed input with an instructive ToolError instead of coercing.
        Each result is (question_text, header, option_labels, option_descriptions).
        """
        if not isinstance(questions, list) or not questions:
            raise ToolError("'questions' must be a non-empty array of question objects. " + ASK_USER_CHOICE_SHAPE_HINT)

        normalized: list[tuple[str, str, list[str], list[str]]] = []
        for question in questions:
            if not isinstance(question, dict):
                raise ToolError("each item in 'questions' must be an object. " + ASK_USER_CHOICE_SHAPE_HINT)

            clean_question = str(question.get("question", "")).strip()
            if not clean_question:
                raise ToolError("each question must include non-empty 'question' text. " + ASK_USER_CHOICE_SHAPE_HINT)

            header = str(question.get("header", "")).strip()

            raw_options = question.get("options")
            if not isinstance(raw_options, list):
                raise ToolError(
                    "each question's 'options' must be an array of {label, description} objects. "
                    + ASK_USER_CHOICE_SHAPE_HINT
                )

            labels: list[str] = []
            descriptions: list[str] = []
            for option in raw_options:
                if not isinstance(option, dict):
                    raise ToolError(
                        "each option must be an object with a 'label' (and ideally a 'description'). "
                        + ASK_USER_CHOICE_SHAPE_HINT
                    )
                label = str(option.get("label", "")).strip()
                if not label:
                    continue
                labels.append(label)
                descriptions.append(str(option.get("description", "")).strip())

            if len(labels) < 2:
                raise ToolError(
                    "each question needs at least two options, each with a non-empty 'label'. "
                    + ASK_USER_CHOICE_SHAPE_HINT
                )

            normalized.append((clean_question, header, labels, descriptions))

        return normalized

    async def _ask_user_choice(
        self, question: str, options: list[str], descriptions: Optional[list[str]] = None
    ) -> str:
        """Ask the controlling client a planning question over the control channel.

        Like permission prompts, this goes out as an event and comes back as a
        response, so any frontend can answer it. The tool raises rather than
        inventing an answer when nobody can: a plan built on a fabricated choice
        would be worse than a failed tool call.
        """
        response = await self.session_runtime.control.request(
            "question",
            {
                "question": question,
                "options": list(options),
                "descriptions": list(descriptions or []),
            },
            default={"answer": None},
        )
        answer = response.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ToolError(
                "No answer was given for this planning question. "
                "Proceed without it, or state the assumption you are making instead."
            )
        return answer

    async def _answer_question_option(self, option_index: int) -> None:
        if self._pending_question is None:
            return
        if option_index < 0 or option_index >= len(self._pending_question.options):
            return
        await self._answer_pending_question(self._pending_question.options[option_index])

    def _begin_question(
        self,
        request_id: str,
        question: str,
        options: list[str],
        descriptions: Optional[list[str]] = None,
    ) -> None:
        """Show a planning question announced on the control channel."""
        self._pending_question = PendingQuestion(
            question=question,
            options=options,
            descriptions=descriptions,
            request_id=request_id,
        )
        self._show_question_options(question, options, descriptions)
        self._set_composer_status(messages.QUESTION_PLACEHOLDER)
        self._set_chat_enabled(True)
        self._refresh_input_area_visibility()
        self._update_activity_progress(messages.WAITING_FOR_ANSWER, state=TurnState.WAITING_FOR_USER)

    def _clear_question_if_resolved(self, request_id: str) -> None:
        """Drop the prompt once the channel reports it settled by any means."""
        pending = self._pending_question
        if pending is None or pending.request_id != request_id:
            return
        self._pending_question = None
        self._set_question_actions_visible(False)
        self._refresh_input_area_visibility()

    async def _answer_pending_question(self, answer: str) -> None:
        pending_question = self._pending_question
        if pending_question is None:
            return

        clean_answer = answer.strip()
        if not clean_answer:
            self._set_composer_status(messages.QUESTION_PLACEHOLDER)
            return

        self._pending_question = None
        self._set_question_actions_visible(False)
        self._refresh_input_area_visibility()
        self._add_conversation_entry(ConversationEntry(kind="question", content=pending_question.question))
        self._add_conversation_entry(ConversationEntry(kind="user", content=clean_answer))
        self._answer_question_request(pending_question.request_id, clean_answer)

        if self._turn_active:
            self._set_composer_status(messages.QUEUE_PLACEHOLDER)
            self._clear_composer_hint()
            self._set_chat_enabled(True)
            self._update_progress(messages.WORKING, complete=False, state=TurnState.GENERATING)
        else:
            self._restore_composer_placeholder()
            self._set_chat_enabled(self.agent is not None)

    def _show_question_options(
        self, question: str, options: list[str], descriptions: Optional[list[str]] = None
    ) -> None:
        try:
            panel = self.query_one("#question_prompt", PromptPanel)
        except Exception:
            return
        option_widgets = [
            Option(
                self._question_option_label(index, option, self._option_description(descriptions, index)),
                id=f"{QUESTION_OPTION_ID_PREFIX}{index}",
            )
            for index, option in enumerate(options)
        ]
        panel.prompt(escape(question), option_widgets)
        self._focus_active_prompt()

    def _begin_approval(self, request_id: str, request: PermissionRequest) -> None:
        """Show a permission prompt announced on the control channel.

        This client learns about a prompt the same way any other frontend does, by
        receiving a control_requested event, rather than through a callback only a
        co-located UI could be handed. Mode checks and saved-rule matching already
        happened in the session runtime, so anything reaching here genuinely needs
        a person.
        """
        rule_options = allow_rule_options(request)
        self._pending_approval = PendingApproval(
            request=request,
            rule_options=rule_options,
            request_id=request_id,
        )
        self._show_approval_options(rule_options)
        self._set_composer_status(messages.APPROVAL_PLACEHOLDER)
        self._set_chat_enabled(False)
        self._refresh_input_area_visibility()
        self._update_activity_progress(messages.WAITING_FOR_PERMISSION, state=TurnState.WAITING_FOR_USER)

    def _clear_approval_if_resolved(self, request_id: str) -> None:
        """Drop the prompt once the channel reports it answered.

        Covers the answers this client did not give: a timeout, or the request
        being settled because control was released.
        """
        pending = self._pending_approval
        if pending is None or pending.request_id != request_id:
            return
        self._pending_approval = None
        self._set_approval_actions_visible(False)
        self._refresh_input_area_visibility()

    def _show_approval_options(self, rule_options: list[PermissionRuleOption]) -> None:
        self._set_approval_actions_visible(True)

    async def _answer_approval_option(self, option_index: int) -> None:
        pending = self._pending_approval
        if pending is None:
            return

        decision: PermissionDecision
        if option_index == 0:
            decision = PermissionDecision(allowed=True, reason="Allowed once by the user.")
            chosen_label = "Allow once"
        elif option_index == 1:
            decision = PermissionDecision(allowed=False, reason="Denied by the user.")
            chosen_label = "Deny"
        else:
            rule_index = option_index - 2
            if rule_index < 0 or rule_index >= len(pending.rule_options):
                return
            rule = pending.rule_options[rule_index].rule
            chosen_label = pending.rule_options[rule_index].label
            try:
                ProjectPermissionStore(self.project_path).add_rule(rule)
            except PermissionStoreError as exc:
                self._notify_user(str(exc), severity="warning")
                decision = PermissionDecision(allowed=True, reason="Allowed once because the rule could not be saved.")
            else:
                decision = PermissionDecision(allowed=True, reason="Allowed by a saved rule.", rule=rule)

        self._pending_approval = None
        self._set_approval_actions_visible(False)
        self._refresh_input_area_visibility()
        self._add_conversation_entry(
            ConversationEntry(kind="question", content=self._format_permission_content(pending.request))
        )
        self._add_conversation_entry(ConversationEntry(kind="user", content=chosen_label))
        self._answer_permission_request(pending.request_id, decision)

        if self._turn_active:
            self._set_composer_status(messages.QUEUE_PLACEHOLDER)
            self._clear_composer_hint()
            self._set_chat_enabled(True)
            self._update_progress(messages.WORKING, complete=False, state=TurnState.GENERATING)
        else:
            self._restore_composer_placeholder()
            self._set_chat_enabled(self.agent is not None)

    def _format_permission_content(self, request: PermissionRequest) -> str:
        if request.kind.value == "command":
            return "\n".join(["Allow the agent to run this command?", "", request.command])
        if request.kind.value == "mcp":
            return "\n".join(
                [
                    "Allow the agent to call this MCP tool?",
                    "",
                    f"Server: {request.mcp_server}",
                    f"Tool: {request.mcp_tool}",
                ]
            )
        target = f" on {request.path}" if request.path else ""
        return f"Allow the agent to run {request.tool_name}{target}?"

    def _show_effort_options(self) -> None:
        self._set_effort_actions_visible(True)
        self._focus_active_prompt()

    def _show_model_options(self) -> None:
        self._set_model_actions_visible(True)
        self._focus_active_prompt()

    def _show_theme_options(self) -> None:
        self._set_theme_actions_visible(True)
        self._focus_active_prompt()

    def _set_question_actions_visible(self, visible: bool) -> None:
        try:
            panel = self.query_one("#question_prompt", PromptPanel)
        except Exception:
            return
        if visible:
            panel.display = True
            panel.actions.display = True
            self._focus_active_prompt()
        else:
            panel.hide()

    def _set_approval_actions_visible(self, visible: bool) -> None:
        try:
            panel = self.query_one("#approval_prompt", PromptPanel)
        except Exception:
            return
        if visible and self._pending_approval is not None:
            labels = ["Allow once", "Deny", *(option.label for option in self._pending_approval.rule_options)]
            options = [
                Option(self._question_option_label(index, label), id=f"{APPROVAL_OPTION_ID_PREFIX}{index}")
                for index, label in enumerate(labels)
            ]
            panel.prompt(escape(self._format_permission_content(self._pending_approval.request)), options)
            self._focus_active_prompt()
        else:
            panel.hide()

    def _cancel_pending_question(self) -> None:
        pending_question = self._pending_question
        if pending_question is not None:
            # Settle the request so a cancelled turn does not leave the tool
            # waiting on a prompt this client has stopped showing.
            self._answer_question_request(pending_question.request_id, "")
        self._pending_question = None
        self._set_question_actions_visible(False)
        self._refresh_input_area_visibility()

    def _cancel_pending_approval(self) -> None:
        pending_approval = self._pending_approval
        if pending_approval is not None:
            # Cancelling the turn must not leave the agent waiting on a prompt
            # this client has stopped showing.
            self._answer_permission_request(
                pending_approval.request_id,
                PermissionDecision(allowed=False, reason="The turn was cancelled before this was answered."),
            )
        self._pending_approval = None
        self._set_approval_actions_visible(False)
        self._refresh_input_area_visibility()

    def _question_option_label(self, index: int, option: str, description: str = "") -> str:
        if description:
            return f"{index + 1}. {option} — {description}"
        return f"{index + 1}. {option}"

    @staticmethod
    def _option_description(descriptions: Optional[list[str]], index: int) -> str:
        if descriptions is not None and 0 <= index < len(descriptions):
            return descriptions[index]
        return ""

    def _cancel_pending_model_selection(self) -> None:
        self._pending_model_selection = None
        self._set_model_actions_visible(False)

    def _model_option_label(self, index: int, label: str, value: str, provider: str) -> str:
        current_provider, current_model = self._startup_model()
        current_suffix = " current" if provider == current_provider and value == current_model else ""
        return f"{index + 1}. {label} ({value}){current_suffix}"

    def _cancel_pending_effort_selection(self) -> None:
        self._pending_effort_selection = None
        self._set_effort_actions_visible(False)

    def _effort_option_label(self, index: int, label: str, value: str) -> str:
        current_suffix = " current" if value == self._startup_thinking_effort() else ""
        return f"{index + 1}. {label} ({value}){current_suffix}"

    def _cancel_pending_theme_selection(self) -> None:
        self._pending_theme_selection = None
        self._set_theme_actions_visible(False)

    def _theme_option_label(self, index: int, name: str) -> str:
        current = self.settings.active_theme or theme.DEFAULT_THEME_NAME
        current_suffix = " current" if name == current else ""
        return f"{index + 1}. {name}{current_suffix}"
