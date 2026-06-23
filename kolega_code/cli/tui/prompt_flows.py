"""Interactive prompt flows for planning questions and permissions."""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from rich.markup import escape
from textual.widgets.option_list import Option

from kolega_code.agent import PromptExtension, ToolExtension
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.prompts import PLANNING_QUESTION_PROMPT, SHARED_TASK_LIST_PROMPT
from kolega_code.permissions import (
    PermissionDecision,
    PermissionMode,
    PermissionRequest,
    PermissionRuleOption,
    PermissionStoreError,
    ProjectPermissionStore,
    allow_rule_options,
)
from kolega_code.tools import ASK_USER_CHOICE_INPUT_SCHEMA, ASK_USER_CHOICE_SHAPE_HINT, ToolError

from .. import messages, theme
from .constants import APPROVAL_OPTION_ID_PREFIX, PLAN_INTERACTION_MODE, QUESTION_OPTION_ID_PREFIX, QUESTION_TOOL_NAME
from .state import ConversationEntry, PendingApproval, PendingQuestion, TurnState
from .widgets import PromptPanel


class PromptFlowMixin:
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

    def _planning_question_tool_extension(self) -> ToolExtension:
        async def ask_user_choice(questions: list[dict]) -> str:
            """
            Ask the user one or more multiple-choice planning questions and wait for their answers.

            Use this only for planning decisions that materially affect the final plan. Each question has a
            short `header`, the `question` text, a `multiSelect` flag, and an `options` array of
            `{label, description}` choices. The user selects one option per question or types a custom
            free-text answer. Questions are asked one at a time, in order.

            Returns:
                A JSON object mapping each question's header (or its text) to the chosen option label
                or the user's custom answer.
            """
            if self.interaction_mode != PLAN_INTERACTION_MODE:
                raise ToolError("ask_user_choice is only available in planning mode.")

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
            tool_groups={"planning_tools": [QUESTION_TOOL_NAME]},
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
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_question = PendingQuestion(
            question=question, options=options, future=future, descriptions=descriptions
        )
        self._show_question_options(question, options, descriptions)
        self._set_composer_status(messages.QUESTION_PLACEHOLDER)
        self._set_chat_enabled(True)
        self._update_activity_progress(messages.WAITING_FOR_ANSWER, state=TurnState.WAITING_FOR_USER)

        try:
            return await future
        finally:
            if self._pending_question is not None and self._pending_question.future is future:
                self._pending_question = None
                self._set_question_actions_visible(False)

    async def _answer_question_option(self, option_index: int) -> None:
        if self._pending_question is None:
            return
        if option_index < 0 or option_index >= len(self._pending_question.options):
            return
        await self._answer_pending_question(self._pending_question.options[option_index])

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
        self._add_conversation_entry(ConversationEntry(kind="question", content=pending_question.question))
        self._add_conversation_entry(ConversationEntry(kind="user", content=clean_answer))
        if not pending_question.future.done():
            pending_question.future.set_result(clean_answer)

        if self._turn_active:
            self._restore_composer_placeholder()
            self._set_chat_enabled(False)
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

    async def _permission_callback(self, request: PermissionRequest) -> PermissionDecision:
        if self.permission_mode != PermissionMode.ASK:
            return PermissionDecision(allowed=True)

        async with self._permission_lock:
            store = ProjectPermissionStore(self.project_path)
            try:
                matched_rule = store.first_match(request)
            except PermissionStoreError as exc:
                matched_rule = None
                self._notify_user(str(exc), severity="warning")

            if matched_rule is not None:
                return PermissionDecision(allowed=True, reason=f"Allowed by saved rule {matched_rule.id}.")

            return await self._ask_permission(request)

    async def _ask_permission(self, request: PermissionRequest) -> PermissionDecision:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[PermissionDecision] = loop.create_future()
        rule_options = allow_rule_options(request)
        self._pending_approval = PendingApproval(request=request, future=future, rule_options=rule_options)
        self._show_approval_options(rule_options)
        self._set_composer_status(messages.APPROVAL_PLACEHOLDER)
        self._set_chat_enabled(False)
        self._update_activity_progress(messages.WAITING_FOR_PERMISSION, state=TurnState.WAITING_FOR_USER)

        try:
            return await future
        finally:
            if self._pending_approval is not None and self._pending_approval.future is future:
                self._pending_approval = None
                self._set_approval_actions_visible(False)

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
        self._add_conversation_entry(
            ConversationEntry(kind="question", content=self._format_permission_content(pending.request))
        )
        self._add_conversation_entry(ConversationEntry(kind="user", content=chosen_label))
        if not pending.future.done():
            pending.future.set_result(decision)

        if self._turn_active:
            self._restore_composer_placeholder()
            self._set_chat_enabled(False)
            self._update_progress(messages.WORKING, complete=False, state=TurnState.GENERATING)
        else:
            self._restore_composer_placeholder()
            self._set_chat_enabled(self.agent is not None)

    def _format_permission_content(self, request: PermissionRequest) -> str:
        if request.kind.value == "command":
            return "\n".join(["Allow the agent to run this command?", "", request.command])
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
        if pending_question is not None and not pending_question.future.done():
            pending_question.future.cancel()
        self._pending_question = None
        self._set_question_actions_visible(False)

    def _cancel_pending_approval(self) -> None:
        pending_approval = self._pending_approval
        if pending_approval is not None and not pending_approval.future.done():
            pending_approval.future.cancel()
        self._pending_approval = None
        self._set_approval_actions_visible(False)

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
