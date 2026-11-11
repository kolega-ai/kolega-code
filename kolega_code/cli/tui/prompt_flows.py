"""Interactive prompt flows for planning questions and permissions."""

from __future__ import annotations

from typing import Optional

from rich.markup import escape
from textual.widgets.option_list import Option

from kolega_code.agent import PromptExtension, ToolExtension
from kolega_code.events import AgentEventEmitter
from kolega_code.permissions import (
    PermissionDecision,
    PermissionRequest,
    PermissionRuleOption,
)
from kolega_code.session.host_extensions import (
    build_question_prompt_extension,
    planning_question_prompt_extension,
    planning_question_tool_extension,
    shared_task_list_prompt_extension,
    shared_task_list_readonly_prompt_extension,
    shared_task_list_readonly_tool_extension,
    shared_task_list_tool_extension,
)
from kolega_code.tools import ToolError
from kolega_code.tools.ask_user import normalize_choice_questions

from .. import messages, theme
from . import app_base as tui_app_base
from .constants import APPROVAL_OPTION_ID_PREFIX, QUESTION_OPTION_ID_PREFIX
from .state import ConversationEntry, PendingApproval, PendingQuestion, TurnState
from .widgets import PromptPanel


class PromptFlowMixin(tui_app_base.KolegaAppBase):
    def _shared_task_list_prompt_extension(self) -> PromptExtension:
        return shared_task_list_prompt_extension()

    def _shared_task_list_tool_extension(self) -> ToolExtension:
        session = self.session
        token = object()
        self._task_list_extension_token = token
        emitter = self._task_list_event_emitter()

        def set_markdown(markdown: str) -> None:
            if (
                self.session is not session
                or self._task_list_extension_token is not token
                or self.agent is None
                or self._exit
            ):
                raise ToolError("This task-list tool belongs to an inactive agent.")
            session.task_list_markdown = markdown

        return shared_task_list_tool_extension(
            get_task_list_markdown=lambda: session.task_list_markdown,
            set_task_list_markdown=set_markdown,
            save_session=lambda: self._save_session_async(),
            refresh_task_list=lambda: self._refresh_planning_sidebar() if self.session is session else None,
            empty_message=messages.TASK_LIST_EMPTY_MESSAGE,
            emit_task_list=emitter.task_list_updated,
        )

    def _task_list_event_emitter(self) -> AgentEventEmitter:
        return AgentEventEmitter(
            self.recording_connection_manager,
            self.session.workspace_id,
            self.session.thread_id,
            sender="cli",
        )

    def _shared_task_list_readonly_prompt_extension(self) -> PromptExtension:
        return shared_task_list_readonly_prompt_extension()

    def _shared_task_list_readonly_tool_extension(self) -> ToolExtension:
        self._task_list_extension_token = object()
        session = self.session
        return shared_task_list_readonly_tool_extension(
            get_task_list_markdown=lambda: session.task_list_markdown,
            empty_message=messages.TASK_LIST_EMPTY_MESSAGE,
        )

    def _planning_question_prompt_extension(self) -> PromptExtension:
        return planning_question_prompt_extension()

    def _build_question_prompt_extension(self) -> PromptExtension:
        return build_question_prompt_extension()

    def _planning_question_tool_extension(self) -> ToolExtension:
        return planning_question_tool_extension(
            ask_user_choice=lambda question, options, descriptions: self._ask_user_choice(
                question, options, descriptions
            ),
            has_pending_question=lambda: self._pending_question is not None,
            normalize_questions=lambda questions: self._normalize_choice_questions(questions),
        )

    def _normalize_choice_questions(self, questions: object) -> list[tuple[str, str, list[str], list[str]]]:
        """Validate the structured questions input and return normalized questions.

        Strict: rejects malformed input with an instructive ToolError instead of coercing.
        Each result is (question_text, header, option_labels, option_descriptions).
        """
        return normalize_choice_questions(questions)

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
    ) -> bool:
        """Show a planning question announced on the control channel."""
        self._pending_question = PendingQuestion(
            question=question,
            options=options,
            descriptions=descriptions,
            request_id=request_id,
        )
        try:
            shown = self._show_question_options(question, options, descriptions)
        except Exception:
            shown = False
        if not shown:
            self._pending_question = None
            return False
        self._set_composer_status(messages.QUESTION_PLACEHOLDER)
        self._set_chat_enabled(True)
        self._refresh_input_area_visibility()
        self._update_activity_progress(messages.WAITING_FOR_ANSWER, state=TurnState.WAITING_FOR_USER)
        return True

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
    ) -> bool:
        try:
            panel = self.query_one("#question_prompt", PromptPanel)
        except Exception:
            return False
        option_widgets = [
            Option(
                self._question_option_label(index, option, self._option_description(descriptions, index)),
                id=f"{QUESTION_OPTION_ID_PREFIX}{index}",
            )
            for index, option in enumerate(options)
        ]
        panel.prompt(escape(question), option_widgets)
        self._focus_active_prompt()
        return True

    def _begin_approval(
        self,
        request_id: str,
        request: PermissionRequest,
        rule_options: list[PermissionRuleOption],
    ) -> bool:
        """Show a permission prompt announced on the control channel.

        This client learns about a prompt the same way any other frontend does, by
        receiving a control_requested event, rather than through a callback only a
        co-located UI could be handed. Mode checks and saved-rule matching already
        happened in the session runtime, so anything reaching here genuinely needs
        a person.
        """
        self._pending_approval = PendingApproval(
            request=request,
            rule_options=rule_options,
            request_id=request_id,
        )
        try:
            shown = self._show_approval_options(rule_options)
        except Exception:
            shown = False
        if not shown:
            self._pending_approval = None
            return False
        self._set_composer_status(messages.APPROVAL_PLACEHOLDER)
        self._set_chat_enabled(False)
        self._refresh_input_area_visibility()
        self._update_activity_progress(messages.WAITING_FOR_PERMISSION, state=TurnState.WAITING_FOR_USER)
        return True

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

    def _show_approval_options(self, rule_options: list[PermissionRuleOption]) -> bool:
        return self._set_approval_actions_visible(True)

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
            decision = PermissionDecision(allowed=True, reason="Allowed and remembered.", rule=rule)

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
        if request.kind.value == "message":
            recipient = str(request.inputs.get("recipient") or "").strip()
            text = str(request.inputs.get("text") or "").strip()
            body = "\n".join(["Allow the agent to send a peer message?", "", f"To: {recipient}"])
            if text:
                preview = text if len(text) <= 400 else text[:397] + "..."
                body += "\n\n" + preview
            return body
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

    def _set_approval_actions_visible(self, visible: bool) -> bool:
        try:
            panel = self.query_one("#approval_prompt", PromptPanel)
        except Exception:
            return False
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
        return True

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
