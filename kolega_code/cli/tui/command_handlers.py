"""Slash-command and command-adjacent behavior for the CLI TUI."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional


from kolega_code.agent import PromptExtension
from kolega_code.agent.prompt_dump import (
    dump_prompt_overrides,
    format_prompt_dump_result,
    format_prompt_list_result,
    format_prompt_validation_result,
    list_prompt_overrides,
    validate_prompt_overrides,
)
from kolega_code.agent.prompts import build_init_agents_prompt
from kolega_code.auth import constants as chatgpt_constants
from kolega_code.auth.chatgpt_oauth import run_login_flow
from kolega_code.llm.models import TextBlock
from kolega_code.permissions import normalize_permission_mode

from .. import messages, theme
from ..goal import (
    GOAL_CLEAR_ALIASES,
    GoalState,
    build_goal_task_prompt,
    format_goal_status,
)
from ..provider_registry import (
    UI_DEFAULT_PROVIDER,
    default_ui_thinking_effort,
    ui_model_options,
    ui_thinking_effort_options,
)
from ..skills import activated_skill_names
from ..slash_commands import SKILLS_LIST_COMMAND, TUI_COMMAND_NAMES, agent_command_names
from ..diagnostics import assemble_bug_bundle
from ..updater import check_for_update, current_version, run_self_update, update_status_message
from . import app_base as tui_app_base
from . import constants as tui_constants
from . import state as tui_state
from . import widgets as tui_widgets


class CommandHandlersMixin(tui_app_base.KolegaAppBase):
    def _tui_command_handlers(self) -> dict[str, Callable[[str], Awaitable[None]]]:
        return {
            "/attach": self._command_attach,
            "/detach": self._command_detach,
            "/init": self._command_init,
            "/plan": self._command_plan,
            "/build": self._command_build,
            "/sidebar": self._command_sidebar,
            "/permissions": self._command_permissions,
            "/model": self._command_model,
            "/effort": self._command_effort,
            "/lsp": self._command_lsp,
            "/login": self._command_login,
            "/logout": self._command_logout,
            "/gigacode": self._command_gigacode,
            "/goal": self._command_goal,
            "/prompts": self._command_prompts,
            "/queue-clear": self._command_queue_clear,
            "/theme": self._command_theme,
            "/copy": self._command_copy,
            "/diagnostics": self._command_diagnostics,
            "/bug": self._command_bug,
            "/version": self._command_version,
            "/update": self._command_update,
            "/quit": self._command_quit,
            "/exit": self._command_quit,
        }

    async def _handle_tui_slash_command(self, stripped_text: str, composer: tui_widgets.ChatComposer) -> bool:
        if not stripped_text.startswith("/"):
            return False
        command_text, _, args = stripped_text.partition(" ")
        handler = self._tui_command_handlers().get(command_text.lower())
        if handler is None:
            return False
        if command_text.lower() != "/model":
            self._cancel_pending_model_selection()
        if command_text.lower() != "/effort":
            self._cancel_pending_effort_selection()
        if command_text.lower() != "/theme":
            self._cancel_pending_theme_selection()
        composer.load_text("")
        await handler(args.strip())
        return True

    def _model_supports_vision(self) -> bool:
        """Safely check if the current agent's model supports vision input.

        Returns ``False`` when no agent is loaded (conservative: images won't
        work without a model, so a warning is the right default).
        """
        if self.agent is None:
            return False
        return getattr(self.agent, "supports_vision", False)

    def _add_vision_mismatch_system_message(self, *, context: str) -> None:
        """Add a persistent warning to the transcript when images meet a non-vision model.

        ``context`` is ``"attachment"`` (new image attached) or ``"model_switch"``
        (switched to a non-vision model with images in history). Deduplicated per
        model session so repeated attachments don't spam the transcript — the
        composer hint still updates with each attachment (showing all names), but
        the transcript system message appears only once.
        """
        if self._vision_warning_shown:
            return
        model_config = getattr(self.agent, "primary_model_config", None)
        model_name = getattr(model_config, "model", None) or "The current model"
        if context == "attachment":
            message = (
                f"⚠ {model_name} does not support vision. Use /detach to remove image "
                f"attachments or /model to switch to a vision-capable model."
            )
        else:  # model_switch
            message = messages.MODEL_NON_VISION_IMAGE_HISTORY
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=message, tone="warning"))
        self._vision_warning_shown = True

    def add_pending_image_attachment(self, attachment: dict) -> None:
        """Stash a pending image attachment for the next submitted message.

        Centralized vision check: when the current model does not support vision,
        shows a combined warning-tone hint (attachment confirmation + vision
        mismatch) and a persistent system message in the transcript. This is the
        single funnel for clipboard paste (both Ctrl+Shift+V and on_paste) and
        /attach, so the warning is consistent across all three paths.
        """
        self._pending_image_attachments.append(attachment)
        names = ", ".join(a.get("path", "image") for a in self._pending_image_attachments)
        if not self._model_supports_vision():
            self._show_composer_hint(
                f"Attached: {names}. {messages.MODEL_NON_VISION_IMAGE_ATTACHED}",
                tone="warning",
            )
            self._add_vision_mismatch_system_message(context="attachment")
        else:
            self._show_composer_hint(f"Attached images: {names} (press Enter to send, × to remove)", tone="info")

    async def _command_attach(self, args: str) -> None:
        arg = args.strip()
        if not arg:
            self._show_composer_hint("Usage: /attach <path-to-image>  (PNG, JPEG, GIF, WebP, BMP)", tone="warning")
            return
        from pathlib import Path as _Path

        from kolega_code.utils.images import encode_image_file

        candidate = _Path(arg)
        if not candidate.is_absolute():
            candidate = (self.project_path / arg).resolve()
        attachment = encode_image_file(candidate)
        if attachment is None:
            self._show_composer_hint(
                f"Could not attach {arg}: not a supported image, missing, or too large (>20MB). Use /attach <path>",
                tone="warning",
            )
            return
        attachment["path"] = arg
        self.add_pending_image_attachment(attachment)

    async def _command_detach(self, args: str) -> None:
        """Remove all pending image attachments (clears the attach queue).

        The user has no other way to discard a pending image once attached,
        especially on a non-vision model where the image can't be sent.
        """
        if not self._pending_image_attachments:
            self._show_composer_hint("No pending image attachments to remove.", tone="info")
            return
        names = ", ".join(a.get("path", "image") for a in self._pending_image_attachments)
        count = len(self._pending_image_attachments)
        self._pending_image_attachments.clear()
        self._show_composer_hint(f"Removed {count} image attachment(s): {names}", tone="info")

    async def _paste_clipboard_image_worker(self) -> None:
        from kolega_code.cli.clipboard_image import read_clipboard_image
        from kolega_code.utils.images import encode_image_attachment

        result = await read_clipboard_image()
        if result is None:
            self._show_composer_hint(
                "No image on the clipboard, or your terminal doesn't support image paste. "
                "Use /attach <path> or @image.png instead.",
                tone="warning",
            )
            return
        data, media_type = result
        attachment = encode_image_attachment(data, media_type, path="clipboard")
        # Vision check is centralized in add_pending_image_attachment.
        self.add_pending_image_attachment(attachment)

    async def _command_plan(self, args: str) -> None:
        if self._mode_switch_blocked():
            return
        await self._set_interaction_mode(tui_constants.PLAN_INTERACTION_MODE)

    async def _command_build(self, args: str) -> None:
        if self._mode_switch_blocked():
            return
        await self._set_interaction_mode(tui_constants.BUILD_INTERACTION_MODE)

    async def _command_gigacode(self, args: str) -> None:
        if self.agent is None:
            self._set_settings_status(messages.SETTINGS_REQUIRED, tone="warning")
            return

        clean = args.strip().lower()
        if clean in ("", "toggle"):
            new_state = not self._gigacode_enabled
        elif clean in ("on", "enable", "enabled", "true"):
            new_state = True
        elif clean in ("off", "disable", "disabled", "false"):
            new_state = False
        else:
            self._notify_user("Usage: /gigacode [on|off]", severity="warning")
            return

        self._gigacode_enabled = new_state
        self.agent.apply_gigacode(new_state, self._gigacode_prompt_extension() if new_state else None)
        await self._save_session_async()

        if new_state:
            note = (
                "gigacode workflow orchestration enabled — I can now author multi-agent "
                "workflows with the run_workflow tool for large fan-out tasks."
            )
            if self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE:
                note += " In plan mode, workflow sub-agents are read-only (parallel research only)."
        else:
            note = "gigacode workflow orchestration disabled."
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=note))
        self._update_mode_chrome()

    def _gigacode_prompt_extension(self) -> PromptExtension:
        from kolega_code.agent.orchestration.guide import GIGACODE_AUTHORING_GUIDE

        return PromptExtension(
            id="gigacode",
            title="gigacode — workflow orchestration",
            markdown=GIGACODE_AUTHORING_GUIDE,
            agent_types=None,
            modes=None,
            # Sub-agents can't run workflows (run_workflow is gated off for them),
            # so the authoring guide is just prompt bloat for a sub-agent.
            propagate_to_sub_agents=False,
        )

    async def _command_goal(self, args: str) -> None:
        clean = args.strip()
        first, _, rest = clean.partition(" ")
        first_lower = first.lower()

        # /goal clear | stop | off | reset | none | cancel
        if first_lower in GOAL_CLEAR_ALIASES:
            if self._turn_active or self.agent_worker is not None:
                self._show_composer_hint(messages.GOAL_BLOCK_STOP_FIRST)
                self._notify_user(messages.GOAL_BLOCK_STOP_FIRST, severity="warning")
                return
            if self._goal is None:
                self._add_conversation_entry(
                    tui_state.ConversationEntry(kind="system", content=messages.GOAL_NONE_ACTIVE)
                )
                return
            await self._clear_active_goal(note=messages.GOAL_CLEARED)
            return

        # /goal (no args) -> status
        if not clean:
            if self._goal is None:
                self._add_conversation_entry(
                    tui_state.ConversationEntry(kind="system", content=messages.GOAL_NONE_ACTIVE)
                )
            else:
                self._add_conversation_entry(
                    tui_state.ConversationEntry(kind="system", content=format_goal_status(self._goal))
                )
            return

        # Setting a goal requires a connected agent and an idle turn.
        if self.agent is None:
            self._set_settings_status(messages.GOAL_BLOCK_SETTINGS, tone="warning")
            return
        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.GOAL_BLOCK_STOP_FIRST)
            self._notify_user(messages.GOAL_BLOCK_STOP_FIRST, severity="warning")
            return

        # /goal -p <condition> | /goal --print <condition> -> run to completion
        run_to_completion = False
        condition_text = clean
        if first_lower in ("-p", "--print"):
            run_to_completion = True
            condition_text = rest.strip()
        if not condition_text:
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=messages.GOAL_USAGE))
            return

        replacing = self._goal is not None and self._goal.condition and not self._goal.met
        goal = GoalState.create(condition_text, run_to_completion=run_to_completion)
        self._set_goal_state(goal)
        await self._persist_goal_async()

        transcript = f"/goal {clean}"
        self._add_conversation_entry(tui_state.ConversationEntry(kind="user", content=transcript))
        if run_to_completion:
            self._add_conversation_entry(
                tui_state.ConversationEntry(kind="system", content=messages.GOAL_RUN_TO_COMPLETION)
            )
        self._add_conversation_entry(
            tui_state.ConversationEntry(
                kind="system",
                content=messages.GOAL_REPLACED if replacing else messages.GOAL_SET,
            )
        )

        # Kick off the first work turn; _process_message runs the goal loop after it.
        prompt = build_goal_task_prompt(condition_text)
        self.agent_worker = self.run_worker(
            self._process_message(prompt), name="kolega-turn", group="turns", exclusive=True
        )

    async def _command_prompts(self, args: str) -> None:
        clean_args = args.strip()
        parts = clean_args.split() if clean_args else []
        usage = "Usage: /prompts list | /prompts validate | /prompts dump [--force] [prompt ...]"
        if not parts:
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=usage))
            return

        command = parts[0].lower()
        if command == "list" and len(parts) == 1:
            result = list_prompt_overrides(self.project_path)
            self._add_conversation_entry(
                tui_state.ConversationEntry(kind="system", content=format_prompt_list_result(result))
            )
            return

        if command == "validate" and len(parts) == 1:
            context = self.agent.build_prompt_context() if self.agent is not None else None
            prompt_provider = getattr(self.agent, "prompt_provider", None) if self.agent is not None else None
            mode = getattr(self.agent, "agent_mode", None) if self.agent is not None else None
            project_template_slug = (
                getattr(self.agent, "project_template_slug", None) if self.agent is not None else None
            )
            result = validate_prompt_overrides(
                self.project_path,
                context=context,
                mode=mode,
                project_template_slug=project_template_slug,
                prompt_provider=prompt_provider,
            )
            self._add_conversation_entry(
                tui_state.ConversationEntry(kind="system", content=format_prompt_validation_result(result))
            )
            if not result.ok:
                self._notify_user("Some prompt override files are invalid.", severity="error")
            return

        if command == "dump":
            force = False
            selectors: list[str] = []
            for part in parts[1:]:
                if part == "--force":
                    force = True
                elif part.startswith("--"):
                    self._add_conversation_entry(
                        tui_state.ConversationEntry(kind="system", content=f"Unknown option: {part}\n\n{usage}")
                    )
                    return
                else:
                    selectors.append(part)

            if self._turn_active or self.agent_worker is not None:
                message = "Stop the current turn before dumping prompt overrides."
                self._show_composer_hint(message)
                self._notify_user(message, severity="warning")
                return
            base_context = self.agent.build_prompt_context() if self.agent is not None else None
            prompt_provider = getattr(self.agent, "prompt_provider", None) if self.agent is not None else None
            try:
                result = dump_prompt_overrides(
                    self.project_path,
                    force=force,
                    selectors=selectors,
                    base_context=base_context,
                    prompt_provider=prompt_provider,
                )
            except ValueError as exc:
                self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=str(exc)))
                self._notify_user(str(exc), severity="warning")
                return
            if self.agent is not None and result.written:
                self.agent.refresh_system_prompt()
            self._add_conversation_entry(
                tui_state.ConversationEntry(kind="system", content=format_prompt_dump_result(result))
            )
            if result.errors:
                self._notify_user("Some prompt override files could not be written.", severity="error")
            return

        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=usage))

    async def _command_sidebar(self, args: str) -> None:
        await self.action_toggle_sidebar()

    async def _command_init(self, args: str) -> None:
        if self._pending_question is not None:
            self._set_composer_status(messages.QUESTION_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_QUESTION_INIT, severity="warning")
            return

        if self._pending_approval is not None:
            self._set_composer_status(messages.APPROVAL_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_APPROVAL, severity="warning")
            return

        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_INIT)
            self._notify_user(messages.BLOCK_STOP_BEFORE_INIT, severity="warning")
            return

        if self._plan_decision_active:
            self._set_composer_status(messages.PLAN_READY_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PLAN_DECISION_INIT, severity="warning")
            return

        if self.agent is None:
            self._set_settings_status(messages.SETTINGS_REQUIRED, tone="warning")
            return

        if self.interaction_mode != tui_constants.BUILD_INTERACTION_MODE:
            await self._set_interaction_mode(tui_constants.BUILD_INTERACTION_MODE)

        if self.agent is None:
            self._set_settings_status(messages.SETTINGS_REQUIRED, tone="warning")
            return

        prompt = build_init_agents_prompt(args)
        transcript = "/init" if not args else f"/init {args}"
        self._add_conversation_entry(tui_state.ConversationEntry(kind="user", content=transcript))
        self.agent_worker = self.run_worker(
            self._process_message(prompt), name="kolega-turn", group="turns", exclusive=True
        )

    async def _command_permissions(self, args: str) -> None:
        if self._permission_mode_switch_blocked():
            return

        clean_args = args.strip().lower()
        if not clean_args:
            lines = [
                messages.PERMISSIONS_STATUS.format(mode=self.permission_mode.value),
                messages.PERMISSIONS_SWITCH_HINT,
            ]
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content="\n".join(lines)))
            return

        if clean_args == "toggle":
            await self.action_toggle_permission_mode()
            return

        try:
            mode = normalize_permission_mode(clean_args, default=self.permission_mode)
        except ValueError as exc:
            self._notify_user(str(exc), severity="warning")
            return

        await self._set_permission_mode(mode)

    async def _command_model(self, args: str) -> None:
        provider = self.settings.active_provider or UI_DEFAULT_PROVIDER
        model_options = ui_model_options(provider)
        if not args:
            if self._turn_active or self.agent_worker is not None:
                self._show_composer_hint(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH)
                self._notify_user(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH, severity="warning")
                return

            current_provider, current_model = self._startup_model()
            current_effort = self._startup_thinking_effort()
            active_model_line = (
                messages.SETTINGS_ACTIVE_MODEL.format(provider=current_provider, model=current_model)
                if current_model
                else messages.SETTINGS_ACTIVE_MODEL_UNCONFIGURED
            )
            lines = [
                active_model_line,
                messages.SETTINGS_THINKING_EFFORT_LINE.format(effort=current_effort or "not supported"),
                "",
                "Available models:",
                *(f"- `{value}` ({label})" for label, value in model_options),
                "",
                messages.MODEL_SWITCH_HINT,
            ]
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content="\n".join(lines)))
            self._pending_model_selection = tui_state.PendingModelSelection(provider=provider, options=model_options)
            self._cancel_pending_effort_selection()
            self._show_model_options()
            self._set_composer_status(messages.MODEL_PLACEHOLDER)
            return

        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH)
            self._notify_user(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH, severity="warning")
            return

        matched = self._match_model_value(model_options, args)
        if matched is None:
            self._notify_user(messages.MODEL_UNKNOWN.format(model=args, provider=provider), severity="warning")
            return

        await self._switch_model(provider, matched)

    async def _answer_model_option(self, option_index: int) -> None:
        pending = self._pending_model_selection
        if pending is None:
            return
        if option_index < 0 or option_index >= len(pending.options):
            return
        await self._switch_model(pending.provider, pending.options[option_index][1])

    async def _answer_model_selection(self, answer: str) -> None:
        pending = self._pending_model_selection
        if pending is None:
            return

        clean_answer = answer.strip()
        if not clean_answer:
            self._set_composer_status(messages.MODEL_PLACEHOLDER)
            return

        matched = self._match_model_value(pending.options, clean_answer)
        if matched is None:
            self._set_composer_status(messages.MODEL_PLACEHOLDER)
            self._notify_user(
                messages.MODEL_UNKNOWN.format(model=clean_answer, provider=pending.provider),
                severity="warning",
            )
            return

        await self._switch_model(pending.provider, matched)

    async def _switch_model(self, provider: str, model: str) -> None:
        self._cancel_pending_model_selection()
        self._cancel_pending_effort_selection()
        self.settings.active_provider = provider
        self.settings.active_model = model
        self.settings.active_thinking_effort = default_ui_thinking_effort(provider, model)
        self.settings_store.save(self.settings)
        await self._ensure_agent_from_settings(rebuild=True)
        try:
            self._populate_settings_controls()
        except Exception:
            pass
        self._restore_composer_placeholder()
        # Reset the per-session dedup flag so the new model gets a fresh warning.
        self._vision_warning_shown = False
        if self.agent is not None and not self._model_supports_vision():
            conversation = getattr(self.agent, "conversation", None)
            if conversation is not None and conversation.has_image_blocks():
                # Dual-channel: persistent transcript message + ephemeral composer hint.
                self._add_vision_mismatch_system_message(context="model_switch")
                self._show_composer_hint(messages.MODEL_NON_VISION_IMAGE_HISTORY, tone="warning")
        elif self.agent is not None and self._model_supports_vision():
            # Switching to a vision-capable model: clear any stale non-vision warning.
            self._clear_composer_hint()
        self._notify_user(
            messages.MODEL_SWITCHED.format(
                provider=provider,
                model=model,
                effort=self.settings.active_thinking_effort or "not supported",
            )
        )

    def _match_model_value(self, model_options: list[tuple[str, str]], value: str) -> Optional[str]:
        clean_value = value.strip().lower()
        return next((model for _, model in model_options if model.lower() == clean_value), None)

    # Providers the user can sign in to with /login <provider>. Add new targets
    # here as more OAuth integrations land.
    LOGIN_TARGETS: tuple[str, ...] = ("chatgpt",)

    async def _command_login(self, args: str) -> None:
        """Sign in to a provider: ``/login <provider>`` (e.g. ``/login chatgpt``)."""
        target = args.strip().lower()
        targets = ", ".join(self.LOGIN_TARGETS)
        if target == "chatgpt":
            await self._login_chatgpt()
        elif target in ("", "help"):
            self._add_conversation_entry(
                tui_state.ConversationEntry(kind="system", content=messages.LOGIN_USAGE.format(targets=targets))
            )
        else:
            self._notify_user(messages.LOGIN_UNKNOWN_TARGET.format(target=target, targets=targets), severity="warning")

    async def _login_chatgpt(self) -> None:
        """Start the browser "Sign in with ChatGPT" flow in a background worker.

        The flow can wait up to a few minutes for the browser round-trip, so it
        runs as a worker to keep the UI responsive.
        """
        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH)
            self._notify_user(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH, severity="warning")
            return
        self._add_conversation_entry(
            tui_state.ConversationEntry(kind="system", content=messages.CHATGPT_LOGIN_STARTING)
        )
        self.run_worker(self._do_chatgpt_login(), name="chatgpt-login", group="auth", exclusive=True)

    def _on_login_url(self, url: str) -> None:
        self._add_conversation_entry(
            tui_state.ConversationEntry(kind="system", content=messages.CHATGPT_LOGIN_URL.format(url=url))
        )

    async def _do_chatgpt_login(self) -> None:
        try:
            tokens = await run_login_flow(on_url=self._on_login_url)
        except Exception as exc:  # LoginError / TokenRefreshError / unexpected
            text = messages.CHATGPT_LOGIN_FAILED.format(error=exc)
            self._notify_user(text, severity="error")
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=text, tone="error"))
            return

        self.settings.set_oauth_token(chatgpt_constants.PROVIDER_KEY, tokens.model_dump(mode="json"))
        self.settings_store.save(self.settings)
        self._add_conversation_entry(
            tui_state.ConversationEntry(
                kind="system",
                content=messages.CHATGPT_LOGIN_SUCCESS.format(
                    email=tokens.email or "your ChatGPT account",
                    plan=tokens.plan_type or "subscription",
                ),
            )
        )
        # Switch to the ChatGPT provider so it's usable immediately. The stored
        # token is already saved above, so the agent rebuild inside _switch_model
        # picks it up.
        try:
            await self._switch_model(chatgpt_constants.PROVIDER_KEY, chatgpt_constants.DEFAULT_MODEL)
        except Exception as exc:
            self._notify_user(messages.CHATGPT_LOGIN_SWITCH_FAILED.format(error=exc), severity="warning")

    async def _command_logout(self, args: str) -> None:
        """Sign out of a provider: ``/logout <provider>`` (e.g. ``/logout chatgpt``)."""
        target = args.strip().lower()
        targets = ", ".join(self.LOGIN_TARGETS)
        if target == "chatgpt":
            self._logout_chatgpt()
        elif target in ("", "help"):
            self._add_conversation_entry(
                tui_state.ConversationEntry(kind="system", content=messages.LOGOUT_USAGE.format(targets=targets))
            )
        else:
            self._notify_user(messages.LOGOUT_UNKNOWN_TARGET.format(target=target, targets=targets), severity="warning")

    def _logout_chatgpt(self) -> None:
        if not self.settings.has_oauth_token(chatgpt_constants.PROVIDER_KEY):
            self._notify_user(messages.CHATGPT_LOGOUT_NONE, severity="warning")
            return
        self.settings.clear_oauth_token(chatgpt_constants.PROVIDER_KEY)
        self.settings_store.save(self.settings)
        self._notify_user(messages.CHATGPT_LOGOUT_DONE)

    async def _command_effort(self, args: str) -> None:
        provider, model = self._startup_model()
        effort_options = ui_thinking_effort_options(provider, model)
        current_effort = self._startup_thinking_effort()
        if not effort_options:
            self._notify_user(messages.EFFORT_UNSUPPORTED.format(provider=provider, model=model), severity="warning")
            return

        if not args:
            if self._turn_active or self.agent_worker is not None:
                self._show_composer_hint(messages.BLOCK_STOP_BEFORE_EFFORT_SWITCH)
                self._notify_user(messages.BLOCK_STOP_BEFORE_EFFORT_SWITCH, severity="warning")
                return

            active_model_line = (
                messages.SETTINGS_ACTIVE_MODEL.format(provider=provider, model=model)
                if model
                else messages.SETTINGS_ACTIVE_MODEL_UNCONFIGURED
            )
            lines = [
                active_model_line,
                messages.SETTINGS_THINKING_EFFORT_LINE.format(effort=current_effort or "not supported"),
                "",
                "Available thinking efforts:",
                *(f"- `{value}` ({label})" for label, value in effort_options),
                "",
                messages.EFFORT_SWITCH_HINT,
            ]
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content="\n".join(lines)))
            self._pending_effort_selection = tui_state.PendingEffortSelection(
                provider=provider,
                model=model,
                options=effort_options,
            )
            self._show_effort_options()
            self._set_composer_status(messages.EFFORT_PLACEHOLDER)
            return

        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_EFFORT_SWITCH)
            self._notify_user(messages.BLOCK_STOP_BEFORE_EFFORT_SWITCH, severity="warning")
            return

        matched = self._match_effort_value(effort_options, args)
        if matched is None:
            self._notify_user(
                messages.EFFORT_UNKNOWN.format(effort=args, provider=provider, model=model),
                severity="warning",
            )
            return

        self._cancel_pending_effort_selection()
        await self._switch_thinking_effort(provider, model, matched)

    async def _answer_effort_option(self, option_index: int) -> None:
        pending = self._pending_effort_selection
        if pending is None:
            return
        if option_index < 0 or option_index >= len(pending.options):
            return
        await self._switch_thinking_effort(pending.provider, pending.model, pending.options[option_index][1])

    async def _answer_effort_selection(self, answer: str) -> None:
        pending = self._pending_effort_selection
        if pending is None:
            return

        clean_answer = answer.strip()
        if not clean_answer:
            self._set_composer_status(messages.EFFORT_PLACEHOLDER)
            return

        matched = self._match_effort_value(pending.options, clean_answer)
        if matched is None:
            self._set_composer_status(messages.EFFORT_PLACEHOLDER)
            self._notify_user(
                messages.EFFORT_UNKNOWN.format(
                    effort=clean_answer,
                    provider=pending.provider,
                    model=pending.model,
                ),
                severity="warning",
            )
            return

        await self._switch_thinking_effort(pending.provider, pending.model, matched)

    async def _switch_thinking_effort(self, provider: str, model: str, effort: str) -> None:
        self._cancel_pending_effort_selection()
        self.settings.active_provider = provider
        self.settings.active_model = model
        self.settings.active_thinking_effort = effort
        self.settings_store.save(self.settings)
        await self._ensure_agent_from_settings(rebuild=True)
        try:
            self._populate_settings_controls()
        except Exception:
            pass
        self._restore_composer_placeholder()
        self._notify_user(messages.EFFORT_SWITCHED.format(effort=effort, provider=provider, model=model))

    def _match_effort_value(self, effort_options: list[tuple[str, str]], value: str) -> Optional[str]:
        clean_value = value.strip().lower()
        return next((effort for _, effort in effort_options if effort.lower() == clean_value), None)

    async def _command_theme(self, args: str) -> None:
        if not args:
            current = self.settings.active_theme or theme.DEFAULT_THEME_NAME
            lines = [
                messages.SETTINGS_ACTIVE_THEME.format(theme=current),
                "",
                "Available themes:",
                *(f"- `{name}`" for name in theme.available_themes()),
                "",
                messages.THEME_SWITCH_HINT,
            ]
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content="\n".join(lines)))
            self._pending_theme_selection = tui_state.PendingThemeSelection(
                options=[(name, name) for name in theme.available_themes()]
            )
            self._show_theme_options()
            self._set_composer_status(messages.THEME_PLACEHOLDER)
            return

        matched = self._match_theme_value(args)
        if matched is None:
            self._notify_user(messages.THEME_UNKNOWN.format(theme=args), severity="warning")
            return
        await self._switch_theme(matched)

    async def _answer_theme_option(self, option_index: int) -> None:
        pending = self._pending_theme_selection
        if pending is None:
            return
        if option_index < 0 or option_index >= len(pending.options):
            return
        await self._switch_theme(pending.options[option_index][1])

    async def _answer_theme_selection(self, answer: str) -> None:
        pending = self._pending_theme_selection
        if pending is None:
            return
        clean_answer = answer.strip()
        if not clean_answer:
            self._set_composer_status(messages.THEME_PLACEHOLDER)
            return
        matched = self._match_theme_value(clean_answer)
        if matched is None:
            self._set_composer_status(messages.THEME_PLACEHOLDER)
            self._notify_user(messages.THEME_UNKNOWN.format(theme=clean_answer), severity="warning")
            return
        await self._switch_theme(matched)

    async def _switch_theme(self, name: str) -> None:
        self._cancel_pending_theme_selection()
        self.settings.active_theme = name
        self.settings_store.save(self.settings)
        self._apply_theme(name)
        try:
            self._populate_settings_controls()
        except Exception:
            pass
        self._restore_composer_placeholder()
        self._notify_user(messages.THEME_SWITCHED.format(theme=name))

    def _match_theme_value(self, value: str) -> Optional[str]:
        clean_value = value.strip().lower()
        return next((name for name in theme.available_themes() if name.lower() == clean_value), None)

    def _apply_theme(self, name: Optional[str]) -> None:
        """Apply a theme live: swap Rich roles + Textual CSS, then re-skin the UI."""
        theme.apply_theme(name)
        try:
            self.theme = theme.textual_theme_name(name)
        except Exception:
            pass
        # Already-mounted Rich renderables baked in the old Color strings; rebuild
        # the conversation and dashboard so they pick up the new palette.
        self._render_conversation()
        self._refresh_status_dashboard()

    async def _command_copy(self, args: str) -> None:
        entry = next(
            (entry for entry in reversed(self.conversation_entries) if entry.kind == "assistant" and entry.content),
            None,
        )
        if entry is None:
            self._notify_user(messages.COPY_NOTHING, severity="warning")
            return
        self.copy_to_clipboard(entry.content)
        self._notify_user(messages.COPY_LAST_RESPONSE)

    def _diagnostics_summary_lines(self) -> list[str]:
        """Human-readable snapshot reused by /diagnostics and the /bug bundle."""
        header: dict = {}
        try:
            header = self._diagnostics_header()
        except Exception:
            pass
        lines = [
            f"Kolega Code {header.get('kolega_version') or current_version()}",
            f"Platform: {header.get('platform', '?')}  |  terminal: {header.get('term_program') or header.get('term') or '?'}",
            f"Python: {header.get('python', '?')}",
        ]
        if header.get("provider"):
            lines.append(f"Model: {header['provider']}/{header.get('model')} (effort: {header.get('thinking_effort')})")
        gigacode = "on" if header.get("gigacode_enabled") else "off"
        session_modes = (
            f"Permission: {header.get('permission_mode')}  |  "
            f"mode: {header.get('interaction_mode')}  |  gigacode: {gigacode}"
        )
        lines.append(session_modes)
        if header.get("providers_with_keys"):
            lines.append(f"Providers with keys: {', '.join(header['providers_with_keys'])}")
        diag = getattr(self, "_diag", None)
        if diag is not None and diag.enabled:
            lines.append(f"Diagnostics log: {diag.path}")
            try:
                text = diag.path.read_text(encoding="utf-8") if diag.path.exists() else ""
                stalls = text.count('"kind": "event_loop_stalled"')
                errors = text.count('"kind": "llm_error"')
                lines.append(f"This session: {stalls} loop stall(s), {errors} LLM error(s) recorded")
            except OSError:
                pass
        else:
            lines.append("Diagnostics: disabled (KOLEGA_CODE_NO_DIAGNOSTICS)")
        return lines

    async def _command_diagnostics(self, args: str) -> None:
        self._add_conversation_entry(
            tui_state.ConversationEntry(kind="system", content="\n".join(self._diagnostics_summary_lines()))
        )

    async def _command_bug(self, args: str) -> None:
        diag = getattr(self, "_diag", None)
        if diag is None or not diag.enabled:
            self._add_conversation_entry(
                tui_state.ConversationEntry(
                    kind="system", content="Diagnostics are disabled, so there is nothing to bundle."
                )
            )
            return
        summary = "\n".join(self._diagnostics_summary_lines())
        session_json = self.store.path_for(self.session.session_id)
        bundle = await asyncio.to_thread(assemble_bug_bundle, diag, summary=summary, session_json=session_json)
        if bundle is None:
            self._add_conversation_entry(
                tui_state.ConversationEntry(kind="system", content="Could not assemble the bug bundle.")
            )
            return
        try:
            self.copy_to_clipboard(str(bundle))
            copied = " (path copied to clipboard)"
        except Exception:
            copied = ""
        content = (
            f"Bug report written to:\n  {bundle}{copied}\n\n"
            "It contains this session's conversation and file contents (API keys are scrubbed) — "
            "review before posting publicly.\n"
            "Open an issue: https://github.com/kolega-ai/kolega-code/issues/new"
        )
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=content))

    async def _command_lsp(self, args: str) -> None:
        """Show LSP status: detected languages, servers, and install instructions."""
        msg = self._format_lsp_status()
        if msg:
            self._add_conversation_entry(tui_state.ConversationEntry(kind="lsp", content=msg))
        else:
            self._add_conversation_entry(
                tui_state.ConversationEntry(
                    kind="system",
                    content="LSP is not available (disabled or not configured).",
                )
            )

    async def _command_version(self, args: str) -> None:
        result = await asyncio.to_thread(check_for_update)
        lines = [messages.VERSION_INFO.format(version=result.current_version)]
        update_message = update_status_message(result, include_up_to_date=True, include_errors=True)
        if update_message:
            lines.append(update_message)
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content="\n".join(lines)))

    async def _command_update(self, args: str) -> None:
        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_UPDATE)
            self._notify_user(messages.BLOCK_STOP_BEFORE_UPDATE, severity="warning")
            return

        self._notify_user(messages.UPDATE_STARTED)
        result = await asyncio.to_thread(run_self_update, capture_output=True)
        severity = "information" if result.returncode == 0 else "error"
        if result.returncode == 0:
            lines = [messages.UPDATE_COMPLETED]
        else:
            lines = [messages.UPDATE_FAILED.format(code=result.returncode)]
            if result.error:
                lines.append(result.error)

        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        if output:
            if len(output) > 4000:
                output = "[output truncated]\n" + output[-4000:]
            lines.extend(["", "Output:", "```text", output, "```"])

        content = "\n".join(lines)
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=content))
        self._notify_user(lines[0], severity=severity)

    async def _command_queue_clear(self, args: str) -> None:
        count = self._clear_queued_messages()
        message = messages.QUEUE_CLEARED.format(count=count)
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=message))
        self._notify_user(message)

    async def _command_quit(self, args: str) -> None:
        await self.action_quit()

    async def _handle_skill_slash_command(self, stripped_text: str, composer: tui_widgets.ChatComposer) -> bool:
        command = self._parse_skill_slash_command(stripped_text)
        if command is None:
            return False

        command_name, prompt = command
        composer.load_text("")

        if command_name == "skills":
            self._add_conversation_entry(
                tui_state.ConversationEntry(kind="system", content=self.skill_catalog.format_catalog())
            )
            self._log_status(messages.SKILLS_LISTED, "ok")
            return True

        if self._pending_question is not None:
            self._set_composer_status(messages.QUESTION_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_QUESTION_SKILL, severity="warning")
            return True

        if self._pending_approval is not None:
            self._set_composer_status(messages.APPROVAL_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_APPROVAL, severity="warning")
            return True

        if self._plan_decision_active:
            self._set_composer_status(messages.PLAN_READY_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PLAN_DECISION_SKILL, severity="warning")
            return True

        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_SKILL)
            self._notify_user(messages.BLOCK_STOP_BEFORE_SKILL, severity="warning")
            return True

        if self.agent is None:
            self._set_settings_status(messages.SETTINGS_REQUIRED_SKILL, tone="warning")
            return True

        activated = self._activate_skill_in_agent(command_name)
        self._add_conversation_entry(tui_state.ConversationEntry(kind="skill", content=activated))
        self._notify_user(messages.SKILL_ACTIVATED.format(name=command_name))

        if prompt:
            attachments = self._build_mention_attachments(prompt)
            self._add_conversation_entry(tui_state.ConversationEntry(kind="user", content=prompt))
            self.agent_worker = self.run_worker(
                self._process_message(prompt, attachments), name="kolega-turn", group="turns", exclusive=True
            )
        else:
            await self._save_session_history_async()
            self._restore_composer_placeholder()
            self._set_chat_enabled(True)

        return True

    def _parse_skill_slash_command(self, stripped_text: str) -> Optional[tuple[str, str]]:
        if not stripped_text.startswith("/"):
            return None

        command_text, _, prompt = stripped_text.partition(" ")
        command = command_text.lower()
        if command == SKILLS_LIST_COMMAND:
            return "skills", prompt.strip()
        if command in agent_command_names() or command in TUI_COMMAND_NAMES:
            return None

        skill_name = command.removeprefix("/")
        if self.skill_catalog.get(skill_name) is None:
            return None

        return skill_name, prompt.strip()

    def _activate_skill_in_agent(self, skill_name: str) -> str:
        if self.agent is None:
            raise RuntimeError("Cannot activate a skill before an agent exists.")

        active_names = activated_skill_names(self.agent.history)
        content = self.skill_catalog.activation_content(skill_name, active_names=active_names)
        if skill_name not in active_names:
            self.agent.append_user_message([TextBlock(text=content)])
        return content
