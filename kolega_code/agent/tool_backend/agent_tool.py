import uuid
import inspect
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Union, Optional, cast
from datetime import datetime, timezone
import time

from kolega_code.config import AgentConfig
from kolega_code.events import AgentEvent
from kolega_code.hooks import HookDispatcher, HookEvent
from kolega_code.permissions import PermissionMode, auto_allow_permission_callback
from .base_tool import BaseTool


class AgentTool(BaseTool):
    """
    Unified tool for dispatching all types of sub-agents.

    This tool provides a consistent interface for creating and managing sub-agents
    with proper interrupt handling, error management, and cleanup.
    """

    def __init__(
        self,
        project_path: Union[str, Path],
        workspace_id: str,
        thread_id: str,
        connection_manager,
        config: AgentConfig,
        caller,
        filesystem=None,
        terminal_manager=None,
        browser_manager=None,
        langfuse_client=None,
        memory_manager=None,
    ):
        super().__init__(
            project_path,
            workspace_id,
            thread_id,
            connection_manager,
            config,
            caller,
            filesystem,
            terminal_manager=terminal_manager,
            browser_manager=browser_manager,
        )
        self.agents = {}
        self.langfuse_client = langfuse_client
        self.memory_manager = memory_manager
        self.sub_agent_recorder = getattr(caller, "sub_agent_recorder", None) if caller else None
        # No need to store these separately since they're already in the parent class
        # self.terminal_manager = terminal_manager
        # self.browser_manager = browser_manager

    async def _maybe_await(self, value):
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _sub_agent_extensions(extensions):
        """Filter a caller's prompt/tool extensions down to those that should be
        inherited by sub-agents.

        Interactive or session-shared host extensions (the task list, planning
        questions, the gigacode authoring guide) are marked
        ``propagate_to_sub_agents=False`` so that parallel sub-agents cannot
        clobber shared host state and so sub-agents aren't told to use tools they
        don't have.
        """
        if not extensions:
            return extensions
        return [ext for ext in extensions if getattr(ext, "propagate_to_sub_agents", True)]

    async def _call_recorder(self, method_name: str, *args, **kwargs):
        """Call an optional host-provided sub-agent recorder method."""
        if not self.sub_agent_recorder:
            return None

        method = getattr(self.sub_agent_recorder, method_name, None)
        if method is None:
            return None

        return await self._maybe_await(method(*args, **kwargs))

    async def _start_conversation(
        self,
        tool_call_id: str,
        agent_name: str,
        class_name: str,
        agent_id: str,
        task: str,
    ) -> Optional[str]:
        payload = {
            "parent_thread_id": self.thread_id,
            "parent_tool_call_id": tool_call_id,
            "agent_name": agent_name,
            "agent_type": class_name,
            "agent_id": agent_id,
            "initial_task": task,
        }
        return await self._call_recorder("start_conversation", payload)

    async def _record_message(self, conversation_id: str, message: dict, sequence: int) -> None:
        await self._call_recorder("record_message", conversation_id, message, sequence)

    async def _complete_conversation(self, conversation_id: str, update_data: dict) -> None:
        await self._call_recorder("complete_conversation", conversation_id, update_data)

    async def _fail_conversation(self, conversation_id: str, update_data: dict) -> None:
        await self._call_recorder("fail_conversation", conversation_id, update_data)

    async def _interrupt_conversation(self, conversation_id: str, update_data: dict) -> None:
        await self._call_recorder("interrupt_conversation", conversation_id, update_data)

    async def _send_status_event(
        self,
        status: str,
        message: str,
        sub_agent_info: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Helper method to send status events."""
        content = {"status": status, "message": message}
        if extra:
            content.update(extra)
        event = AgentEvent(
            event_type="chat_message",
            content=content,
            sender=self.caller.agent_name if self.caller else "agent-tool",
            sub_agent_info=sub_agent_info,
        )
        await self.connection_manager.broadcast_event(event, self.workspace_id, self.thread_id)

    async def _apply_subagent_stop_hooks(self, agent_name: str, result: str, sub_agent_info: Optional[dict]) -> str:
        """Fire SubagentStop on the parent agent and fold any augmentation into the result."""
        dispatcher = getattr(self.caller, "hook_dispatcher", None)
        # ``fire_hook`` is an async method on the parent agent; ``getattr`` returns a
        # broad type, so cast to a concrete async-callable signature before awaiting.
        fire = cast(Callable[..., Awaitable[Any]], getattr(self.caller, "fire_hook", None))
        if not isinstance(dispatcher, HookDispatcher) or not dispatcher.is_active or not callable(fire):
            return result
        outcome = await fire(
            HookEvent.SUBAGENT_STOP,
            {"agent_name": agent_name, "result": result, "sub_agent_info": sub_agent_info},
            target=agent_name,
        )
        if outcome.additional_context:
            result = f"{result}\n\n{outcome.additional_context}"
        if outcome.blocked and outcome.reason:
            result = f"{result}\n\n[hook] {outcome.reason}"
        return result

    async def _dispatch_agent(
        self,
        agent_class_import: str,
        task: str,
        *,
        agent_name_override: Optional[str] = None,
        agent_kwargs: Optional[dict[str, Any]] = None,
        sub_agent_info_extra: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Generic method to dispatch any agent type.

        Args:
            agent_class_import: Full import path to agent class (e.g., "kolega_code.agent.investigationagent.InvestigationAgent")
            task: Task description for the agent

        Returns:
            The agent's recap of its work
        """
        # Extract the agent name from the class
        module_path, class_name = agent_class_import.rsplit(".", 1)

        # Import the module and get the class
        module = __import__(module_path, fromlist=[class_name])
        agent_class = getattr(module, class_name)
        agent_name = agent_name_override or agent_class.agent_name

        # Create a unique agent ID
        agent_id = str(uuid.uuid4())

        # Use the app's unique execution ID for DB/UI links, not the provider's tool-use ID.
        tool_call_id = getattr(self.caller, "current_tool_execution_id", None)
        if not isinstance(tool_call_id, str):
            tool_call_id = getattr(self.caller, "current_tool_call_id", None)
        if not isinstance(tool_call_id, str):
            tool_call_id = None
        conversation_id = None
        start_time = time.time()

        # Create sub-agent conversation record if the host application supplied a recorder.
        if tool_call_id:
            conversation_id = await self._start_conversation(
                tool_call_id=tool_call_id,
                agent_name=agent_name,
                class_name=class_name,
                agent_id=agent_id,
                task=task,
            )

        parent_context = getattr(self.caller, "sub_agent_context", None)
        if not isinstance(parent_context, dict):
            parent_context = {}
        context_depth = parent_context.get("depth")
        if isinstance(context_depth, int) and not isinstance(context_depth, bool):
            parent_depth = context_depth
        else:
            parent_depth = 1 if getattr(self.caller, "sub_agent", False) else 0

        # Attached to every event from this dispatch so the UI can group and
        # disambiguate concurrently running sub-agents.
        sub_agent_info = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "task": task[:120],
            "task_full": task,
            "conversation_id": conversation_id,
            "parent_tool_call_id": tool_call_id,
            "depth": parent_depth + 1,
        }
        # A nested dispatch from a workflow worker inherits the workflow's
        # delegation policy and grouping metadata. Cosmetic call labels/indexes
        # belong only to the direct workflow call and are deliberately not copied.
        workflow_run_id = parent_context.get("workflow_run_id")
        max_agent_depth = parent_context.get("max_agent_depth")
        if isinstance(workflow_run_id, str) and isinstance(max_agent_depth, int):
            sub_agent_info.update(
                workflow_run_id=workflow_run_id,
                max_agent_depth=max_agent_depth,
                phase=parent_context.get("phase"),
                parent_agent_id=parent_context.get("agent_id"),
            )
        if sub_agent_info_extra:
            sub_agent_info.update(sub_agent_info_extra)

        # Send start status
        await self._send_status_event("GENERATING", f"Starting {agent_name} task", sub_agent_info=sub_agent_info)
        conversation_finished = False

        try:
            # Create the agent instance
            agent = agent_class(
                project_path=self.project_path,
                workspace_id=self.workspace_id,
                thread_id=self.thread_id,
                connection_manager=self.connection_manager,
                config=self.config,
                sub_agent=True,
                filesystem=self.filesystem,
                terminal_manager=self.terminal_manager,
                browser_manager=self.browser_manager,
                langfuse_client=self.langfuse_client,
                user_id=getattr(self.caller, "user_id", None) if self.caller else None,
                user_email=getattr(self.caller, "user_email", None) if self.caller else None,
                project_template_slug=getattr(self.caller, "project_template_slug", None) if self.caller else None,
                protected_files=getattr(self.caller, "protected_files", None) if self.caller else None,
                agent_mode=getattr(self.caller, "agent_mode", None) if self.caller else None,
                workspace_env_var_descriptions=getattr(self.caller, "workspace_env_var_descriptions", None)
                if self.caller
                else None,
                workspace_memories=getattr(self.caller, "workspace_memories", None) if self.caller else None,
                # Share the parent's PromptProvider so each sub-agent doesn't build its
                # own Jinja2 Environment + loaders + template cache over the same bundled
                # templates. Environments are designed to be shared; per-agent prompt
                # extensions are passed as render args, not stored on the provider.
                prompt_provider=getattr(self.caller, "prompt_provider", None) if self.caller else None,
                prompt_extensions=self._sub_agent_extensions(getattr(self.caller, "prompt_extensions", None)),
                tool_extensions=self._sub_agent_extensions(getattr(self.caller, "tool_extensions", None)),
                permission_mode=getattr(self.caller, "permission_mode", None) if self.caller else None,
                permission_callback=getattr(self.caller, "permission_callback", None) if self.caller else None,
                usage_recorder=getattr(self.caller, "usage_recorder", None) if self.caller else None,
                sub_agent_recorder=getattr(self.caller, "sub_agent_recorder", None) if self.caller else None,
                hook_dispatcher=getattr(self.caller, "hook_dispatcher", None) if self.caller else None,
                max_iterations=getattr(self.caller, "max_iterations", None),
                memory_manager=self.memory_manager,
                **(agent_kwargs or {}),
            )

            # Store agent reference
            self.agents[agent_id] = agent

            # Set parent context so the agent's own events carry sub_agent_info
            agent.parent_tool_call_id = tool_call_id
            agent.conversation_id = conversation_id
            agent.sub_agent_context = sub_agent_info

            # Track messages and their sequence
            last_saved_index = -1  # Track what we've already saved
            streamed_messages = {}  # Track messages by UUID for assembly

            # Process the task and stream messages
            async for msg in agent.process_message_stream(task):
                # Extract message details
                message_type = msg.get("type", "agent")
                content = msg.get("content", "")
                complete = msg.get("complete", False)
                msg_uuid = msg.get("uuid", str(uuid.uuid4()))
                timestamp = datetime.now().isoformat()

                content_payload = {"text": content}
                if message_type != "response":
                    content_payload["message_type"] = message_type

                evt = AgentEvent(
                    event_type="chat_message",
                    content=content_payload,
                    sender=agent_name,
                    timestamp=timestamp,
                    is_streaming=(message_type in ["response", "thinking"] and not complete),
                    uuid=msg_uuid,
                    sub_agent_info=sub_agent_info,
                )

                # Broadcast to connection manager
                await self.connection_manager.broadcast_event(evt, self.workspace_id, self.thread_id)

                # Track streaming messages
                if msg_uuid not in streamed_messages:
                    streamed_messages[msg_uuid] = {
                        "content": "",
                        "type": message_type,
                        "uuid": msg_uuid,
                        "complete": False,
                    }

                streamed_messages[msg_uuid]["content"] += content
                streamed_messages[msg_uuid]["complete"] = complete

                # Save new messages when a message completes
                if conversation_id and complete:
                    # Get current complete history from agent
                    current_history = agent.dump_message_history()

                    # Only save messages we haven't saved yet
                    for i in range(last_saved_index + 1, len(current_history)):
                        hist_msg = current_history[i]
                        await self._record_message(
                            conversation_id,
                            {
                                "role": hist_msg.get("role", "assistant"),
                                "content": hist_msg.get("content", []),
                                "stream_uuid": None,
                            },
                            i + 1,
                        )

                    # Update last saved index
                    last_saved_index = len(current_history) - 1

                    # Mark streamed message as saved
                    streamed_messages[msg_uuid]["saved"] = True

            # Get final history and save any remaining messages
            final_history = agent.dump_message_history()

            if conversation_id:
                # Save any messages we haven't saved yet
                for i in range(last_saved_index + 1, len(final_history)):
                    hist_msg = final_history[i]
                    await self._record_message(
                        conversation_id,
                        {
                            "role": hist_msg.get("role", "assistant"),
                            "content": hist_msg.get("content", []),
                            "stream_uuid": None,
                        },
                        i + 1,
                    )

            # Get agent recap
            result = await agent.recap_agent_outcome()

            # SubagentStop hooks: let the parent observe completion and augment the
            # result the parent sees. (Forcing the sub-agent to resume on a blocked
            # SubagentStop is deferred; the per-agent Stop hook covers keep-working.)
            result = await self._apply_subagent_stop_hooks(agent_name, result, sub_agent_info)

            # Update conversation with completion status
            if conversation_id:
                execution_time = time.time() - start_time

                update_data = {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc),
                    "recap": result,
                    "message_count": len(final_history),
                    "execution_time_seconds": execution_time,
                }

                # Try to get token count if available
                if hasattr(agent, "total_tokens_used"):
                    update_data["total_tokens"] = agent.total_tokens_used

                await self._complete_conversation(conversation_id, update_data)
                conversation_finished = True

            # Send completion status, including the sub-agent's token total when available
            # so the host UI can show per-agent usage on the finished card/roster.
            total_tokens = getattr(agent, "total_tokens_used", None)
            extra = {"total_tokens": total_tokens} if isinstance(total_tokens, int) else None
            await self._send_status_event(
                "STOPPED", f"Completed {agent_name} task", sub_agent_info=sub_agent_info, extra=extra
            )

            return result

        except Exception as e:
            # Update conversation with error status
            if conversation_id:
                execution_time = time.time() - start_time
                await self._fail_conversation(
                    conversation_id,
                    {
                        "status": "failed",
                        "completed_at": datetime.now(timezone.utc),
                        "error": str(e),
                        "execution_time_seconds": execution_time,
                    },
                )
                conversation_finished = True

            # Log and re-raise the error
            await self.log_error(f"Error in {agent_name}: {str(e)}", sender="AgentTool")
            await self._send_status_event("ERROR", f"Error in {agent_name}: {str(e)}", sub_agent_info=sub_agent_info)
            raise

        finally:
            # Handle interrupted conversations
            if conversation_id and not conversation_finished:
                execution_time = time.time() - start_time
                await self._interrupt_conversation(
                    conversation_id,
                    {
                        "status": "interrupted",
                        "completed_at": datetime.now(timezone.utc),
                        "execution_time_seconds": execution_time,
                    },
                )

            # Clean up agent reference
            if agent_id in self.agents:
                del self.agents[agent_id]

    async def dispatch_investigation_agent(self, task: str) -> str:
        """
        Dispatch an investigation agent to perform a specific task with read-only access to the codebase.

        Args:
            task: A detailed description of the investigation task to perform

        Returns:
            A comprehensive report of the investigation findings
        """
        return await self._dispatch_agent(
            agent_class_import="kolega_code.agent.investigationagent.InvestigationAgent",
            task=task,
        )

    async def dispatch_browser_agent(self, task: str) -> str:
        """
        Dispatch a browser agent to perform web-based tasks and interactions.

        Args:
            task: A detailed description of the browser task to perform

        Returns:
            A comprehensive report of the browser agent's findings and actions
        """
        return await self._dispatch_agent(
            agent_class_import="kolega_code.agent.browseragent.BrowserAgent",
            task=task,
        )

    async def dispatch_coding_agent(self, task: str) -> str:
        """
        Dispatch a coding agent for processing coding-related tasks with streaming output.

        Args:
            task: A detailed description of the coding task to perform

        Returns:
            A summary of the coding process outcome
        """
        return await self._dispatch_agent(
            agent_class_import="kolega_code.agent.coder.CoderAgent",
            task=task,
        )

    async def dispatch_general_agent(self, task: str) -> str:
        """
        Dispatch a general-purpose agent to autonomously complete a self-contained task.

        Args:
            task: A detailed, self-contained description of the task to perform

        Returns:
            The agent's final report on the completed task
        """
        return await self._dispatch_agent(
            agent_class_import="kolega_code.agent.generalagent.GeneralAgent",
            task=task,
        )

    def _eligible_custom_agent_tools(self) -> set[str]:
        """Return the caller's propagatable, non-recursive tool surface."""
        collection = getattr(self.caller, "tool_collection", None)
        if collection is None:
            raise RuntimeError("Custom agents require an initialized parent tool collection.")

        eligible = set(collection.registry().names())
        from ..tools import ToolCollection  # lazy import: tools.py imports AgentTool

        eligible.difference_update(ToolCollection.agent_dispatch_tools)
        eligible.difference_update(ToolCollection.orchestration_tools)

        for extension in getattr(self.caller, "tool_extensions", None) or []:
            if not getattr(extension, "propagate_to_sub_agents", True):
                eligible.difference_update(getattr(extension, "tools", {}).keys())
        return eligible

    async def dispatch_custom_agent(self, agent: str, task: str) -> str:
        """Dispatch a discovered custom agent within the caller's capability ceiling."""
        if getattr(self.caller, "sub_agent", False):
            raise ValueError("Custom agents cannot be dispatched from another sub-agent.")

        catalog = getattr(self.caller, "custom_agent_catalog", None)
        if catalog is None or not catalog.has_agents():
            raise ValueError("No custom agents are available in this session.")
        definition = catalog.get(agent)
        if definition is None:
            available = ", ".join(catalog.names()) or "none"
            raise ValueError(f"Unknown custom agent `{agent}`. Available agents: {available}.")

        eligible = self._eligible_custom_agent_tools()
        if definition.tools is None:
            allowed_tools = eligible
        else:
            requested = set(definition.tools)
            unavailable = sorted(requested - eligible)
            if unavailable:
                raise ValueError(
                    f"Custom agent `{definition.name}` requests unavailable tool(s): {', '.join(unavailable)}. "
                    "A custom agent may only narrow the invoking agent's tool set."
                )
            allowed_tools = requested

        return await self._dispatch_agent(
            agent_class_import="kolega_code.agent.custom_agents.CustomAgent",
            task=task,
            agent_name_override=definition.name,
            agent_kwargs={
                "definition": definition,
                "allowed_tools": sorted(allowed_tools),
            },
            sub_agent_info_extra={
                "agent_scope": definition.scope,
                "agent_definition_path": str(definition.source_path),
            },
        )

    # ------------------------------------------------------------------
    # gigacode workflow dispatch
    #
    # dispatch_workflow_agent mirrors _dispatch_agent's lifecycle (conversation
    # recording, event streaming, SubagentStop hooks) but adds three things the
    # orchestration runtime needs: a per-call config override (model/effort), an
    # optional forced-structured-output tool, and extra sub_agent_info keys so the
    # UI can group agents under their workflow run and phase. It reuses the same
    # helper methods as _dispatch_agent; only construction and the return shape differ.
    # ------------------------------------------------------------------
    _STRUCTURED_OUTPUT_INSTRUCTION = (
        "\n\nWhen you have finished, you MUST report your result by calling the "
        "`submit_result` tool with arguments matching the requested schema. Do not "
        "answer in prose — the `submit_result` call is the only output that is read."
    )
    _STRUCTURED_OUTPUT_NUDGE = (
        "You have not called `submit_result` yet. Call it now with your result, matching the requested schema exactly."
    )

    def _structured_output_extension(self, schema: dict, capture: dict):
        """Build a ToolExtension exposing a `submit_result` tool whose input schema
        is the caller-requested JSON Schema. The handler stashes the validated
        payload into ``capture['value']``.
        """
        from ..tools import ToolExtension  # lazy import: tools.py imports this module

        wrapped = not (isinstance(schema, dict) and schema.get("type") == "object")
        if wrapped:
            # Tool inputs are always a top-level object; wrap non-object schemas.
            input_schema = {
                "type": "object",
                "properties": {"result": schema},
                "required": ["result"],
            }
        else:
            input_schema = schema

        async def submit_result(**kwargs):
            capture["value"] = kwargs.get("result") if wrapped else kwargs
            return "Result recorded."

        return ToolExtension(
            name="workflow_structured_output",
            tools={"submit_result": submit_result},
            tool_schemas={"submit_result": input_schema},
            # submit_result only reports a result (no side effects), so mark it
            # read-only-safe; otherwise read-only sub-agents would have it filtered
            # out and could never return structured output.
            tool_groups={"read_only_tools": ["submit_result"]},
        )

    def _construct_workflow_sub_agent(self, agent_class, config, extra_tool_extensions):
        """Construct a sub-agent the same way _dispatch_agent does, but allowing a
        config override and additional tool extensions.
        """
        tool_extensions = self._sub_agent_extensions(getattr(self.caller, "tool_extensions", None))
        if extra_tool_extensions:
            tool_extensions = list(tool_extensions or []) + list(extra_tool_extensions)
        return agent_class(
            project_path=self.project_path,
            workspace_id=self.workspace_id,
            thread_id=self.thread_id,
            connection_manager=self.connection_manager,
            config=config or self.config,
            sub_agent=True,
            filesystem=self.filesystem,
            terminal_manager=self.terminal_manager,
            browser_manager=self.browser_manager,
            langfuse_client=self.langfuse_client,
            user_id=getattr(self.caller, "user_id", None) if self.caller else None,
            user_email=getattr(self.caller, "user_email", None) if self.caller else None,
            project_template_slug=getattr(self.caller, "project_template_slug", None) if self.caller else None,
            protected_files=getattr(self.caller, "protected_files", None) if self.caller else None,
            agent_mode=getattr(self.caller, "agent_mode", None) if self.caller else None,
            workspace_env_var_descriptions=getattr(self.caller, "workspace_env_var_descriptions", None)
            if self.caller
            else None,
            workspace_memories=getattr(self.caller, "workspace_memories", None) if self.caller else None,
            # Share the parent's PromptProvider (see _dispatch_agent) to avoid a
            # per-sub-agent Jinja2 Environment over the same bundled templates.
            prompt_provider=getattr(self.caller, "prompt_provider", None) if self.caller else None,
            prompt_extensions=self._sub_agent_extensions(getattr(self.caller, "prompt_extensions", None)),
            tool_extensions=tool_extensions,
            # Workflow sub-agents run unattended in parallel, so they default to AUTO
            # permission mode (no prompts) regardless of the session's mode — per-agent
            # prompting would be unworkable across a fan-out.
            permission_mode=PermissionMode.AUTO,
            permission_callback=auto_allow_permission_callback,
            usage_recorder=getattr(self.caller, "usage_recorder", None) if self.caller else None,
            sub_agent_recorder=getattr(self.caller, "sub_agent_recorder", None) if self.caller else None,
            hook_dispatcher=getattr(self.caller, "hook_dispatcher", None) if self.caller else None,
            max_iterations=getattr(self.caller, "max_iterations", None),
        )

    @staticmethod
    def _write_jsonl_message(path: Optional[Path], message: dict) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message, default=str) + "\n")

    @staticmethod
    def _render_workflow_message(message: dict) -> str:
        role = str(message.get("role") or "message")
        content = message.get("content", "")
        lines = [f"### {role}", ""]
        if isinstance(content, str):
            lines.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "content")
                    if item_type == "text":
                        lines.append(str(item.get("text", "")))
                    elif item_type == "tool_use":
                        lines.append(f"**Tool use:** `{item.get('name', '')}`")
                        lines.append("```json")
                        lines.append(json.dumps(item.get("input", {}), indent=2, default=str))
                        lines.append("```")
                    elif item_type == "tool_result":
                        status = "Error" if item.get("is_error") else "Result"
                        lines.append(f"**Tool {status}:** `{item.get('name', '')}`")
                        lines.append("```")
                        lines.append(str(item.get("content", "")))
                        lines.append("```")
                    else:
                        lines.append("```json")
                        lines.append(json.dumps(item, indent=2, default=str))
                        lines.append("```")
                else:
                    rendered = item.to_markdown() if hasattr(item, "to_markdown") else str(item)
                    lines.append(rendered)
        else:
            lines.append(str(content))
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _render_value_markdown(value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, str):
            return value
        try:
            return "```json\n" + json.dumps(value, indent=2, default=str) + "\n```"
        except (TypeError, ValueError):
            return str(value)

    def _write_workflow_agent_markdown(
        self,
        artifact_paths: Optional[dict],
        *,
        metadata: dict,
        prompt: str,
        status: str,
        result: Optional[str] = None,
        structured: Any = None,
        tokens: int = 0,
        error: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> None:
        if not artifact_paths or not artifact_paths.get("markdown"):
            return
        path = Path(artifact_paths["markdown"])
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Workflow agent {metadata.get('call_index', '')}: {metadata.get('label') or metadata.get('agent_name') or 'agent'}",
            "",
            "## Metadata",
            "",
            f"- Call index: {metadata.get('call_index', '')}",
            f"- Label: {metadata.get('label') or ''}",
            f"- Phase: {metadata.get('phase') or ''}",
            f"- Agent type: {metadata.get('agent_type') or metadata.get('agent_name') or ''}",
            f"- Max agent depth: {metadata.get('max_agent_depth', '')}",
            f"- Status: {status}",
            f"- Tokens: {tokens}",
        ]
        if error:
            lines.append(f"- Error: {error}")
        jsonl_path = artifact_paths.get("jsonl")
        if jsonl_path:
            lines.append(f"- Raw transcript: `{jsonl_path}`")
        lines.extend(["", "## Task prompt", "", prompt.strip() or "(empty)"])
        if structured is not None:
            lines.extend(["", "## Structured result", "", self._render_value_markdown(structured)])
        if result is not None:
            lines.extend(["", "## Final recap", "", result])
        if history:
            lines.extend(["", "## Message history", ""])
            for message in history:
                lines.append(self._render_workflow_message(message))
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    async def dispatch_workflow_agent(
        self,
        agent_class,
        task: str,
        *,
        config=None,
        schema: Optional[dict] = None,
        sub_agent_info_extra: Optional[dict] = None,
        artifact_paths: Optional[dict] = None,
        artifact_metadata: Optional[dict] = None,
    ):
        """Dispatch one sub-agent for a gigacode workflow.

        Returns a tuple of ``(recap_text, total_tokens, structured)`` where
        ``structured`` is the validated ``submit_result`` payload (or ``None`` when
        no schema was requested or the agent declined to call it).
        """
        agent_name = getattr(agent_class, "agent_name", agent_class.__name__)
        agent_id = str(uuid.uuid4())

        tool_call_id = getattr(self.caller, "current_tool_execution_id", None)
        if not isinstance(tool_call_id, str):
            tool_call_id = getattr(self.caller, "current_tool_call_id", None)
        if not isinstance(tool_call_id, str):
            tool_call_id = None

        conversation_id = None
        start_time = time.time()
        if tool_call_id:
            conversation_id = await self._start_conversation(
                tool_call_id=tool_call_id,
                agent_name=agent_name,
                class_name=agent_class.__name__,
                agent_id=agent_id,
                task=task,
            )

        parent_depth = 1 if getattr(self.caller, "sub_agent", False) else 0
        sub_agent_info = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "task": task[:120],
            "task_full": task,
            "conversation_id": conversation_id,
            "parent_tool_call_id": tool_call_id,
            "depth": parent_depth + 1,
        }
        if sub_agent_info_extra:
            sub_agent_info.update({k: v for k, v in sub_agent_info_extra.items() if v is not None})

        artifact_metadata = dict(artifact_metadata or {})
        artifact_metadata.setdefault("agent_name", agent_name)
        if artifact_paths:
            for key in ("markdown", "jsonl"):
                if artifact_paths.get(key):
                    artifact_paths[key] = Path(artifact_paths[key])
            if artifact_paths.get("jsonl"):
                artifact_paths["jsonl"].parent.mkdir(parents=True, exist_ok=True)
                artifact_paths["jsonl"].write_text("", encoding="utf-8")

        capture: dict = {}
        extra_extensions = []
        effective_task = task
        if schema:
            extra_extensions.append(self._structured_output_extension(schema, capture))
            effective_task = task + self._STRUCTURED_OUTPUT_INSTRUCTION

        await self._send_status_event("GENERATING", f"Starting {agent_name} task", sub_agent_info=sub_agent_info)
        conversation_finished = False
        agent = None

        try:
            agent = self._construct_workflow_sub_agent(agent_class, config, extra_extensions)
            self.agents[agent_id] = agent
            agent.parent_tool_call_id = tool_call_id
            agent.conversation_id = conversation_id
            agent.sub_agent_context = sub_agent_info

            last_saved_index = await self._stream_workflow_agent(
                agent,
                agent_name,
                sub_agent_info,
                conversation_id,
                last_saved_index=-1,
                task=effective_task,
                artifact_jsonl_path=artifact_paths.get("jsonl") if artifact_paths else None,
            )

            # Single re-prompt if a schema was requested but submit_result wasn't called.
            if schema and "value" not in capture:
                last_saved_index = await self._stream_workflow_agent(
                    agent,
                    agent_name,
                    sub_agent_info,
                    conversation_id,
                    last_saved_index=last_saved_index,
                    task=self._STRUCTURED_OUTPUT_NUDGE,
                    artifact_jsonl_path=artifact_paths.get("jsonl") if artifact_paths else None,
                )

            result = await agent.recap_agent_outcome()
            result = await self._apply_subagent_stop_hooks(agent_name, result, sub_agent_info)

            total_tokens = getattr(agent, "total_tokens_used", None)
            if conversation_id:
                await self._complete_conversation(
                    conversation_id,
                    {
                        "status": "completed",
                        "completed_at": datetime.now(timezone.utc),
                        "recap": result,
                        "execution_time_seconds": time.time() - start_time,
                        "total_tokens": total_tokens if isinstance(total_tokens, int) else None,
                    },
                )
                conversation_finished = True

            extra = {"total_tokens": total_tokens} if isinstance(total_tokens, int) else None
            await self._send_status_event(
                "STOPPED", f"Completed {agent_name} task", sub_agent_info=sub_agent_info, extra=extra
            )

            structured = capture.get("value") if schema else None
            final_history = agent.dump_message_history() if agent is not None else []
            if artifact_paths and artifact_paths.get("jsonl"):
                # Persist any history messages that completed after the last stream event.
                for i in range(last_saved_index + 1, len(final_history)):
                    self._write_jsonl_message(artifact_paths.get("jsonl"), final_history[i])
            self._write_workflow_agent_markdown(
                artifact_paths,
                metadata=artifact_metadata,
                prompt=task,
                status="completed",
                result=result,
                structured=structured,
                tokens=total_tokens if isinstance(total_tokens, int) else 0,
                history=final_history,
            )
            return result, (total_tokens if isinstance(total_tokens, int) else 0), structured

        except Exception as e:
            if conversation_id:
                await self._fail_conversation(
                    conversation_id,
                    {
                        "status": "failed",
                        "completed_at": datetime.now(timezone.utc),
                        "error": str(e),
                        "execution_time_seconds": time.time() - start_time,
                    },
                )
                conversation_finished = True
            final_history = []
            if agent is not None:
                try:
                    final_history = agent.dump_message_history()
                except Exception:  # noqa: BLE001 - transcript capture must not mask the real failure
                    final_history = []
            self._write_workflow_agent_markdown(
                artifact_paths,
                metadata=artifact_metadata,
                prompt=task,
                status="failed",
                error=str(e),
                history=final_history,
            )
            await self.log_error(f"Error in workflow {agent_name}: {str(e)}", sender="AgentTool")
            await self._send_status_event("ERROR", f"Error in {agent_name}: {str(e)}", sub_agent_info=sub_agent_info)
            raise

        finally:
            if conversation_id and not conversation_finished:
                await self._interrupt_conversation(
                    conversation_id,
                    {
                        "status": "interrupted",
                        "completed_at": datetime.now(timezone.utc),
                        "execution_time_seconds": time.time() - start_time,
                    },
                )
            if agent_id in self.agents:
                del self.agents[agent_id]

    async def _stream_workflow_agent(
        self,
        agent,
        agent_name: str,
        sub_agent_info: dict,
        conversation_id: Optional[str],
        *,
        last_saved_index: int,
        task: str,
        artifact_jsonl_path: Optional[Path] = None,
    ) -> int:
        """Stream one process_message_stream pass, broadcasting events and recording
        newly-completed history messages. Returns the updated last_saved_index.
        """
        async for msg in agent.process_message_stream(task):
            message_type = msg.get("type", "agent")
            content = msg.get("content", "")
            complete = msg.get("complete", False)
            msg_uuid = msg.get("uuid", str(uuid.uuid4()))

            content_payload = {"text": content}
            if message_type != "response":
                content_payload["message_type"] = message_type

            evt = AgentEvent(
                event_type="chat_message",
                content=content_payload,
                sender=agent_name,
                timestamp=datetime.now().isoformat(),
                is_streaming=(message_type in ["response", "thinking"] and not complete),
                uuid=msg_uuid,
                sub_agent_info=sub_agent_info,
            )
            await self.connection_manager.broadcast_event(evt, self.workspace_id, self.thread_id)

            if complete:
                current_history = agent.dump_message_history()
                for i in range(last_saved_index + 1, len(current_history)):
                    hist_msg = current_history[i]
                    if conversation_id:
                        await self._record_message(
                            conversation_id,
                            {
                                "role": hist_msg.get("role", "assistant"),
                                "content": hist_msg.get("content", []),
                                "stream_uuid": None,
                            },
                            i + 1,
                        )
                    self._write_jsonl_message(artifact_jsonl_path, hist_msg)
                last_saved_index = len(current_history) - 1

        return last_saved_index
