import uuid
import inspect
from pathlib import Path
from typing import Union, Optional
from datetime import datetime, timezone
import time

from ..config import AgentConfig
from ..models.public import AgentEvent
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
        self.sub_agent_recorder = getattr(caller, "sub_agent_recorder", None) if caller else None
        # No need to store these separately since they're already in the parent class
        # self.terminal_manager = terminal_manager
        # self.browser_manager = browser_manager

    async def _maybe_await(self, value):
        if inspect.isawaitable(value):
            return await value
        return value

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

    async def _send_status_event(self, status: str, message: str, sub_agent_info: Optional[dict] = None) -> None:
        """Helper method to send status events."""
        event = AgentEvent(
            event_type="chat_message",
            content={"status": status, "message": message},
            sender=self.caller.agent_name if self.caller else "agent-tool",
            sub_agent_info=sub_agent_info,
        )
        await self.connection_manager.broadcast_event(event, self.workspace_id, self.thread_id)

    async def _dispatch_agent(self, agent_class_import: str, task: str) -> str:
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
        agent_name = agent_class.agent_name

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

        # Calculate depth based on whether the caller is also a sub-agent
        parent_depth = 0
        if hasattr(self.caller, "sub_agent") and self.caller.sub_agent:
            # If the caller is a sub-agent, get its depth
            # For now, we'll increment from 1, but ideally we'd track this
            parent_depth = 1

        # Attached to every event from this dispatch so the UI can group and
        # disambiguate concurrently running sub-agents.
        sub_agent_info = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "task": task[:120],
            "conversation_id": conversation_id,
            "parent_tool_call_id": tool_call_id,
            "depth": parent_depth + 1,
        }

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
                prompt_extensions=getattr(self.caller, "prompt_extensions", None) if self.caller else None,
                tool_extensions=getattr(self.caller, "tool_extensions", None) if self.caller else None,
                usage_recorder=getattr(self.caller, "usage_recorder", None) if self.caller else None,
                sub_agent_recorder=getattr(self.caller, "sub_agent_recorder", None) if self.caller else None,
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

            # Send completion status
            await self._send_status_event(
                "STOPPED", f"Completed {agent_name} task", sub_agent_info=sub_agent_info
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
