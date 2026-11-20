import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Set, TypeVar, Union

from ..common import LogMixin
from kolega_code.config import AgentConfig
from kolega_code.events import AgentConnectionManager, AgentEvent
from kolega_code.services.file_system import FileSystem, LocalFileSystem
from kolega_code.services.base import TerminalManager, BrowserManager
from ..prompt_provider import AgentMode

if TYPE_CHECKING:
    from kolega_code.services.snapshots import SnapshotService


T = TypeVar("T")


class BaseTool(LogMixin):
    def __init__(
        self,
        project_path: Union[str, Path],
        workspace_id: str,
        thread_id: str,
        connection_manager: AgentConnectionManager,
        config: AgentConfig,
        caller,
        filesystem: Optional[FileSystem] = None,
        terminal_manager: Optional[TerminalManager] = None,
        browser_manager: Optional[BrowserManager] = None,
    ) -> None:
        self.workspace_id = workspace_id
        self.thread_id = thread_id
        self.project_path = Path(project_path) if isinstance(project_path, str) else project_path
        self.connection_manager = connection_manager
        self.config = config
        self.caller = caller

        # Create filesystem instance if not provided
        if filesystem is None:
            self.filesystem = LocalFileSystem(root_path=self.project_path)
        else:
            self.filesystem = filesystem

        # Store optional managers (individual tools will use them if needed)
        self.terminal_manager = terminal_manager
        self.browser_manager = browser_manager

    # --- Vibe-mode edit policy helpers ---
    def _is_vibe_mode(self) -> bool:
        """Return True if caller agent is in VIBE mode (same check pattern as agents)."""
        return getattr(self.caller, "agent_mode", None) == AgentMode.VIBE.value

    def _get_vibe_blacklist_basenames(self) -> Set[str]:
        """Basename blacklist for files that should not be edited in vibe mode."""
        protected_files = getattr(self.caller, "protected_files", None) if self.caller else None
        if protected_files:
            return set(protected_files)
        return {"package.json", "tsconfig.json"}

    def _enforce_vibe_edit_policy(self, path: str) -> Optional[str]:
        """Return message if blocked; None if allowed."""
        if not self._is_vibe_mode():
            return None

        if os.path.basename(path) in self._get_vibe_blacklist_basenames():
            return f"You are not allowed to edit this file: {path}"
        return None

    def _mutate_with_optional_snapshot(
        self,
        *,
        tool_name: str,
        reason: str,
        paths: list[str],
        mutate: Callable[[], T],
    ) -> T:
        """Run one mutation, recording it only when every path can be snapshotted."""

        snapshot_service: Optional["SnapshotService"] = getattr(self, "_snapshot_service", None)
        if snapshot_service is None or not snapshot_service.can_snapshot_paths(paths):
            return mutate()
        mutation = snapshot_service.record_mutation(
            tool_name=tool_name,
            tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
            reason=reason,
            paths=paths,
            mutate=mutate,
        )
        return mutation.result

    async def send_edit_preview(self, preview: Optional[dict], *, tool_call_id: Optional[str], tool_name: str) -> None:
        """Broadcast a UI-only diff/head preview for inline rendering in the chat.

        The preview rides a dedicated ``file_edit_preview`` event and is never added to
        the tool result, so it costs zero model tokens. No-op when there is nothing to
        show (``preview`` is None) or no tool_call_id is available to route it.
        """
        if not preview or not tool_call_id:
            return

        # Attach dispatch metadata for sub-agent callers so the UI can route correctly
        # (mirrors StreamingTool.send_streaming_update).
        sub_agent_info = None
        sub_agent_context = getattr(self.caller, "sub_agent_context", None)
        if getattr(self.caller, "sub_agent", False) is True and isinstance(sub_agent_context, dict):
            sub_agent_info = dict(sub_agent_context)

        event = AgentEvent(
            sender=self.caller.agent_name,
            event_type="file_edit_preview",
            content={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "workspace_root": str(self.project_path.resolve()),
                **preview,
            },
            is_streaming=False,
            sub_agent_info=sub_agent_info,
        )
        await self.connection_manager.broadcast_event(event, self.workspace_id, self.thread_id)
