import abc
from typing import Any

from .models.public import AgentEvent


class AgentConnectionManager(abc.ABC):
    """Abstract base class for agent connection managers."""

    @abc.abstractmethod
    async def connect(self, websocket: Any, workspace_id: str, thread_id: str, connection_type: str, user_info=None) -> None:
        """
        Connect a client to a specific workspace, thread and connection type.

        Args:
            websocket: The WebSocket connection
            workspace_id: ID of the workspace to connect to
            thread_id: ID of the thread to connect to
            connection_type: Type of connection ('chat', 'terminal', or 'logs')
            user_info: Optional user information dictionary
        """

    @abc.abstractmethod
    def disconnect(self, websocket: Any, workspace_id: str, thread_id: str, connection_type: str) -> None:
        """
        Disconnect a client from a specific workspace, thread and connection type.

        Args:
            websocket: The WebSocket connection
            workspace_id: ID of the workspace to disconnect from
            thread_id: ID of the thread to disconnect from
            connection_type: Type of connection ('chat', 'terminal', or 'logs')
        """

    @abc.abstractmethod
    async def broadcast_event(self, event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        """
        Broadcast a chat message to all connected clients for a thread.

        Args:
            event: The event to broadcast
            workspace_id: ID of the workspace
            thread_id: ID of the thread
        """

    @abc.abstractmethod
    def get_connection_count(self, workspace_id: str, thread_id: str) -> dict:
        """
        Get the number of connections for each type for a thread.

        Args:
            workspace_id: ID of the workspace
            thread_id: ID of the thread

        Returns:
            Dictionary with connection counts
        """
