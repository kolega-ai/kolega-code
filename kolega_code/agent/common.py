from kolega_code.events import AgentEvent


class LogMixin:
    """
    A mixin class providing logging functionality to agents.

    This mixin expects the implementing class to have a connection_manager attribute,
    a workspace_id attribute, and a thread_id attribute.
    """

    async def log_info(self, message: str, sender: str = "agent") -> None:
        """
        Log an informational message to the logs panel.

        Args:
            message: The information message to log
        """
        log_event = AgentEvent(event_type="log_message", sender=sender, content={"text": message, "level": "info"})

        await self.connection_manager.broadcast_event(log_event, self.workspace_id, self.thread_id)

    async def log_error(self, message: str, sender: str = "agent") -> None:
        """
        Log an error message to the logs panel.

        Args:
            message: The error message to log
        """
        log_event = AgentEvent(event_type="log_message", sender=sender, content={"text": message, "level": "error"})
        await self.connection_manager.broadcast_event(log_event, self.workspace_id, self.thread_id)

    async def log_warning(self, message: str, sender: str = "agent") -> None:
        """
        Log a warning message to the logs panel.

        Args:
            message: The warning message to log
        """
        log_event = AgentEvent(event_type="log_message", sender=sender, content={"text": message, "level": "warning"})
        await self.connection_manager.broadcast_event(log_event, self.workspace_id, self.thread_id)
