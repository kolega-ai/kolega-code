from functools import wraps
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

from ..llm.models import MessageHistory


class CommandProcessor:
    """
    Processes special commands in agent messages and handles their execution.
    """

    def __init__(self, agent: Any) -> None:
        """
        Initialize the CommandProcessor with a reference to its parent agent.

        Args:
            agent: The agent instance this processor belongs to
        """
        self.agent = agent
        self.commands = {
            "/help": self._handle_help,
            "/compress": self._handle_compress,
            "/clear": self._handle_clear,
            "/reset": self._handle_clear,
            "/context": self._handle_context,
        }

    async def _handle_help(self) -> str:
        """Handle the /help command."""
        help_text = """# Available Commands

- `/help` - Show this help message
- `/compress` - Compress message history
- `/clear` - Clear message history
- `/reset` - Clear message history
- `/context` - Show current context token count"""
        return help_text

    async def _handle_compress(self) -> str:
        """Handle the /compress command."""
        await self.agent._compress_message_history()
        return "Message history compressed."

    async def _handle_clear(self) -> str:
        """Handle the /clear command."""
        self.agent.history = MessageHistory()
        return "Message history cleared."

    async def _handle_context(self) -> str:
        """Handle the /context command."""
        input_tokens = 0
        if len(self.agent.history) > 0:
            token_count = await self.agent.count_current_context()
            input_tokens = token_count.input_tokens
        return f"Current context token count: {input_tokens}"

    @staticmethod
    def process_commands(cls: Type) -> Type:
        """
        Class decorator that adds command processing to process_message_stream.

        Args:
            cls: The class to decorate

        Returns:
            Decorated class with command processing
        """
        original_method = cls.process_message_stream

        @wraps(original_method)
        async def wrapped_process_message_stream(
            self, message: str, attachments: Optional[List[Dict[str, Any]]] = None
        ) -> AsyncGenerator[Dict[str, Any], None]:
            # Check if message is a command
            stripped_message = message.strip()
            if stripped_message in self.command_processor.commands:
                command_result = await self.command_processor.commands[stripped_message]()

                yield {"type": "response", "content": command_result, "complete": True}

                return

            # If not a command, process normally
            async for response in original_method(self, message, attachments):
                yield response

        cls.process_message_stream = wrapped_process_message_stream
        return cls
