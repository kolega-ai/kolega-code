from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest

from ..llm.models import Message, MessageHistory
from ..models.public import AgentStatus
from ..utils.commands import CommandProcessor


class MockAgent:
    """Mock agent class for testing CommandProcessor"""

    def __init__(self):
        self.history = MessageHistory()
        self.command_processor = CommandProcessor(self)
        self._compress_message_history = AsyncMock()
        self.count_current_context = AsyncMock()


@pytest.fixture
def mock_agent():
    """Fixture to create a mock agent for testing"""
    return MockAgent()


@pytest.fixture
def command_processor(mock_agent):
    """Fixture to create a CommandProcessor instance with a mock agent"""
    return mock_agent.command_processor


@pytest.fixture
def mock_message():
    """Fixture to create a mock Message object"""
    message = create_autospec(Message)
    message.role = "user"
    message.content = "Test message"
    return message


@pytest.mark.asyncio
async def test_handle_help(command_processor):
    """Test the /help command handler"""
    help_text = await command_processor._handle_help()

    # Verify help text contains all available commands
    assert "/help" in help_text
    assert "/compress" in help_text
    assert "/clear" in help_text
    assert "/context" in help_text

    # Verify help text formatting
    assert help_text.startswith("# Available Commands")
    assert all(line.startswith("- `/") for line in help_text.split("\n")[2:])


@pytest.mark.asyncio
async def test_handle_compress(command_processor, mock_agent):
    """Test the /compress command handler"""
    result = await command_processor._handle_compress()

    # Verify compress was called and returned expected message
    mock_agent._compress_message_history.assert_called_once()
    assert result == "Message history compressed."


@pytest.mark.asyncio
async def test_handle_clear(command_processor, mock_agent, mock_message):
    """Test the /clear command handler"""
    # Add some messages to history first
    mock_agent.history.append(mock_message)
    mock_agent.history.append(mock_message)
    assert len(mock_agent.history) == 2

    result = await command_processor._handle_clear()

    # Verify history was cleared and returned expected message
    assert len(mock_agent.history) == 0
    assert result == "Message history cleared."


@pytest.mark.asyncio
async def test_handle_context_empty_history(command_processor):
    """Test the /context command handler with empty history"""
    result = await command_processor._handle_context()
    assert result == "Current context token count: 0"


@pytest.mark.asyncio
async def test_handle_context_with_history(command_processor, mock_agent, mock_message):
    """Test the /context command handler with non-empty history"""
    # Add mock message to history
    mock_agent.history.append(mock_message)

    # Mock token count response
    mock_token_count = MagicMock()
    mock_token_count.input_tokens = 100
    mock_agent.count_current_context.return_value = mock_token_count

    result = await command_processor._handle_context()

    # Verify token count was called and returned expected message
    mock_agent.count_current_context.assert_called_once()
    assert result == "Current context token count: 100"


@pytest.mark.asyncio
async def test_process_commands_decorator():
    """Test the process_commands decorator functionality"""

    @CommandProcessor.process_commands
    class TestAgent:
        def __init__(self):
            self.history = MessageHistory()
            self.command_processor = CommandProcessor(self)

        async def process_message_stream(self, message, attachments=None):
            yield {"type": "response", "content": "Normal processing", "complete": True}

    agent = TestAgent()

    # Test command processing
    responses = []
    async for response in agent.process_message_stream("/help"):
        responses.append(response)

    assert len(responses) == 1
    assert responses[0]["type"] == "response"
    assert "Available Commands" in responses[0]["content"]
    assert responses[0]["complete"] is True

    # Test normal message processing
    responses = []
    async for response in agent.process_message_stream("normal message"):
        responses.append(response)

    assert len(responses) == 1
    assert responses[0]["type"] == "response"
    assert responses[0]["content"] == "Normal processing"
    assert responses[0]["complete"] is True


@pytest.mark.asyncio
async def test_process_commands_invalid_command():
    """Test processing an invalid command"""

    @CommandProcessor.process_commands
    class TestAgent:
        def __init__(self):
            self.history = MessageHistory()
            self.command_processor = CommandProcessor(self)

        async def process_message_stream(self, message, attachments=None):
            yield {"type": "response", "content": "Normal processing", "complete": True}

    agent = TestAgent()

    # Test invalid command
    responses = []
    async for response in agent.process_message_stream("/invalid"):
        responses.append(response)

    assert len(responses) == 1
    assert responses[0]["type"] == "response"
    assert responses[0]["content"] == "Normal processing"
    assert responses[0]["complete"] is True
