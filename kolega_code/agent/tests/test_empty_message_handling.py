"""Tests for shared BaseAgent empty message handling."""

from unittest.mock import Mock, patch

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.agent.llm.models import Message


def test_append_assistant_message_with_empty_content():
    """Empty assistant messages get placeholder text."""
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.history = []

    empty_message = Message(role="assistant", content=[])

    with patch("builtins.print") as mock_print:
        BaseAgent.append_assistant_message(mock_agent, empty_message)

    mock_print.assert_called_with("Warning: Assistant message has empty content, replacing with placeholder")

    assert len(mock_agent.history) == 1
    appended_msg = mock_agent.history[0]
    assert appended_msg.role == "assistant"
    assert len(appended_msg.content) == 1
    assert appended_msg.content[0].text == "[Assistant returned no message content]"
