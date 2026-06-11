"""Tests for Conversation empty message handling."""

import logging

from kolega_code.agent.conversation import Conversation
from kolega_code.agent.llm.models import Message, TextBlock


def test_append_assistant_message_with_empty_content(caplog):
    """Empty assistant messages get placeholder text."""
    conversation = Conversation()

    empty_message = Message(role="assistant", content=[])

    with caplog.at_level(logging.WARNING, logger="kolega_code.agent.conversation"):
        conversation.append_assistant(empty_message)

    assert "Assistant message has empty content" in caplog.text

    assert len(conversation.history) == 1
    appended_msg = conversation.history[0]
    assert appended_msg.role == "assistant"
    assert len(appended_msg.content) == 1
    assert appended_msg.content[0].text == "[Assistant returned no message content]"


def test_append_user_message_with_empty_content(caplog):
    """Empty user messages get placeholder text."""
    conversation = Conversation()

    with caplog.at_level(logging.WARNING, logger="kolega_code.agent.conversation"):
        conversation.append_user([])

    assert "User message has empty content" in caplog.text

    assert len(conversation.history) == 1
    appended_msg = conversation.history[0]
    assert appended_msg.role == "user"
    assert appended_msg.content[0].text == "[User provided no message content]"


def test_append_assistant_message_with_content_is_untouched():
    conversation = Conversation()

    message = Message(role="assistant", content=[TextBlock(text="hello")])
    conversation.append_assistant(message)

    assert conversation.history[-1] is message
