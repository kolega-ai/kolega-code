# ruff: noqa: F401,F811,E402
import pytest

from kolega_code.llm.models import Message, MessageHistory, TextBlock


@pytest.fixture
def simple_messages():
    """Simple test messages."""
    return MessageHistory([Message("user", [TextBlock("Hello, how are you?")])])


@pytest.fixture
def simple_system():
    """Simple system message."""
    return Message("system", [TextBlock("You are a helpful assistant.")])


@pytest.fixture
def complex_messages():
    """Multi-turn conversation with various content types."""
    return MessageHistory(
        [
            Message("user", [TextBlock("Hello! I need help with Python.")]),
            Message("assistant", [TextBlock("I'd be happy to help you with Python. What would you like to know?")]),
            Message("user", [TextBlock("Can you explain list comprehensions?")]),
            Message(
                "assistant",
                [
                    TextBlock(
                        "List comprehensions are a concise way to create lists in Python. Here's the syntax: [expression for item in iterable if condition]"
                    )
                ],
            ),
            Message("user", [TextBlock("Can you give me an example?")]),
        ]
    )


def calculate_percentage_difference(local_count: int, api_count: int) -> float:
    """Calculate percentage difference between local and API counts."""
    if api_count == 0:
        return 0.0 if local_count == 0 else 100.0
    return abs(local_count - api_count) / api_count * 100


def get_accuracy_threshold(api_count_or_complexity: int | str = "simple", *, has_tools: bool = False) -> float:
    """Get allowed local-vs-API token-count variance as a percentage.

    Most integration tests pass the API prompt-token count so the threshold can
    be stricter for realistic contexts and looser for tiny samples where fixed
    overhead dominates. Older callers may still pass a named complexity bucket;
    keep that supported for compatibility.
    """
    if isinstance(api_count_or_complexity, str):
        thresholds = {
            "simple": 10.0,
            "system": 15.0,
            "tools": 20.0,
            "complex": 15.0,
            "images": 25.0,
            "full_context": 20.0,
        }
        if has_tools and api_count_or_complexity not in {"images"}:
            return max(thresholds.get(api_count_or_complexity, 15.0), 20.0)
        return thresholds.get(api_count_or_complexity, 15.0)

    api_count = api_count_or_complexity
    if has_tools:
        return 20.0
    return 15.0 if api_count < 200 else 10.0
