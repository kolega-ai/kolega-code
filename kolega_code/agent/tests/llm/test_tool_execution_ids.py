from kolega_code.llm.models import ContentBlock, Message, ToolCall
from kolega_code.llm.tool_execution_ids import ToolExecutionIdRegistry, new_tool_execution_id


def test_new_tool_execution_id_uses_internal_prefix_and_is_unique():
    first = new_tool_execution_id()
    second = new_tool_execution_id()

    assert first.startswith("tool_exec_")
    assert second.startswith("tool_exec_")
    assert first != second


def test_tool_execution_id_registry_reuses_id_within_response():
    registry = ToolExecutionIdRegistry()

    first = registry.get_or_create("provider_tool_call_id")
    second = registry.get_or_create("provider_tool_call_id")

    assert first == second


def test_tool_execution_id_registry_is_response_scoped():
    first_registry = ToolExecutionIdRegistry()
    second_registry = ToolExecutionIdRegistry()

    first = first_registry.get_or_create("provider_tool_call_id")
    second = second_registry.get_or_create("provider_tool_call_id")

    assert first != second


def test_tool_call_from_dict_preserves_existing_execution_id():
    tool_call = ToolCall.from_dict(
        {
            "type": "tool_call",
            "id": "provider_tool_call_id",
            "name": "read_file",
            "input": {"path": "README.md"},
            "execution_id": "tool_exec_existing",
        }
    )

    assert tool_call.id == "provider_tool_call_id"
    assert tool_call.execution_id == "tool_exec_existing"


def test_tool_call_from_dict_generates_execution_id_for_legacy_records():
    tool_call = ToolCall.from_dict(
        {
            "type": "tool_call",
            "id": "provider_tool_call_id",
            "name": "read_file",
            "input": {"path": "README.md"},
        }
    )

    assert tool_call.id == "provider_tool_call_id"
    assert tool_call.execution_id.startswith("tool_exec_")


def test_message_from_anthropic_uses_supplied_execution_id_registry():
    class ToolUseBlock:
        type = "tool_use"
        id = "provider_tool_call_id"
        name = "read_file"
        input = {"path": "README.md"}

    class AnthropicMessage:
        role = "assistant"
        stop_reason = "tool_use"
        content = [ToolUseBlock()]

    registry = ToolExecutionIdRegistry()
    execution_id = registry.get_or_create("provider_tool_call_id")

    message = Message.from_anthropic(AnthropicMessage(), tool_execution_ids=registry)

    assert message.tool_calls[0].id == "provider_tool_call_id"
    assert message.tool_calls[0].execution_id == execution_id
    assert message.content[0].execution_id == execution_id


def test_message_from_openai_uses_supplied_execution_id_registry():
    class Function:
        name = "read_file"
        arguments = '{"path": "README.md"}'

    class ToolCall:
        id = "provider_tool_call_id"
        function = Function()

    class OpenAIMessage:
        content = None
        finish_reason = "tool_calls"
        tool_calls = [ToolCall()]

    registry = ToolExecutionIdRegistry()
    execution_id = registry.get_or_create("provider_tool_call_id")

    message = Message.from_openai(OpenAIMessage(), tool_execution_ids=registry)

    assert message.tool_calls[0].id == "provider_tool_call_id"
    assert message.tool_calls[0].execution_id == execution_id
    assert message.content[0].execution_id == execution_id


def test_message_from_google_uses_supplied_execution_id_registry():
    class Content:
        parts = []

    class Candidate:
        content = Content()

    class FunctionCall:
        id = "provider_tool_call_id"
        name = "read_file"
        args = {"path": "README.md"}

    class GoogleMessage:
        candidates = [Candidate()]
        function_calls = [FunctionCall()]
        finish_reason = "STOP"

    registry = ToolExecutionIdRegistry()
    execution_id = registry.get_or_create("provider_tool_call_id")

    message = Message.from_google(GoogleMessage(), tool_execution_ids=registry)

    assert message.tool_calls[0].id == "provider_tool_call_id"
    assert message.tool_calls[0].execution_id == execution_id
    assert message.content == []


def test_message_from_openai_stream_uses_supplied_execution_id_registry():
    class Function:
        name = "read_file"
        arguments = '{"path": "README.md"}'

    class ToolCall:
        id = "provider_tool_call_id"
        function = Function()

    registry = ToolExecutionIdRegistry()
    execution_id = registry.get_or_create("provider_tool_call_id")

    message = Message.from_openai_stream(
        role="assistant",
        content="",
        tool_calls={0: ToolCall()},
        stop_reason="tool_calls",
        tool_execution_ids=registry,
    )

    assert message.tool_calls[0].id == "provider_tool_call_id"
    assert message.tool_calls[0].execution_id == execution_id
    assert message.content[0].execution_id == execution_id


def test_message_from_google_stream_uses_supplied_execution_id_registry():
    class ToolCall:
        id = "provider_tool_call_id"
        name = "read_file"
        args = {"path": "README.md"}

    registry = ToolExecutionIdRegistry()
    execution_id = registry.get_or_create("provider_tool_call_id")

    message = Message.from_google_stream(
        role="assistant",
        content="",
        tool_calls={0: ToolCall()},
        stop_reason="STOP",
        tool_execution_ids=registry,
    )

    assert message.tool_calls[0].id == "provider_tool_call_id"
    assert message.tool_calls[0].execution_id == execution_id
    assert message.content[0].execution_id == execution_id


def test_content_block_from_dict_generates_execution_id_for_legacy_tool_call():
    block = ContentBlock.from_dict(
        {
            "type": "tool_call",
            "id": "provider_tool_call_id",
            "name": "read_file",
            "input": {"path": "README.md"},
        }
    )

    assert isinstance(block, ToolCall)
    assert block.execution_id.startswith("tool_exec_")
