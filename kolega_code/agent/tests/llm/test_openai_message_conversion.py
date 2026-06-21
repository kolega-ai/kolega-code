import pytest

from kolega_code.llm.models import Message, MessageChunk, MessageHistory, ThinkingBlock, ToolCall, ToolResult, TextBlock


# TODO: Fix after qwen-3-coder-plus PR is merged - needs tool result filtering in Message.to_openai()
def test_mixed_content_splits_tool_result_to_separate_tool_messages():
    # Assistant message with text + tool_call + tool_result
    tool_call = ToolCall(id='call_123', name='sum', input={'a': 1, 'b': 2})
    tool_result = ToolResult(tool_use_id='call_123', content='3', name='sum', is_error=False)
    assistant = Message(role='assistant', content=[TextBlock('ok'), tool_call, tool_result])

    mh = MessageHistory([assistant])
    messages = mh.to_openai()

    # Expect two messages: assistant (with tool_calls) then tool with matching tool_call_id
    assert messages[0]['role'] == 'assistant'
    assert 'tool_calls' in messages[0]
    assert messages[0]['tool_calls'][0]['id'] == 'call_123'
    # content should not include tool_result; only text parts
    assert isinstance(messages[0]['content'], list)
    assert all(isinstance(p, dict) and p.get('type') in ('text', 'image_url') for p in messages[0]['content'])

    assert messages[1]['role'] == 'tool'
    assert messages[1]['tool_call_id'] == 'call_123'
    # tool content can be a string or structured parts depending on our representation
    assert 'content' in messages[1]


def test_openai_message_reasoning_content_maps_to_thinking_block():
    class OpenAIMessage:
        content = 'final answer'
        reasoning_content = 'internal reasoning summary'
        tool_calls = None

    message = Message.from_openai(OpenAIMessage())

    assert isinstance(message.content[0], ThinkingBlock)
    assert message.content[0].thinking == 'internal reasoning summary'
    assert isinstance(message.content[1], TextBlock)
    assert message.content[1].text == 'final answer'


def test_openai_stream_reasoning_content_maps_to_thinking_block():
    message = Message.from_openai_stream(
        role='assistant',
        reasoning_content='streamed reasoning',
        content='streamed answer',
        stop_reason='stop',
    )

    assert isinstance(message.content[0], ThinkingBlock)
    assert message.content[0].thinking == 'streamed reasoning'
    assert isinstance(message.content[1], TextBlock)
    assert message.content[1].text == 'streamed answer'
    assert message.stop_reason == 'end_turn'


def test_openai_stream_chunk_reasoning_content_delta_maps_to_thinking_chunk():
    class Delta:
        content = None
        reasoning_content = 'reasoning delta'

    class Choice:
        delta = Delta()

    class Chunk:
        choices = [Choice()]

    chunk = MessageChunk.from_openai(Chunk())

    assert chunk.type == 'thinking'
    assert chunk.thinking == 'reasoning delta'

