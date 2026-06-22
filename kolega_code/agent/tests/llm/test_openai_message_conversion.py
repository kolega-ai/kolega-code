
from kolega_code.llm.models import ImageBlock, Message, MessageChunk, MessageHistory, ThinkingBlock, ToolCall, ToolResult, TextBlock


def _image() -> ImageBlock:
    return ImageBlock(image_type="base64", media_type="image/png", data="BASE64")


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
    assert messages[1]['content'] == '3'


def test_image_tool_result_serializes_as_tool_text_plus_user_image():
    tool_call = ToolCall(id='call_img', name='read_image', input={'path': 'shot.png'})
    assistant = Message(role='assistant', content=[tool_call])
    result = Message(
        role='user',
        content=[ToolResult(tool_use_id='call_img', content=[_image()], name='read_image', is_error=False)],
    )

    messages = MessageHistory([assistant, result]).to_openai()

    assert messages[0]['role'] == 'assistant'
    assert messages[0]['tool_calls'][0]['id'] == 'call_img'
    assert messages[1]['role'] == 'tool'
    assert messages[1]['tool_call_id'] == 'call_img'
    assert isinstance(messages[1]['content'], str)
    assert 'image' in messages[1]['content']
    assert 'image_url' not in messages[1]['content']
    assert messages[2]['role'] == 'user'
    assert messages[2]['content'][0]['type'] == 'text'
    assert messages[2]['content'][1] == {
        'type': 'image_url',
        'image_url': {'url': 'data:image/png;base64,BASE64'},
    }


def test_mixed_text_and_image_tool_result_preserves_text_in_tool_output():
    tool_call = ToolCall(id='call_img', name='read_image', input={'path': 'shot.png'})
    result = ToolResult(
        tool_use_id='call_img',
        content=[TextBlock('caption text'), _image()],
        name='read_image',
        is_error=False,
    )

    messages = MessageHistory([
        Message(role='assistant', content=[tool_call]),
        Message(role='user', content=[result]),
    ]).to_openai()

    assert messages[1]['role'] == 'tool'
    assert 'caption text' in messages[1]['content']
    assert 'attached in the following user message' in messages[1]['content']
    assert messages[2]['role'] == 'user'
    assert any(part.get('type') == 'image_url' for part in messages[2]['content'])


def test_multiple_tool_results_stay_contiguous_before_image_followup():
    call_1 = ToolCall(id='call_1', name='read_image', input={'path': 'shot.png'})
    call_2 = ToolCall(id='call_2', name='read_file', input={'path': 'README.md'})
    result_1 = ToolResult(tool_use_id='call_1', content=[_image()], name='read_image', is_error=False)
    result_2 = ToolResult(tool_use_id='call_2', content='file text', name='read_file', is_error=False)

    messages = MessageHistory([
        Message(role='assistant', content=[call_1, call_2]),
        Message(role='user', content=[result_1, result_2]),
    ]).to_openai()

    assert [message['role'] for message in messages] == ['assistant', 'tool', 'tool', 'user']
    assert messages[1]['tool_call_id'] == 'call_1'
    assert messages[2]['tool_call_id'] == 'call_2'
    assert messages[2]['content'] == 'file text'
    assert any(part.get('type') == 'image_url' for part in messages[3]['content'])


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

