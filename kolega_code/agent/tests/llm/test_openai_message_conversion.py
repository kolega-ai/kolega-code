import pytest

from kolega_code.agent.llm.models import Message, MessageHistory, ToolCall, ToolResult, TextBlock


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



