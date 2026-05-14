import pytest

from kolega_code.agent.llm.instrumented_client import MinimalLangfuseStreamWrapper
from kolega_code.agent.llm.models import Message


# TODO: Fix after qwen-3-coder-plus PR is merged - needs OpenAI cache token support in Langfuse
def test_langfuse_normalizes_openai_cache_tokens():
    msg = Message(role='assistant', content='ok', usage_metadata={
        'provider': 'openai',
        'prompt_tokens': 10,
        'completion_tokens': 2,
        'total_tokens': 12,
        'cache_read_input_tokens': 2048,
    })

    wrapper = MinimalLangfuseStreamWrapper(stream=None, generation=None, trace=None, instrumented_client=None, model='x')
    usage = wrapper._extract_langfuse_usage(msg)
    assert usage['input'] == 10
    assert usage['output'] == 2
    assert usage['total'] == 12
    assert usage['cache_read_input_tokens'] == 2048



