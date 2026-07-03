from kolega_code.llm.instrumented_client import MinimalLangfuseStreamWrapper
from kolega_code.llm.models import Message


# TODO: Fix after qwen-3-coder-plus PR is merged - needs OpenAI cache token support in Langfuse
def test_langfuse_normalizes_openai_cache_tokens():
    msg = Message(
        role="assistant",
        content="ok",
        usage_metadata={
            "provider": "openai",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cache_read_input_tokens": 2048,
        },
    )

    wrapper = MinimalLangfuseStreamWrapper(
        stream=None, generation=None, trace=None, instrumented_client=None, model="x"
    )
    usage = wrapper._extract_langfuse_usage(msg)
    assert usage is not None
    assert usage["input"] == 10
    assert usage["output"] == 2
    assert usage["total"] == 12
    assert usage["cache_read_input_tokens"] == 2048


def test_langfuse_normalizes_deepseek_usage():
    # DeepSeek now uses the OpenAI-compatible endpoint, so its usage is OpenAI-shaped
    # (prompt_tokens/completion_tokens).
    msg = Message(
        role="assistant",
        content="ok",
        usage_metadata={
            "provider": "deepseek",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cache_read_input_tokens": 3,
            "cache_write_input_tokens": 4,
        },
    )

    wrapper = MinimalLangfuseStreamWrapper(
        stream=None, generation=None, trace=None, instrumented_client=None, model="x"
    )
    usage = wrapper._extract_langfuse_usage(msg)
    assert usage is not None
    assert usage["input"] == 10
    assert usage["output"] == 2
    assert usage["total"] == 12
    assert usage["cache_read_input_tokens"] == 3
    assert usage["cache_creation_input_tokens"] == 4


def test_langfuse_normalizes_fireworks_openai_usage():
    msg = Message(
        role="assistant",
        content="ok",
        usage_metadata={
            "provider": "fireworks",
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
            "cache_read_input_tokens": 6,
        },
    )

    wrapper = MinimalLangfuseStreamWrapper(
        stream=None, generation=None, trace=None, instrumented_client=None, model="x"
    )
    usage = wrapper._extract_langfuse_usage(msg)
    assert usage is not None
    assert usage["input"] == 20
    assert usage["output"] == 5
    assert usage["total"] == 25
    assert usage["cache_read_input_tokens"] == 6
