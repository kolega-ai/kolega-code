import pytest

from kolega_code.agent.llm.instrumented_client import InstrumentedLLMClient


class _UsageRecorder:
    def __init__(self):
        self.payload = None

    def record_usage(self, usage_data):
        self.payload = usage_data


@pytest.mark.asyncio
async def test_usage_recorder_maps_openai_cached_tokens():
    recorder = _UsageRecorder()
    client = InstrumentedLLMClient(
        provider='openai',
        api_key='sk',
        langfuse_client=None,
        user_id='u1',
        workspace_id='w1',
        thread_id='t1',
        usage_recorder=recorder,
    )

    usage = {
        'provider': 'openai',
        'prompt_tokens': 10,
        'completion_tokens': 2,
        'cache_read_input_tokens': 2048,
    }

    await client._record_usage(usage, model='m1', success=True)
    assert recorder.payload['input_tokens'] == 10
    assert recorder.payload['output_tokens'] == 2
    assert recorder.payload['cache_read_input_tokens'] == 2048


@pytest.mark.asyncio
async def test_usage_recorder_maps_moonshot_response_usage():
    recorder = _UsageRecorder()
    client = InstrumentedLLMClient(
        provider='moonshot',
        api_key='sk',
        langfuse_client=None,
        user_id='u1',
        workspace_id='w1',
        thread_id='t1',
        usage_recorder=recorder,
    )

    usage = {
        'provider': 'moonshot',
        'input_tokens': 123,
        'output_tokens': 45,
        'cache_read_input_tokens': 67,
        'cache_write_input_tokens': 89,
        # Moonshot may return these aliases too; billing should use the
        # Anthropic-shaped fields above for Kimi accounting.
        'prompt_tokens': 999,
        'completion_tokens': 888,
        'total_tokens': 1887,
    }

    await client._record_usage(usage, model='kimi-k2.6', success=True)

    assert recorder.payload['provider'] == 'moonshot'
    assert recorder.payload['model'] == 'kimi-k2.6'
    assert recorder.payload['input_tokens'] == 123
    assert recorder.payload['output_tokens'] == 45
    assert recorder.payload['cache_read_input_tokens'] == 67
    assert recorder.payload['cache_write_input_tokens'] == 89
    assert recorder.payload['metadata']['raw_usage'] == usage
