"""Request shaping and usage accounting for the OpenRouter gateway.

OpenRouter routes each request to an upstream provider, so optional request
fields act as routing hints and its usage payload carries a cache-write counter
the direct OpenAI-compatible providers never send. These tests pin that behavior
without touching the network.
"""

import asyncio

import pytest

from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import Message, MessageHistory
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.openai import OpenAIProvider, OpenAIStreamWrapper

REASONING_MODEL = "z-ai/glm-5.2"
NO_TEMPERATURE_MODEL = "openai/gpt-5.6-sol"
TEMPERATURE_MODEL = "anthropic/claude-opus-5"


class _Recorder:
    """Minimal AsyncOpenAI stand-in that captures the request kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Stream:
            async def __anext__(self):
                raise StopAsyncIteration

            async def aclose(self):
                return None

        return _Stream()

    @property
    def last(self) -> dict:
        assert self.calls, "no request was issued"
        return self.calls[-1]


def _stream_request(monkeypatch, *, model: str, thinking=None, provider_name: str = "openrouter") -> dict:
    provider = OpenAIProvider(api_key="sk-or-test", provider_name=provider_name)
    recorder = _Recorder()
    monkeypatch.setattr(provider, "async_client", recorder)

    async def run():
        await provider.stream(
            MessageHistory([]),
            params=GenerationParams(temperature=0.7, max_completion_tokens=4096, thinking=thinking),
            model=model,
        )

    asyncio.run(run())
    return recorder.last


def test_client_targets_openrouter_with_attribution_and_session_headers() -> None:
    client = LLMClient("openrouter", "sk-or-test")

    assert isinstance(client.provider, OpenAIProvider)
    assert client.provider.provider_name == "openrouter"
    assert client.provider.base_url == "https://openrouter.ai/api/v1"

    headers = dict(client.provider.async_client.default_headers)
    assert headers["HTTP-Referer"] == "https://github.com/kolega-ai/kolega-code"
    assert headers["X-Title"] == "Kolega Code"
    # Sticky routing: keeps a conversation on one upstream so its cache hits.
    assert headers["x-session-id"] == client.provider._session_id


def test_reasoning_effort_is_nested_under_extra_body(monkeypatch) -> None:
    request = _stream_request(monkeypatch, model=REASONING_MODEL, thinking="high")
    assert request["extra_body"]["reasoning"] == {"effort": "high"}
    # The gateway object must not leak into the top-level OpenAI request.
    assert "reasoning" not in request


def test_disabling_reasoning_uses_enabled_false_not_effort_none(monkeypatch) -> None:
    # "none" is not honored by every upstream and reasoning-mandatory models
    # reject it, so the documented off switch is {"enabled": False}.
    request = _stream_request(monkeypatch, model=REASONING_MODEL, thinking="none")
    assert request["extra_body"]["reasoning"] == {"enabled": False}


def test_caching_and_usage_accounting_are_requested(monkeypatch) -> None:
    request = _stream_request(monkeypatch, model=REASONING_MODEL, thinking="high")
    extra_body = request["extra_body"]

    # Automatic prompt caching: OpenRouter advances this breakpoint as the
    # conversation grows, so Anthropic/Vertex/Bedrock caching works with no
    # per-block markers in our serializer.
    assert extra_body["cache_control"] == {"type": "ephemeral"}
    assert extra_body["usage"] == {"include": True}
    # Every contributor survives: a plain dict update would drop the others.
    assert set(extra_body) == {"reasoning", "cache_control", "usage"}


def test_max_tokens_is_omitted_unless_an_operator_opts_in(monkeypatch) -> None:
    # A catalog-default cap excludes every upstream whose output cap is lower,
    # biasing routing and price for no benefit.
    request = _stream_request(monkeypatch, model=REASONING_MODEL)
    assert "max_tokens" not in request
    assert "max_completion_tokens" not in request

    monkeypatch.setenv("KOLEGA_CODE_OPENROUTER_MAX_TOKENS", "8192")
    request = _stream_request(monkeypatch, model=REASONING_MODEL)
    assert request["max_tokens"] == 8192


@pytest.mark.parametrize("value", ["0", "-5", "not-a-number", ""])
def test_invalid_max_tokens_override_falls_back_to_omitting(monkeypatch, value: str) -> None:
    monkeypatch.setenv("KOLEGA_CODE_OPENROUTER_MAX_TOKENS", value)
    request = _stream_request(monkeypatch, model=REASONING_MODEL)
    assert "max_tokens" not in request


def test_temperature_follows_the_catalog_capability_flag(monkeypatch) -> None:
    # gpt-5.x reject an explicit temperature outright.
    assert "temperature" not in _stream_request(monkeypatch, model=NO_TEMPERATURE_MODEL)
    assert _stream_request(monkeypatch, model=TEMPERATURE_MODEL)["temperature"] == 0.7


def test_unknown_models_keep_sending_temperature(monkeypatch) -> None:
    # Overlay/unlisted models must not lose sampling params just for being absent.
    assert _stream_request(monkeypatch, model="vendor/not-in-catalog")["temperature"] == 0.7


def test_routing_preferences_are_absent_unless_configured(monkeypatch) -> None:
    assert "provider" not in _stream_request(monkeypatch, model=REASONING_MODEL)["extra_body"]

    monkeypatch.setenv("KOLEGA_CODE_OPENROUTER_PROVIDER_ORDER", "anthropic, google")
    monkeypatch.setenv("KOLEGA_CODE_OPENROUTER_PROVIDER_ONLY", "anthropic")
    extra_body = _stream_request(monkeypatch, model=REASONING_MODEL)["extra_body"]
    assert extra_body["provider"] == {"order": ["anthropic", "google"], "only": ["anthropic"]}


def test_other_openai_compatible_providers_are_untouched(monkeypatch) -> None:
    request = _stream_request(monkeypatch, model="glm-5.2", provider_name="zai")
    assert request["max_tokens"] == 4096
    assert request["temperature"] == 0.7
    assert "extra_body" not in request


def test_merge_request_params_combines_extra_body_instead_of_replacing() -> None:
    params: dict = {}
    OpenAIProvider._merge_request_params(params, {"extra_body": {"reasoning": {"effort": "high"}}})
    OpenAIProvider._merge_request_params(params, {"extra_body": {"usage": {"include": True}}, "top_p": 0.9})

    assert params["extra_body"] == {"reasoning": {"effort": "high"}, "usage": {"include": True}}
    assert params["top_p"] == 0.9


class _Usage:
    def __init__(self, prompt_tokens: int, details: object) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = 40
        self.total_tokens = prompt_tokens + 40
        self.prompt_tokens_details = details


class _Details:
    def __init__(self, cached_tokens: int, cache_write_tokens: int) -> None:
        self.cached_tokens = cached_tokens
        self.cache_write_tokens = cache_write_tokens


def _final_message_from_stream(usage) -> Message:
    class _Chunk:
        choices: list = []

        def __init__(self, usage_obj) -> None:
            self.usage = usage_obj

    class _Stream:
        def __init__(self) -> None:
            self._chunks = iter([_Chunk(usage)])

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self):
            return None

    async def run():
        wrapper = OpenAIStreamWrapper(_Stream(), provider_name="openrouter")
        async with wrapper:
            async for _ in wrapper:
                pass
            return await wrapper.get_final_message()

    return asyncio.run(run())


def test_streamed_cache_write_tokens_are_split_out_of_prompt_tokens() -> None:
    # OpenRouter counts cache writes inside prompt_tokens and bills them at a
    # premium; Anthropic-shaped providers report them separately, so normalize.
    message = _final_message_from_stream(_Usage(1000, _Details(cached_tokens=200, cache_write_tokens=300)))

    assert message.usage_metadata["cache_read_input_tokens"] == 200
    assert message.usage_metadata["cache_write_input_tokens"] == 300
    assert message.usage_metadata["prompt_tokens"] == 700
    assert message.usage_metadata["provider"] == "openrouter"


def test_cache_read_only_usage_leaves_prompt_tokens_alone() -> None:
    message = _final_message_from_stream(_Usage(1000, _Details(cached_tokens=200, cache_write_tokens=0)))

    assert message.usage_metadata["cache_read_input_tokens"] == 200
    assert "cache_write_input_tokens" not in message.usage_metadata
    assert message.usage_metadata["prompt_tokens"] == 1000


def test_dict_shaped_usage_details_are_read_too() -> None:
    message = _final_message_from_stream(_Usage(500, {"cached_tokens": 64, "cache_write_tokens": 128}))

    assert message.usage_metadata["cache_read_input_tokens"] == 64
    assert message.usage_metadata["cache_write_input_tokens"] == 128
    assert message.usage_metadata["prompt_tokens"] == 372


def test_generate_applies_the_same_usage_normalization(monkeypatch) -> None:
    class _Message:
        content = "done"
        reasoning = None
        tool_calls = None

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]
        usage = _Usage(900, _Details(cached_tokens=10, cache_write_tokens=400))

    class _Client:
        def __init__(self) -> None:
            self.chat = self
            self.completions = self
            self.calls: list[dict] = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return _Response()

    provider = OpenAIProvider(api_key="sk-or-test", provider_name="openrouter")
    client = _Client()
    monkeypatch.setattr(provider, "async_client", client)

    async def run():
        return await provider.generate(
            MessageHistory([]),
            params=GenerationParams(temperature=0.7, max_completion_tokens=4096),
            model=REASONING_MODEL,
        )

    message = asyncio.run(run())

    assert message.usage_metadata["cache_write_input_tokens"] == 400
    assert message.usage_metadata["prompt_tokens"] == 500
    # generate() must shape the request exactly like stream() does.
    assert "max_tokens" not in client.calls[-1]
    assert client.calls[-1]["extra_body"]["cache_control"] == {"type": "ephemeral"}
