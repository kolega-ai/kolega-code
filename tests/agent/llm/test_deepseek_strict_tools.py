"""Strict tool schemas on the Responses providers, and the DeepSeek endpoint that enforces them."""

import asyncio

import pytest

from kolega_code.llm.models import MessageHistory, ToolDefinition, ToolParameter
from kolega_code.llm.providers.deepseek_responses import DEFAULT_BASE_URL, DeepSeekResponsesProvider
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.openai_responses import OpenAIResponsesProvider
from kolega_code.llm.providers.responses_common import responses_tools

FLASH = "deepseek-flash"
STRICT_SCHEMA = {
    "type": "object",
    "properties": {"count": {"type": "integer"}, "text": {"type": "string"}},
    "required": ["count", "text"],
    "additionalProperties": False,
}


def _tool(name: str = "save_note", *, strict: bool = False) -> ToolDefinition:
    return ToolDefinition(
        name=name, description="Save a note.", parameters=[], input_schema=STRICT_SCHEMA, strict=strict
    )


class _Recorder:
    """Minimal AsyncOpenAI stand-in that captures Responses request kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Stream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def aclose(self):
                return None

        return _Stream()


def test_definitions_are_not_strict_unless_asked() -> None:
    plain = ToolDefinition(
        name="read_file",
        description="Read a file.",
        parameters=[ToolParameter(name="path", type="string", description="Path", required=True)],
    )
    assert plain.strict is False
    assert "strict" not in plain.to_public_dict()
    assert _tool(strict=True).to_public_dict()["strict"] is True


def test_strict_flag_is_sent_only_for_strict_definitions() -> None:
    tools = responses_tools(GenerationParams(tools=[_tool("a", strict=True), _tool("b")]))
    assert tools is not None
    assert tools[0] == {
        "type": "function",
        "name": "a",
        "description": "Save a note.",
        "parameters": STRICT_SCHEMA,
        "strict": True,
    }
    assert "strict" not in tools[1]


@pytest.mark.parametrize(
    "provider_cls,model", [(DeepSeekResponsesProvider, FLASH), (OpenAIResponsesProvider, "gpt-5.6-sol")]
)
def test_build_request_carries_the_strict_flag(provider_cls, model) -> None:
    provider = provider_cls(api_key="sk-test")
    request = provider._build_request(
        MessageHistory([]), None, GenerationParams(tools=[_tool(strict=True)]), {"model": model}
    )
    assert request["tools"][0]["strict"] is True


def test_build_request_omits_the_flag_for_ordinary_tools() -> None:
    provider = DeepSeekResponsesProvider(api_key="sk-test")
    request = provider._build_request(MessageHistory([]), None, GenerationParams(tools=[_tool()]), {"model": FLASH})
    assert "strict" not in request["tools"][0]


def test_deepseek_targets_the_beta_endpoint_that_enforces_strict() -> None:
    provider = DeepSeekResponsesProvider(api_key="sk-test")
    assert DEFAULT_BASE_URL == "https://api.deepseek.com/beta"
    assert provider.base_url == DEFAULT_BASE_URL
    assert str(provider.async_client.base_url).rstrip("/") == DEFAULT_BASE_URL


def test_an_explicit_base_url_is_kept() -> None:
    provider = DeepSeekResponsesProvider(api_key="sk-test", base_url="https://proxy.example/deepseek")
    assert str(provider.async_client.base_url).rstrip("/") == "https://proxy.example/deepseek"


@pytest.mark.parametrize("strict", [True, False])
def test_strict_and_ordinary_requests_use_the_same_client(strict: bool) -> None:
    provider = DeepSeekResponsesProvider(api_key="sk-test")
    recorder = _Recorder()
    provider.async_client = recorder  # type: ignore[assignment]

    async def run() -> None:
        await provider.stream(MessageHistory([]), params=GenerationParams(tools=[_tool(strict=strict)]), model=FLASH)

    asyncio.run(run())
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["tools"][0].get("strict", False) is strict
